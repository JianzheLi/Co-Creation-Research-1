"""Paired alpha=2 RUOT vs WFR+Brownian-bridge experiment across datasets/sigmas."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
from src.runtime import configure, file_hash, json_write, load_config, snapshots_for
from src.matching import make_feedback, validate_feedback_architecture
from evaluate_matching_stability import checked_checkpoint, simulate_batch

DATASETS=('gene','mouse','emt','dygen')
SIGMAS=(0.1,0.25,0.5,1.0)
OUT=ROOT/'../Results/experiment1_alpha2_sigma_sweep'


def output_root(dataset):
    return OUT


def sigma_tag(sigma):
    return str(sigma).replace('.','p')


def prepare(dataset,sigma):
    tag=sigma_tag(sigma)
    stem=f'experiment1_{dataset}_alpha2_sigma{tag}'
    parent={'gene':'gene_distance_hermite_primal.yaml',
            'mouse':'mouse_distance_hermite_primal.yaml',
            'emt':'emt_distance_hermite_primal.yaml',
            'dygen':'gene_distance_hermite_primal.yaml'}[dataset]
    run_base=(output_root(dataset).relative_to(ROOT)/dataset/f'sigma_{tag}')
    ruot=dict(extends=parent,device='cuda',seed=42,growth_penalty=2.0,model_sigma=float(sigma),
        pair=dict(hidden_width=64,hidden_layers=3,training_mode='steps',training_steps=3000,
                  cost_objective='renormalized_surrogate'),
        matching=dict(hidden_width=400,hidden_layers=6,feedback_activation='silu',
                      last_hidden_activation='tanh',steps=30000,batch_size=2048),
        coupling=dict(partition_seed=43,partition_order='dataset_default',stationarity_tolerance=0.001))
    if dataset=='dygen':
        ruot.update(dataset='wfr_fm_dygen',data_path='data/wfr_fm_dygen.csv',
                    data_url='https://raw.githubusercontent.com/QiangweiPeng/WFR-FM/main/data/dygen.csv',
                    data_sha256='63ce34c38b2ec3d25f3c40a7ffee1dfb06b213563ecc3bfb8b05c393e7409062',
                    dimension=5,model_times=[0,1,2,3,4])
        ruot['coupling'].update(mode='full')
    if dataset=='mouse':
        ruot['coupling'].update(mode='minibatch',minibatch_blocks=4)
    ruot_file=ROOT/'configs'/f'{stem}_ruot.yaml'
    ruot_file.write_text(yaml.safe_dump(ruot,sort_keys=False),encoding='utf-8')
    ruot_cfg=load_config(ruot_file)
    # Keep the same dataset, partition, alpha/sigma, feedback model, training
    # budget, rollout and evaluation settings. Only the pair/coupling method
    # changes to analytic WFR-FM and its Brownian-bridge variance heuristic.
    wfr=dict(extends=ruot_file.name,experiment_method='wfr_fm_bridge',
        comparison_run=str(run_base/'ruot_seed42'),
        pair=dict(training_mode='analytic_wfr_bridge',training_steps=0),
        coupling=dict(method='wfr_fm_oet',cost_evaluation='analytic_wfr',
                      max_mm_iterations=1000,mm_stop_tolerance=1e-15,
                      max_oet_semicoupling_relative_gap=0.001))
    wfr_file=ROOT/'configs'/f'{stem}_wfr_brownian_bridge.yaml'
    wfr_file.write_text(yaml.safe_dump(wfr,sort_keys=False),encoding='utf-8')
    wfr_cfg=load_config(wfr_file)
    for key in ('dataset','data_path','dimension','model_times','model_sigma','growth_penalty',
                'generation','matching','evaluation'):
        if ruot_cfg[key]!=wfr_cfg[key]:
            raise ValueError(f'paired configs differ at {key}: {dataset}, sigma={sigma}')
    if ruot_cfg['coupling']['mode']!=wfr_cfg['coupling']['mode']:
        raise ValueError('paired methods must use the same block partition mode')
    if dataset=='mouse' and ruot_cfg['coupling']['minibatch_blocks']!=4:
        raise ValueError('Mouse comparison must use four blocks for both methods')
    return ruot_file,wfr_file,output_root(dataset)/dataset/f'sigma_{tag}'/'ruot_seed42',output_root(dataset)/dataset/f'sigma_{tag}'/'wfr_brownian_bridge_seed42'


def check_completed(run,cfg_path):
    status_file=run/'status.json'
    if not status_file.exists():
        return False
    try:
        status=json.loads(status_file.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    if status.get('matching',{}).get('status')!='completed' or status.get('evaluation',{}).get('status')!='completed':
        return False
    cfg=load_config(cfg_path)
    saved=load_config(run/'config.yaml')
    if cfg!=saved:
        raise ValueError(f'Completed run has a different configuration: {run}')
    checked_checkpoint(run)
    return True


def paired_action(dataset,sigma,ruot_run,wfr_run):
    configs=[load_config(run/'config.yaml') for run in (ruot_run,wfr_run)]
    a,b=configs
    for key in ('dataset','data_path','dimension','model_times','model_sigma','growth_penalty','generation','matching'):
        if a[key]!=b[key]:
            raise ValueError(f'paired Action comparison differs at {key}')
    file_sets=[sorted((run/'couplings').glob('interval*.npz')) for run in (ruot_run,wfr_run)]
    if [p.name for p in file_sets[0]]!=[p.name for p in file_sets[1]]:
        raise ValueError('RUOT/WFR Coupling files differ')
    for p,q in zip(*file_sets):
        with np.load(p) as x,np.load(q) as y:
            for key in ('source_indices','target_indices','source_positions','target_positions',
                        'source_weights','target_weights'):
                np.testing.assert_array_equal(x[key],y[key])
    device=configure(a)
    if device.type!='cuda':
        raise RuntimeError('This experiment requires the available CUDA GPU')
    truth=snapshots_for(a,full=True)
    first=torch.as_tensor(truth[0].x,dtype=torch.float32,device=device)
    seeds=list(range(1000,1020))
    n=int(a['generation']['n_particles'])
    steps=(len(a['model_times'])-1)*int(a['generation']['steps_per_unit_interval'])
    x0=[];noise=[]
    for seed in seeds:
        gen=torch.Generator(device=device).manual_seed(seed+300)
        ix=torch.randint(len(first),(n,),device=device,generator=gen)
        x0.append(first[ix])
        noise.append(torch.randn((steps,n,a['dimension']),device=device,dtype=torch.float32,generator=gen))
    x0=torch.stack(x0);noise=torch.stack(noise)
    methods={}
    for name,run,cfg in (('RUOT',ruot_run,a),('WFR+Brownian bridge',wfr_run,b)):
        cp=checked_checkpoint(run)
        model=make_feedback(cfg,device)
        validate_feedback_architecture(cp,cfg)
        model.load_state_dict(cp['model']);model.eval()
        rows=simulate_batch(model,x0,noise,cfg,seeds)
        vals=np.asarray([r['total_action'] if r['finite'] else np.nan for r in rows],dtype=np.float64)
        with (run/'metrics.csv').open(encoding='utf-8-sig') as f:
            metrics=list(csv.DictReader(f))
        finite_vals=vals[np.isfinite(vals)]
        methods[name]=dict(run=str(run.resolve()),feedback_sha256=file_hash(run/'feedback.pt'),
            finite_seeds=int(len(finite_vals)),action_mean=float(finite_vals.mean()) if len(finite_vals) else None,
            action_sd=float(finite_vals.std(ddof=1)) if len(finite_vals)>1 else None,
            action_median=float(np.median(finite_vals)) if len(finite_vals) else None,
            W1_mean=float(np.mean([float(m['W1_normalized']) for m in metrics])),
            W2_mean=float(np.mean([float(m['W2_normalized']) for m in metrics])),
            mass_relative_error_mean=float(np.mean([float(m['mass_relative_error']) for m in metrics])),
            replicates=rows)
        del model
        torch.cuda.empty_cache()
    ruot=np.asarray([v['total_action'] for v in methods['RUOT']['replicates']])
    wfr=np.asarray([v['total_action'] for v in methods['WFR+Brownian bridge']['replicates']])
    finite_pair=np.isfinite(wfr)&np.isfinite(ruot)
    diff=wfr[finite_pair]-ruot[finite_pair]
    npaired=int(finite_pair.sum())
    paired=dict(paired_finite_seeds=npaired,WFR_minus_RUOT_mean=float(diff.mean()) if npaired else None,
        standard_error=float(diff.std(ddof=1)/np.sqrt(npaired)) if npaired>1 else None,
        relative_percent=(float(100*(methods['WFR+Brownian bridge']['action_mean']/methods['RUOT']['action_mean']-1))
                          if methods['RUOT']['action_mean'] not in (None,0) and methods['WFR+Brownian bridge']['action_mean'] is not None else None),
        seeds_RUOT_lower=int((diff>0).sum()),seeds_tied=int((diff==0).sum()),
        seeds_WFR_lower=int((diff<0).sum()))
    result=dict(dataset=dataset,alpha=2.0,sigma=float(sigma),training_seed=42,
        action_definition='20-seed paired Monte Carlo of the learned RUOT feedback Action; identical initial samples and Brownian increments',
        methods=methods,paired_action_difference=paired,
        settings=dict(pair_path='RUOT 64x3 MLP, 3000 GPU steps; WFR analytic traveling mean/mass and sigma^2*t*(1-t) Brownian-bridge variance',
            feedback='separate V/G/Score MLPs, width 400, 6 hidden layers, SiLU with Tanh on penultimate hidden activations, 30000 GPU steps',
            coupling='RUOT uses sigma^2 entropy and renormalized pair cost; WFR uses alpha-scaled WFR-FM OET with delta^2=alpha and identical block partition',
            mouse_blocks=4 if dataset=='mouse' else None,
            seed42_single_training_run=True,rollout_seeds=seeds,
            rollout_particles=int(a['generation']['n_particles']),steps_per_unit_interval=int(a['generation']['steps_per_unit_interval'])))
    out=output_root(dataset)/dataset/f'sigma_{sigma_tag(sigma)}';out.mkdir(parents=True,exist_ok=True)
    json_write(out/'comparison_20seed.json',result)
    rows=[]
    for key,m in methods.items():
        for r in m['replicates']:
            rows.append(dict(method=key,**r))
    with (out/'action_replicates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='methods'},ensure_ascii=False),flush=True)
    return result


def run_one(dataset,sigma,prepare_only=False):
    ruot_config,wfr_config,ruot_run,wfr_run=prepare(dataset,sigma)
    if prepare_only:
        return dict(dataset=dataset,sigma=sigma,status='config_validated',ruot_config=str(ruot_config),wfr_config=str(wfr_config))
    case_dir=output_root(dataset)/dataset/f'sigma_{sigma_tag(sigma)}';case_dir.mkdir(parents=True,exist_ok=True)
    state_file=case_dir/'suite_status.json'
    comparison_file=case_dir/'comparison_20seed.json'
    if comparison_file.exists() and check_completed(ruot_run,ruot_config) and check_completed(wfr_run,wfr_config):
        try:
            saved=json.loads(comparison_file.read_text(encoding='utf-8'))
            matches=(saved.get('dataset')==dataset and saved.get('alpha')==2.0
                     and saved.get('sigma')==float(sigma)
                     and saved.get('methods',{}).get('RUOT',{}).get('feedback_sha256')==file_hash(ruot_run/'feedback.pt')
                     and saved.get('methods',{}).get('WFR+Brownian bridge',{}).get('feedback_sha256')==file_hash(wfr_run/'feedback.pt'))
        except (OSError, json.JSONDecodeError, KeyError):
            matches=False
        if matches:
            state=dict(dataset=dataset,sigma=sigma,status='completed',comparison=str(comparison_file.resolve()),reused_comparison=True,
                       paired_difference=saved.get('paired_action_difference'))
            json_write(state_file,state)
            print(f'Reusing completed {dataset} sigma={sigma:g} comparison',flush=True)
            return state
    state=dict(dataset=dataset,sigma=sigma,status='running',ruot_run=str(ruot_run),wfr_run=str(wfr_run),stages=[])
    json_write(state_file,state)
    failed_methods=[]
    for method,config,run,script in (
        ('RUOT',ruot_config,ruot_run,ROOT/'scripts/run_experiment.py'),
        ('WFR+Brownian bridge',wfr_config,wfr_run,ROOT/'scripts/run_wfr_heuristic.py')):
        if check_completed(run,config):
            state['stages'].append(dict(method=method,status='reused_completed'))
            json_write(state_file,state);continue
        log_path=case_dir/('ruot.log' if method=='RUOT' else 'wfr_brownian_bridge.log')
        command=[sys.executable,str(script),'--config',str(config.resolve()),'--run-dir',str(run.resolve())]
        print(f'Starting {dataset} sigma={sigma:g}: {method}',flush=True)
        start=time.perf_counter()
        with log_path.open('a',encoding='utf-8') as log:
            proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        stage=dict(method=method,status='completed' if proc.returncode==0 else 'failed',
                   return_code=proc.returncode,seconds=time.perf_counter()-start,log=str(log_path.resolve()))
        if proc.returncode and check_completed(run,config):
            # Evaluation/reporting may finish before a diagnostic plot fails
            # on a damaged JSONL history file. Preserve valid model metrics.
            stage['status']='completed_with_post_evaluation_error'
            stage['reused_valid_checkpoints']=True
        state['stages'].append(stage);json_write(state_file,state)
        if proc.returncode and stage['status']=='failed':
            failed_methods.append(method)
    if failed_methods:
        state['status']='methods_incomplete';state['failed_methods']=failed_methods
        json_write(state_file,state);print(json.dumps(state,ensure_ascii=False),flush=True)
        return state
    try:
        result=paired_action(dataset,sigma,ruot_run,wfr_run)
    except Exception as exc:
        state.update(status='comparison_failed',error=repr(exc));json_write(state_file,state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
        return state
    state.update(status='completed',comparison=str((case_dir/'comparison_20seed.json').resolve()),
                 paired_difference=result['paired_action_difference'])
    json_write(state_file,state)
    return state


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=DATASETS)
    p.add_argument('--all-datasets',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--sigmas',nargs='+',type=float,default=list(SIGMAS))
    a=p.parse_args()
    if not a.all_datasets and a.dataset is None:
        p.error('specify --dataset or --all-datasets')
    if any(s not in SIGMAS for s in a.sigmas):
        p.error(f'sigmas must be selected from {SIGMAS}')
    datasets=DATASETS if a.all_datasets else (a.dataset,)
    result=[]
    for dataset in datasets:
        for sigma in a.sigmas:
            result.append(run_one(dataset,sigma,a.prepare_only))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

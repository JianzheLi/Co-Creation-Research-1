"""Fixed, paired experimental comparison. No parameter or coefficient search.

One worker per dataset; each reruns both algorithms with the same architectures,
training budgets, block partitions and rollout random numbers.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys
import time

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config, configure, snapshots_for, json_write, file_hash
from src.matching import make_feedback, validate_feedback_architecture
from evaluate_matching_stability import simulate_batch, checked_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = 'gaussian_reference'
OUT = ROOT/'../Results/gaussian_reference_comparison'


def prepare(dataset, setting, method):
    alpha, sigma = {'a2_s1': (2., 1.), 'a10_s025': (10., .25)}[setting]
    name = f'{dataset}_{method}_64x3_400x6_{setting}'
    config = dict(extends=f'{dataset}_distance_hermite_primal.yaml', device='cuda',
        growth_penalty=alpha, model_sigma=sigma,
        pair=dict(hidden_width=64, hidden_layers=3, training_steps=3000),
        matching=dict(hidden_width=400, hidden_layers=6, feedback_activation='silu',
                      last_hidden_activation='tanh'))
    if method != 'original':
        config['pair']['cost_objective'] = method+'_upper_bound'
        config['coupling'] = dict(reference_weights='source_target_product')
    file = ROOT/'configs'/f'{name}.yaml'
    if file.exists() and yaml.safe_load(file.read_text()) != config:
        raise ValueError(f'Existing experiment configuration differs: {file}')
    file.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    # Validate and derive reference metadata before launching anything.
    load_config(file)
    return file, ROOT/'../Results'/f'{name}_seed42'


def evaluate_pair(dataset, setting, runs):
    result_file = OUT/f'{dataset}_{setting}_20seed.json'
    configs = [load_config(run/'config.yaml') for run in runs]
    for key in ['dimension','model_times','model_sigma','growth_penalty','generation','matching','data_path']:
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'Unpaired comparison: {key}')
    left, right = [sorted((run/'couplings').glob('interval*.npz')) for run in runs]
    if [p.name for p in left] != [p.name for p in right]:
        raise ValueError('Unpaired coupling block support')
    for a, b in zip(left, right):
        with np.load(a) as x, np.load(b) as y:
            for key in ['source_indices','target_indices','source_positions','target_positions','source_weights','target_weights']:
                np.testing.assert_array_equal(x[key], y[key])
    cfg = configs[0]
    device = configure(cfg)
    seeds = list(range(1000,1020))
    truth = snapshots_for(cfg, full=True)
    first = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)
    particles = cfg['generation']['n_particles']
    steps = (len(cfg['model_times'])-1)*cfg['generation']['steps_per_unit_interval']
    xs, noises = [], []
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(seed+300)
        xs.append(first[torch.randint(len(first), (particles,), device=device, generator=gen)])
        noises.append(torch.randn((steps,particles,cfg['dimension']),device=device,generator=gen))
    xs, noises = torch.stack(xs), torch.stack(noises)
    result = dict(dataset=dataset, setting=setting, alpha=cfg['growth_penalty'], sigma=cfg['model_sigma'],
                  seeds=seeds, definition='paired raw weighted SDE Action; identical Brownian increments and initial particles',
                  training_seed=cfg['seed'], notes='20 rollout seeds, not 20 independently trained networks', methods={})
    for method,run,config in zip(['original',CANDIDATE],runs,configs):
        cp = checked_checkpoint(run)
        model = make_feedback(config,device)
        validate_feedback_architecture(cp,config)
        model.load_state_dict(cp['model']); model.eval()
        rows = simulate_batch(model,xs,noises,config,seeds)
        values = [r['total_action'] for r in rows if r['finite']]
        with (run/'metrics.csv').open(encoding='utf-8-sig') as f:
            metrics = list(csv.DictReader(f))[1:]
        result['methods'][method] = dict(run=str(run),feedback_sha256=file_hash(run/'feedback.pt'),
            finite_seeds=len(values),action_mean=float(np.mean(values)) if values else None,
            action_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
            mean_W1=float(np.mean([float(m['W1_normalized']) for m in metrics])),
            mean_mass_error=float(np.mean([float(m['mass_relative_error']) for m in metrics])),
            replicates=rows)
        del model
    if all(m['finite_seeds']==20 for m in result['methods'].values()):
        old,new = [result['methods'][k] for k in ['original',CANDIDATE]]
        differences = np.array([b['total_action']-a['total_action'] for a,b in zip(old['replicates'],new['replicates'])])
        result['paired_difference'] = dict(mean=float(differences.mean()),
            standard_error=float(differences.std(ddof=1)/np.sqrt(20)),
            relative_percent=float(100*(new['action_mean']/old['action_mean']-1)))
    json_write(result_file,result)
    print(json.dumps({k:v for k,v in result.items() if k!='methods'}),flush=True)
    return result


def main():
    global CANDIDATE, OUT
    p = argparse.ArgumentParser()
    p.add_argument('--candidate',choices=['gaussian_reference','diffusion_full_square'],default='gaussian_reference')
    p.add_argument('--dataset',required=True,choices=['gene','mouse','emt'])
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--settings',nargs='+',default=['a2_s1','a10_s025'])
    args = p.parse_args()
    CANDIDATE = args.candidate
    OUT = ROOT/'../Results'/f'{CANDIDATE}_comparison'
    OUT.mkdir(exist_ok=True)
    for setting in args.settings:
        cases = [prepare(args.dataset,setting,method) for method in ['original',CANDIDATE]]
        if args.prepare_only:
            continue
        for config,run in cases:
            started = time.perf_counter()
            if (run/'status.json').exists():
                status = json.loads((run/'status.json').read_text(encoding='utf-8'))
                if status.get('evaluation',{}).get('status') == 'completed':
                    checked_checkpoint(run)
                    if load_config(config) != load_config(run/'config.yaml'):
                        raise ValueError('Completed experiment configuration differs')
                    print(f'Reusing completed {run.name}',flush=True)
                    continue
            print(f'Starting {run.name}',flush=True)
            with (OUT/f'{run.name}.log').open('a',encoding='utf-8') as log:
                completed = subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),
                    '--config',str(config),'--run-dir',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f'{run.name} failed; inspect the experiment log')
            print(f'Completed {run.name}, seconds={time.perf_counter()-started:.2f}',flush=True)
        evaluate_pair(args.dataset,setting,[r for _,r in cases])


if __name__=='__main__':
    main()

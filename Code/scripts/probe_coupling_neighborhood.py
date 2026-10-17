"""Feasible perturbations only; no local optimization or network training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DeepRUOT/Code: src/
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch

from diagnose_local_coupling import PosteriorObjective
from src.primal_coupling_gpu import kl_project
from src.coupling import OracleError
from src.runtime import json_write


def project_direction(obj,gp,gq):
    step=1.
    for _ in range(20):
        try:
            lp,lq,_=kl_project(obj.P.log()-step*gp,obj.Q.log()-step*gq,obj.a,obj.b,
                obj.cfg['pair']['ratio_min'],obj.cfg['pair']['ratio_max'])
            return lp.exp(),lq.exp(),step
        except OracleError:
            step*=.5
    raise RuntimeError('Could not construct a feasible perturbation')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['gene','mouse','emt'],required=True)
    ap.add_argument('--nodes',type=int,default=8);ap.add_argument('--points',type=int,default=1)
    ap.add_argument('--random-directions',type=int,default=4)
    ap.add_argument('--random-only',action='store_true')
    ap.add_argument('--seed-start',type=int,default=8100)
    ap.add_argument('--validation-per-scale',type=int,default=0)
    ap.add_argument('--output-root',type=Path,default=Path('../Results/coupling_neighborhood'))
    args=ap.parse_args();ds=args.dataset
    run=Path('../Results')/f'{ds}_original_64x3_400x6_a10_s025_seed42'
    block='interval_0_block_0.npz' if ds=='mouse' else 'interval_0.npz'
    out=args.output_root/ds;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    obj=PosteriorObjective(run,block,args.nodes,args.points,7300,1,'all_components')
    _,gp,gq,base=obj.evaluate(obj.P,obj.Q,True,not args.random_only)
    result=dict(dataset=ds,run=str(run),block=block,shape=list(obj.P.shape),
        alpha=obj.alpha,sigma=obj.sigma,screening_nodes=args.nodes,samples_per_component=args.points,
        baseline=base,candidates=[],random_directions=args.random_directions,seed_start=args.seed_start,
        random_only=args.random_only,validation_per_scale=args.validation_per_scale,
        scope='Frozen path and control rule; perturb only the first saved coupling block; all other blocks fixed. Every endpoint component is enumerated.',
        objective='Cren inner product + receiver conditional entropy + actual spatial posterior remainder; equivalent to source-entropy objective plus all three corrections up to candidate-independent constants.',
        no_claim='Finite perturbation evidence only; not a global optimum or continuous Action certificate.')
    directions=[]
    for seed in range(args.seed_start,args.seed_start+args.random_directions):
        gen=torch.Generator(device=obj.device).manual_seed(seed)
        zp=torch.randn(obj.P.shape,device=obj.device,dtype=obj.P.dtype,generator=gen)
        zq=torch.randn(obj.Q.shape,device=obj.device,dtype=obj.Q.dtype,generator=gen)
        p,q,step=project_direction(obj,zp,zq)
        directions.append((f'random_{seed}',p,q,step))
    if not args.random_only:
        gp-=((obj.P*gp).sum(1)/obj.a)[:,None]
        gq-=((obj.Q*gq).sum(0)/obj.b)[None]
        rms=((obj.P*gp.square()).sum()/obj.a.sum()+(obj.Q*gq.square()).sum()/obj.b.sum()).sqrt()
        p,q,step=project_direction(obj,gp/rms,gq/rms)
        directions.append(('negative_corrected_gradient',p,q,step))
        alternate=Path('../Results')/f'{ds}_diffusion_full_square_64x3_400x6_a10_s025_seed42'/'couplings'/block
        with np.load(alternate) as a:
            for key in ['source_positions','target_positions','source_weights','target_weights']:
                np.testing.assert_array_equal(a[key],getattr(obj,key).cpu().numpy())
            p=torch.as_tensor(a['P'],device=obj.device,dtype=obj.P.dtype)
            q=torch.as_tensor(a['Q'],device=obj.device,dtype=obj.Q.dtype)
        directions.append(('toward_bound_coupling_only',p,q,None))
    kept={};base_F=base['pair']+base['source_entropy']
    for name,p1,q1,step in directions:
        extent=max(obj.changes(p1,q1)[k] for k in ['P_relative_TV','Q_relative_TV'])
        last_fraction=None
        for radius in [.001,.01,.05]:
            fraction=min(1.,radius/extent)
            if fraction==last_fraction:continue
            last_fraction=fraction
            p=(1-fraction)*obj.P+fraction*p1;q=(1-fraction)*obj.Q+fraction*q1
            details=obj.evaluate(p,q,True)[3]
            delta=details['objective']-base['objective']
            deltaF=details['pair']+details['source_entropy']-base_F
            key=f'{name}_tv{radius:g}'
            row=dict(id=key,direction=name,requested_max_TV=radius,mixture_fraction=fraction,
                construction_step=step,**obj.changes(p,q),**details,
                delta_corrected=delta,delta_retained=deltaF,delta_dropped=delta-deltaF)
            if args.random_only:
                assert abs(max(row['P_relative_TV'],row['Q_relative_TV'])-radius)<1e-8
            assert max(row['row_relative_l1'],row['column_relative_l1'])<1e-7
            ratio=q/p
            assert bool((ratio>=obj.cfg['pair']['ratio_min']-1e-7).all())
            assert bool((ratio<=obj.cfg['pair']['ratio_max']+1e-7).all())
            result['candidates'].append(row);kept[key]=(p,q)
            json_write(out/'results.json',dict(**result,status='screening'))
            print(json.dumps(row),flush=True)
    # Select before independent verification; optionally cover each tested scale.
    if args.validation_per_scale:
        chosen=[]
        for radius in [.001,.01,.05]:
            group=[v for v in result['candidates'] if v['requested_max_TV']==radius]
            chosen.extend(sorted(group,key=lambda v:v['delta_corrected'])[:args.validation_per_scale])
    else:
        chosen=sorted(result['candidates'],key=lambda v:v['delta_corrected'])[:2]
    result['validation_candidates']=[v['id'] for v in chosen]
    result['validation']=[]
    del obj
    for seed in [7400,7401]:
        check=PosteriorObjective(run,block,args.nodes*2,args.points*2,seed,1,'all_components')
        baseline=check.evaluate(check.P,check.Q,True)[3]
        values=[]
        for record in chosen:
            p,q=kept[record['id']];details=check.evaluate(p,q,True)[3]
            values.append(dict(id=record['id'],**details,delta_corrected=details['objective']-baseline['objective']))
        result['validation'].append(dict(seed=seed,nodes=args.nodes*2,
            samples_per_component=args.points*2,baseline=baseline,candidates=values))
        json_write(out/'results.json',dict(**result,status='validation'))
        print(json.dumps(result['validation'][-1]),flush=True)
        del check
    result['seconds']=time.perf_counter()-start;result['status']='completed'
    json_write(out/'results.json',result)
    torch.save({k:{'P':p.cpu(),'Q':q.cpu()} for k,(p,q) in kept.items()},out/'perturbations.pt')
    print(json.dumps(dict(dataset=ds,status='completed',seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()

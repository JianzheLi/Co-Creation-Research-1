"""Predeclared five-dimensional candidate sweep; no training or selection by R.

Each dataset uses 128 scrambled Sobol points, 32 cube vertices and 3 interior
diagonal points. Couplings and endpoint-compatible Gaussian mean, covariance,
mass and compatible controls are independently varied. Common random numbers
are used within each independent Monte Carlo replicate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DeepRUOT/Code: src/
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import torch
from audit_posterior_corrections import ROOT,candidate,estimate
from src.runtime import configure,load_config,json_write

FIELDS=['entropy_correction','eta_correction','growth_correction','total_correction','surrogate']


def design():
    corners=np.array(list(itertools.product([0.,1.],repeat=5)))
    sobol=torch.quasirandom.SobolEngine(5,scramble=True,seed=314159).draw_base2(7).double().numpy()
    points=np.concatenate([corners,sobol,np.array([[x]*5 for x in (.25,.5,.75)])])
    return [dict(id=f'c{i:03}',coordinates=p.tolist(),kind='corner' if i<32 else 'sobol' if i<160 else 'diagonal') for i,p in enumerate(points)]


def selected_extremes(folder,points):
    records=[]
    for point in points:
        file=folder/f"{point['id']}_screen.json"
        if not file.exists():
            raise ValueError(f'Screening is incomplete: {file}')
        records.append(json.loads(file.read_text(encoding='utf-8')))
    indices={0,31}
    for field in FIELDS:
        values=[np.mean([r[field] for r in rec['replicates']]) for rec in records]
        indices.update([int(np.argmin(values)),int(np.argmax(values))])
    return [points[i] for i in sorted(indices)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=['gene','mouse','emt'])
    parser.add_argument('--stage',default='screen',choices=['screen','confirm','timecheck'])
    parser.add_argument('--ids',nargs='*')
    args=parser.parse_args()
    out=ROOT/'../Results/posterior_candidate_sweep'/args.dataset
    out.mkdir(parents=True,exist_ok=True)
    points=design()
    json_write(out/'design.json',dict(coordinate_order=['coupling','mean_path','covariance_path','mass_path','compatible_control'],
        points=points,selection='Entire predeclared cube: no acceptance filtering by objective or correction.'))
    if args.stage=='screen':
        seeds=[6000,6001]; rows,spatial,nodes=32,16,16
    else:
        points=selected_extremes(out,points)
        seeds=[7000,7001,7002,7003]; rows,spatial,nodes=128,32,32
        if args.stage=='timecheck':
            seeds=[7000]; nodes=64
    if args.ids:
        points=[p for p in points if p['id'] in args.ids]
    device=configure(load_config(f'configs/{args.dataset}_tight_bound.yaml'))
    assert device.type=='cuda'
    cache={}; started=time.perf_counter()
    for point in points:
        dest=out/f"{point['id']}_{args.stage}.json"
        if dest.exists():
            continue
        _,path,blocks,provenance=candidate(args.dataset,0.,device,coordinates=point['coordinates'],cache=cache)
        values=[]
        for seed in seeds:
            value=estimate(path,blocks,point['coordinates'][-1],seed,nodes,rows,spatial)
            values.append(dict(seed=seed,**value))
        json_write(dest,dict(dataset=args.dataset,**point,stage=args.stage,seeds=seeds,rows_per_interval=rows,
            points_per_row=spatial,time_nodes=nodes,provenance=provenance,replicates=values,
            interpretation='Fixed discrete endpoint labels, interior Gaussian posteriors; not learned-feedback Action.'))
        print(json.dumps(dict(dataset=args.dataset,id=point['id'],stage=args.stage,seconds=time.perf_counter()-started,
            **{key:float(np.mean([r[key] for r in values])) for key in FIELDS})),flush=True)
        del path,blocks


if __name__=='__main__':
    main()

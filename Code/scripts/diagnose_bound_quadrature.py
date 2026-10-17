"""Compare fixed standard quadrature maps on the same frozen path/conditions."""
import sys
from pathlib import Path
import json
import math
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config,load_path,configure,snapshots_for,json_write
from src.pair_cost import CostOracle

run=Path(sys.argv[1])
cfg=load_config(run/'config.yaml');device=configure(cfg);path=load_path(run,cfg,device)
snaps=snapshots_for(cfg);rng=np.random.default_rng(cfg['seed']+111)
delta=[]
for _ in range(cfg['pair']['validation_conditions']):
    k=rng.integers(len(snaps)-1)
    delta.append(snaps[k+1].x[rng.integers(len(snaps[k+1].x))]-snaps[k].x[rng.integers(len(snaps[k].x))])
delta=np.asarray(delta)
ratios=np.exp(rng.uniform(math.log(cfg['pair']['ratio_min']),math.log(cfg['pair']['ratio_max']),len(delta)))

def evaluate(nodes,kind):
    oracle=CostOracle(path,nodes,128)
    if kind!='terminal':
        x,w=np.polynomial.legendre.leggauss(nodes)
        u=torch.tensor((x+1)/2,dtype=torch.float64,device=device)
        weights=torch.tensor(w/2,dtype=u.dtype,device=device)
        if kind=='both':
            oracle.t=torch.sin(math.pi/2*u).square()
            oracle.w=weights*(math.pi/2)*torch.sin(math.pi*u)
        else:
            oracle.t=u;oracle.w=weights
    return oracle(delta,ratios)

reference=evaluate(512,'both')
rows=[]
for kind in ['terminal','both','plain']:
    for n in [32,64,128,256]:
        actual=evaluate(n,kind)
        error=[float(np.max(abs(a-b))) for a,b in zip(actual,reference)]
        norm=[float(np.max(abs(a-b)/(cfg['pair']['query_atol']+cfg['pair']['query_rtol']*abs(b)))) for a,b in zip(actual,reference)]
        row=dict(map=kind,nodes=n,max_errors=error,max_normalized_errors=norm)
        rows.append(row);print(json.dumps(row),flush=True)
json_write(run/'quadrature_diagnostic.json',dict(reference_nodes=512,reference_map='both',comparisons=rows))

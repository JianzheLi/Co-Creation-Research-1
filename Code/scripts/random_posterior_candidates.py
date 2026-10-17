"""Small broad random stress test, independent of trained-weight interpolation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code: src/
import sys,json,time,math
from pathlib import Path
import numpy as np
import torch
from audit_posterior_corrections import ROOT,candidate,estimate
from src.runtime import configure,load_config,json_write
from src.primal_coupling_gpu import kl_project


@torch.no_grad()
def randomize_path(path,seed,gain):
    generator=torch.Generator(device=next(path.parameters()).device).manual_seed(seed)
    for network in (path.mu_net,path.var_net,path.mass_net):
        for layer in list(network.hidden)+[network.last]:
            layer.weight.copy_(torch.randn(layer.weight.shape,dtype=layer.weight.dtype,device=layer.weight.device,generator=generator)*gain/math.sqrt(layer.in_features))
            layer.bias.copy_(torch.randn(layer.bias.shape,dtype=layer.bias.dtype,device=layer.bias.device,generator=generator)*gain*.2)


@torch.no_grad()
def randomize_couplings(blocks,seed,scale):
    generator=torch.Generator(device=blocks[0]['P'].device).manual_seed(seed)
    errors=[]
    for block in blocks:
        a,b=block['a'],block['b']; shape=block['P'].shape
        reference=a.log()[:,None]+b.log()[None,:]-b.sum().log()
        lp=reference+scale*torch.randn(shape,dtype=a.dtype,device=a.device,generator=generator)
        lq=reference+(b.sum()/a.sum()).log()+scale*torch.randn(shape,dtype=a.dtype,device=a.device,generator=generator)
        lp,lq,iterations=kl_project(lp,lq,a,b,.1,10.,tolerance=1e-8)
        block.update(P=lp.exp(),Q=lq.exp(),logr=lq-lp)
        error=max(float((block['P'].sum(1)-a).abs().sum()/a.sum()),float((block['Q'].sum(0)-b).abs().sum()/b.sum()))
        assert error<=1.01e-8 and float(block['logr'].min())>=math.log(.1)-1e-10 and float(block['logr'].max())<=math.log(10)+1e-10
        errors.append(dict(block=block['name'],relative_marginal_error=error,projection_iterations=iterations,
            ratio_min=float(block['logr'].min().exp()),ratio_max=float(block['logr'].max().exp())))
    return errors


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['gene','mouse','emt'],required=True)
    args=parser.parse_args();dataset=args.dataset
    device=configure(load_config(ROOT/f'configs/{dataset}_tight_bound.yaml'))
    assert device.type=='cuda'
    out=ROOT/'../Results/random_posterior_candidates'/dataset;out.mkdir(parents=True,exist_ok=True)
    specs=[dict(id='baseline',group='baseline',seed=9100,coupling_scale=0.,path_gain=0.)]
    for group in ('random_coupling','random_path','both_random'):
        for i,(scale,gain) in enumerate(((1.,.5),(3.,.5),(1.,1.5),(3.,1.5))):
            specs.append(dict(id=f'{group}_{i}',group=group,seed=9100+i,coupling_scale=scale,path_gain=gain))
    json_write(out/'design.json',dict(candidates=specs,mc_seeds=[8100,8101],rows=32,points=16,time_nodes=16,
        controls='Original compatible controls for each candidate density; no control interpolation.',
        coupling='Independent log-normal P,Q, KL projection onto row(P)=a, column(Q)=b, 0.1<=Q/P<=10; no transport cost used.',
        path='All weights N(0,gain^2/fan_in), all biases N(0,(0.2 gain)^2), including output layers. Original bounded endpoint-compatible architecture.',
        interpretation='Broad random stress test, not near-optimal or global-uniform sampling.'))
    cache={};start=time.perf_counter()
    for spec in specs:
        dest=out/f"{spec['id']}.json"
        if dest.exists():continue
        _,path,blocks,provenance=candidate(dataset,0.,device,cache=cache)
        diagnostics=[]
        if spec['group'] in ('random_coupling','both_random'):
            diagnostics=randomize_couplings(blocks,spec['seed'],spec['coupling_scale'])
        if spec['group'] in ('random_path','both_random'):
            randomize_path(path,spec['seed']+100,spec['path_gain'])
        reps=[dict(seed=seed,**estimate(path,blocks,0.,seed,16,32,16)) for seed in (8100,8101)]
        json_write(dest,dict(dataset=dataset,**spec,provenance=provenance,feasibility=diagnostics,replicates=reps))
        print(json.dumps(dict(dataset=dataset,id=spec['id'],seconds=time.perf_counter()-start,F=np.mean([r['surrogate'] for r in reps]),R=np.mean([r['total_correction'] for r in reps]))),flush=True)
        del path,blocks


if __name__=='__main__':main()

"""Diagnostic variation of the three corrections along feasible candidates.

This is a discrete-label posterior diagnostic, NOT an exact evaluation of the
continuous-endpoint renormalization identity or the learned feedback Action.
P,Q are interpolated in their convex feasible set; path weights are interpolated
within the endpoint-compatible family; two compatible controls are interpolated
for that SAME intermediate density. No training is changed.
"""
from pathlib import Path
import sys,json,math,argparse,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.runtime import load_config,configure,make_path,json_write,file_hash
from src.pair_path import exprel,spectral_apply

ROOT=Path(__file__).resolve().parents[1]


def apply(axis,eigen,y):
    return eigen[...,-1:]*y+(eigen[...,:1]-eigen[...,-1:])*axis*(axis*y).sum(-1,keepdim=True)


def candidate(dataset,theta,device,coordinates=None,cache=None):
    # Five independently adjustable coordinates: coupling, mean, covariance,
    # mass and compatible control. All share the original and tight endpoints.
    coordinates=[theta]*5 if coordinates is None else coordinates
    coupling,mean,covariance,mass,control=coordinates
    old=ROOT/f'../Results/{dataset}_distance_hermite_primal_seed42'
    new=ROOT/f'../Results/{dataset}_tight_bound_seed42'
    cfg=load_config(new/'config.yaml')
    path=make_path(cfg,device)
    if cache is None:
        cache={}
    if not cache:
        cache['c0']=torch.load(old/'pair_path.pt',map_location=device,weights_only=False)['model']
        cache['c1']=torch.load(new/'pair_path.pt',map_location=device,weights_only=False)['model']
        cache['arrays']=[]
        for filename in sorted((old/'couplings').glob('interval*.npz')):
            with np.load(filename) as a,np.load(new/'couplings'/filename.name) as b:
                cache['arrays'].append((filename.name,dict(a),dict(b)))
        cache['provenance']={str(run.relative_to(ROOT)):file_hash(run/'pair_path.pt') for run in (old,new)}
    c0,c1=cache['c0'],cache['c1']
    coefficient={'mu_net':mean,'var_net':covariance,'mass_net':mass}
    path.load_state_dict({k:(1-coefficient[k.split('.')[0]])*c0[k]+coefficient[k.split('.')[0]]*c1[k] for k in c0}); path.freeze()
    blocks=[]
    for filename,a,b in cache['arrays']:
        for key in ('source_indices','target_indices','source_weights','target_weights','source_positions','target_positions'):
            np.testing.assert_array_equal(a[key],b[key])
        T=lambda v:torch.as_tensor(v,dtype=torch.float64,device=device)
        P=T((1-coupling)*a['P']+coupling*b['P']); Q=T((1-coupling)*a['Q']+coupling*b['Q'])
        ratio=torch.where(P>0,Q/P.clamp_min(torch.finfo(P.dtype).tiny),torch.ones_like(P)).clamp(.1,10)
        weights=T(a['source_weights']); targets=T(a['target_weights'])
        assert (P.sum(1)-weights).abs().sum()/weights.sum()<.002
        assert (Q.sum(0)-targets).abs().sum()/targets.sum()<.002
        blocks.append(dict(P=P,Q=Q,logr=ratio.log(),a=weights,b=targets,
            x0=T(a['source_positions']),x1=T(a['target_positions']),interval=int(a['interval']),name=filename))
    return cfg,path,blocks,cache['provenance']


def compatible_coefficients(state,theta,alpha):
    return ((1-theta)*(-state['b'])+theta*state['drift_mean'],
            (1-theta)*(-state['K'])+theta*state['drift_times_var'],
            (1-theta)*(-state['b']/alpha)+theta*state['growth_linear'],
            (1-theta)*(-state['K']/alpha)+theta*state['growth_quadratic'])


@torch.no_grad()
def estimate(path,blocks,theta,seed,time_nodes,rows_per_interval,points):
    device=next(path.parameters()).device; sigma=path.sigma; alpha=path.alpha; dim=path.dimension
    generator=torch.Generator(device=device).manual_seed(seed)
    gx,gw=np.polynomial.legendre.leggauss(time_nodes)
    times=(gx+1)/2; weights=gw/2
    # Integrals: spatial entropy, raw entropy, eta variance, growth variance,
    # their posterior-free upper bounds, renormalized running integrand, mass.
    sums=torch.zeros(8,dtype=torch.float64,device=device)
    boundary=torch.zeros((),dtype=torch.float64,device=device)
    base=torch.zeros_like(boundary); entropy_source=torch.zeros_like(boundary)
    interval_counts={k:sum(b['interval']==k for b in blocks) for k in {b['interval'] for b in blocks}}
    for block in blocks:
        P,Q,a=block['P'],block['Q'],block['a']
        logprior=P.clamp_min(torch.finfo(P.dtype).tiny).log()-a[:,None].log()
        boundary+=sigma**2*((Q-P)*logprior).sum()
        distances=(block['x1'][None,:,:]-block['x0'][:,None,:]).square().sum(-1)
        base+=(.5*P*distances).sum()-dim*sigma**2/2*Q.sum()
        entropy_source+=sigma**2*(P*P.clamp_min(torch.finfo(P.dtype).tiny).log()).sum()
        nrows=max(4,math.ceil(rows_per_interval/interval_counts[block['interval']]))
        # Draw and reuse sources/noise at all quadrature nodes; this provides
        # common random numbers between candidates and between time rules.
        ids=torch.multinomial(a,nrows,replacement=True,generator=generator)
        uniforms=torch.rand((nrows,points),dtype=P.dtype,device=device,generator=generator)
        noise=torch.randn((nrows,points,dim),dtype=P.dtype,device=device,generator=generator)
        for t,weight in zip(times,weights):
            for start in range(0,nrows,8):
                index=ids[start:start+8]; R=len(index); J=P.shape[1]
                delta=block['x1'][None,:,:]-block['x0'][index,None,:]
                ell=block['logr'][index]
                s=path(torch.full((R*J,1),float(t),dtype=P.dtype,device=device),delta.reshape(-1,dim),ell.reshape(-1,1))
                vec=lambda k:s[k].reshape(R,J,dim)
                scalar=lambda k:s[k].reshape(R,J)
                var=vec('var'); axis=vec('axis'); mu=vec('xi'); gamma=scalar('gamma')
                drift,drift_var,beta,zeta=[v.reshape(R,J,dim) for v in compatible_coefficients(s,theta,alpha)]
                W=P[index]*scalar('mass'); mass=W.sum(1); prior_t=W/mass[:,None]
                cdf=prior_t.cumsum(1); cdf[:,-1]=1
                chosen=torch.searchsorted(cdf.contiguous(),uniforms[start:start+R].contiguous()).clamp_max(J-1)
                row=torch.arange(R,device=device)[:,None]
                x=mu[row,chosen]+apply(axis[row,chosen],var[row,chosen].sqrt(),noise[start:start+R])
                y=x[:,:,None,:]-mu[:,None,:,:]
                ax=axis[:,None,:,:]; sv=var[:,None,:,:]
                logphi=-.5*(var.log().sum(-1)[:,None,:]+(y*apply(ax,1/sv,y)).sum(-1)+dim*math.log(2*math.pi))
                logpost=torch.log_softmax(W.clamp_min(torch.finfo(P.dtype).tiny).log()[:,None,:]+logphi,dim=-1)
                post=logpost.exp()
                g=gamma[:,None,:]+(beta[:,None,:,:]*y).sum(-1)+.5*((y*apply(ax,zeta[:,None,:,:]/sv,y)).sum(-1)-zeta.sum(-1)[:,None,:])
                eta_mean=drift-mu/t
                eta_var=drift_var-var/t+sigma**2
                eta=eta_mean[:,None,:,:]+apply(ax,eta_var[:,None,:,:]/sv,y)
                mean_g=(post*g).sum(-1,keepdim=True)
                mean_eta=(post[:,:,:,None]*eta).sum(-2,keepdim=True)
                spatial=(post*g*(logpost-logprior[index,None,:])).sum(-1).mean(1)
                raw=(post*g*logpost).sum(-1).mean(1)
                vg=(post*(g-mean_g).square()).sum(-1).mean(1)
                ve=(post*(eta-mean_eta).square().sum(-1)).sum(-1).mean(1)
                avg_eta=(prior_t[:,:,None]*eta_mean).sum(1,keepdim=True)
                ve_upper=(prior_t*((eta_mean-avg_eta).square().sum(-1)+(eta_var.square()/var).sum(-1))).sum(1)
                avg_gamma=(prior_t*gamma).sum(1,keepdim=True)
                var_g=(beta*apply(axis,var,beta)).sum(-1)+.5*zeta.square().sum(-1)
                vg_upper=(prior_t*((gamma-avg_gamma).square()+var_g)).sum(1)
                wstar=-delta+t*vec('h_mu')
                F=.5*(1-t)*wstar.square().sum(-1)+.5*sigma**2*t*vec('Z').exp().sum(-1)
                Eres=(drift+wstar).square().sum(-1)+((drift_var+var/(1-t)).square()/var).sum(-1)
                Eg2=gamma.square()+var_g
                EgF=gamma*F+(apply(axis,var,beta)*wstar).sum(-1)+.5*(zeta*var/(1-t)).sum(-1)
                q=-ell+t*scalar('h_mass')
                endpoint=ell.exp()*q*exprel((1-t)*q)
                ren=scalar('mass')*(.5*Eres+.5*alpha*Eg2+EgF)+dim*sigma**2/2*endpoint
                ren_row=(P[index]*ren).sum(1)
                # Source sampling law a_i / sum a_i, with exact row mass factor.
                factor=mass/a[index]*a.sum()/nrows*float(weight)
                sums[:6]+=torch.stack((sigma**2*spatial,sigma**2*raw,-.5*ve,-alpha/2*vg,.5*ve_upper,alpha/2*vg_upper),1).mul(factor[:,None]).sum(0)
                sums[6]+=(ren_row/a[index]*a.sum()/nrows*float(weight)).sum()
                sums[7]+=factor.sum()
    values=sums.cpu().tolist(); boundary=float(boundary); base=float(base); entropy_source=float(entropy_source)
    entropy=boundary+values[0]
    return dict(entropy_correction=entropy,entropy_spatial=values[0],entropy_boundary=boundary,
        entropy_raw_direct=values[1],eta_correction=values[2],growth_correction=values[3],
        total_correction=entropy+values[2]+values[3],eta_magnitude_upper_bound=values[4],growth_magnitude_upper_bound=values[5],
        surrogate=base+values[6]+entropy_source,pair_inner_product=base+values[6],coupling_entropy=entropy_source,integrated_mass=values[7])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--dataset',required=True,choices=['gene','mouse','emt'])
    parser.add_argument('--theta',type=float,nargs='+',default=[0,.25,.5,.75,1])
    parser.add_argument('--seeds',type=int,default=4); parser.add_argument('--time-nodes',type=int,default=16)
    parser.add_argument('--rows',type=int,default=32); parser.add_argument('--points',type=int,default=16)
    parser.add_argument('--tag',default='main'); args=parser.parse_args()
    out=ROOT/'../Results/posterior_correction_audit';out.mkdir(exist_ok=True)
    device=configure(load_config(f'configs/{args.dataset}_tight_bound.yaml'))
    assert device.type=='cuda'
    for theta in args.theta:
        cfg,path,blocks,provenance=candidate(args.dataset,theta,device)
        rows=[]; start=time.perf_counter()
        for seed in range(5000,5000+args.seeds):
            values=estimate(path,blocks,theta,seed,args.time_nodes,args.rows,args.points)
            rows.append(dict(seed=seed,**values))
            print(json.dumps(dict(dataset=args.dataset,theta=theta,seed=seed,seconds=time.perf_counter()-start,**values)),flush=True)
        json_write(out/f'{args.dataset}_{theta:g}_{args.tag}.json',dict(dataset=args.dataset,theta=theta,
            time_nodes=args.time_nodes,rows_per_interval=args.rows,points_per_row=args.points,provenance=provenance,replicates=rows,
            interpretation='discrete posterior corrections along a feasible interpolation; not a continuous-endpoint Action identity'))
        del path,blocks


if __name__=='__main__':main()

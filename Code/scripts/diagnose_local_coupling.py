"""Local reoptimization with the actual discrete posterior correction.

Frozen path weights, full P/Q arrays, original marginal/ratio constraints.
No artificial trust region. A fixed baseline mixture is an importance proposal;
differentiate both its density ratio and the posterior integrand. The objective
is a discrete-label diagnostic, NOT a certified continuous RUOT Action.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config,configure,load_path,json_write,file_hash
from src.distance_cost_table import DistanceHermiteGrid
from src.primal_coupling_gpu import kl_project
from src.coupling import OracleError


def apply(axis,eigen,y):
    return eigen[..., -1:]*y+(eigen[..., :1]-eigen[..., -1:])*axis*(axis*y).sum(-1,keepdim=True)


class PosteriorObjective:
    def __init__(self,run,block,nodes,points,seed,chunk,proposal_mode='reparameterized'):
        self.cfg=load_config(run/'config.yaml');self.device=configure(self.cfg)
        self.path=load_path(run,self.cfg,self.device)
        assert self.path.cost_objective=='renormalized_surrogate'
        self.sigma,self.alpha,self.dim=self.path.sigma,self.path.alpha,self.path.dimension
        with np.load(run/'couplings'/block) as a:
            for key in ['P','Q','source_weights','target_weights','source_positions','target_positions']:
                setattr(self,key,torch.as_tensor(a[key],device=self.device,dtype=torch.float64))
        self.a,self.b=self.source_weights,self.target_weights
        self.n,self.j=self.P.shape;self.chunk=chunk;self.points=points;self.nodes=nodes
        self.proposal_mode=proposal_mode
        self.samples=self.j*points if proposal_mode=='all_components' else points
        self.delta=self.target_positions[None]-self.source_positions[:,None]
        raw=torch.load(run/'distance_cost_table.pt',map_location=self.device,weights_only=False)
        assert raw['path_checkpoint_sha256']==file_hash(run/'pair_path.pt')
        grid=DistanceHermiteGrid(raw['distances'],raw['ratios'],raw['cost'],raw['slope'])
        self.oracle=grid.bind(self.delta.norm(dim=-1))
        x,w=np.polynomial.legendre.leggauss(nodes)
        u=torch.as_tensor((x+1)/2,device=self.device,dtype=torch.float64)
        self.ts=1-(1-u).square()
        self.tw=torch.as_tensor(w,device=self.device,dtype=torch.float64)*(1-u)
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.proposals=[]
        with torch.no_grad():
            for start in range(0,self.n,chunk):
                stop=min(self.n,start+chunk)
                s=self.states(self.P[start:stop],self.Q[start:stop],start,stop)
                r=nodes*(stop-start)
                vec=lambda k:s[k].reshape(r,self.j,self.dim)
                m=s['mass'].reshape(nodes,stop-start,self.j)
                W=(self.P[start:stop][None]*m).reshape(r,self.j)
                cdf=(W/W.sum(-1,keepdim=True)).cumsum(-1);cdf[:,-1]=1
                if proposal_mode=='all_components':
                    ids=torch.arange(self.j,device=self.device).repeat_interleave(points)[None].expand(r,-1)
                else:
                    unif=torch.rand((r,points),device=self.device,dtype=torch.float64,generator=self.generator)
                    ids=torch.searchsorted(cdf.contiguous(),unif.contiguous()).clamp_max(self.j-1)
                row=torch.arange(r,device=self.device)[:,None]
                noise=torch.randn((r,self.samples,self.dim),device=self.device,dtype=torch.float64,generator=self.generator)
                xx=vec('xi')[row,ids]+apply(vec('axis')[row,ids],vec('var')[row,ids].sqrt(),noise)
                if proposal_mode=='spatial':
                    lp=self.log_density(s,xx,W)[1]
                    self.proposals.append((xx,lp))
                else:
                    selected_prior=(torch.full((r,self.samples),1/self.j,device=self.device,dtype=torch.float64)
                        if proposal_mode=='all_components' else (W/W.sum(-1,keepdim=True))[row,ids])
                    self.proposals.append((ids,noise,selected_prior))

    def states(self,p,q,start,stop):
        nr=stop-start
        t=self.ts[:,None,None,None].expand(self.nodes,nr,self.j,1).reshape(-1,1)
        d=self.delta[start:stop][None].expand(self.nodes,nr,self.j,self.dim).reshape(-1,self.dim)
        lr=(q.log()-p.log())[None].expand(self.nodes,nr,self.j).reshape(-1,1)
        return self.path(t,d,lr)

    def log_density(self,s,x,W):
        r=len(x);vec=lambda k:s[k].reshape(r,self.j,self.dim)
        mu,axis,var=vec('xi'),vec('axis'),vec('var')
        norm2=x.square().sum(-1)[:,:,None]+mu.square().sum(-1)[:,None]-2*torch.bmm(x,mu.transpose(1,2))
        along=torch.bmm(x,axis.transpose(1,2))-(axis*mu).sum(-1)[:,None]
        parallel,perp=var[...,0][:,None],var[...,-1][:,None]
        quadratic=norm2/perp+(1/parallel-1/perp)*along.square()
        logphi=-.5*(self.dim*math.log(2*math.pi)+var.log().sum(-1)[:,None]+quadratic)
        logmix=torch.logsumexp(W.log()[:,None]+logphi,dim=-1)-W.sum(-1).log()[:,None]
        return logphi,logmix,(norm2,along)

    def correction_chunk(self,p,q,start,stop,proposal):
        s=self.states(p,q,start,stop);nr=stop-start;r=self.nodes*nr
        vec=lambda k:s[k].reshape(r,self.j,self.dim)
        scalar=lambda k:s[k].reshape(r,self.j)
        W=(p[None]*s['mass'].reshape(self.nodes,nr,self.j)).reshape(r,self.j)
        if self.proposal_mode=='spatial':
            x,proposal_logp=proposal
        else:
            ids,noise,selected_prior=proposal
            row=torch.arange(r,device=self.device)[:,None]
            x=vec('xi')[row,ids]+apply(vec('axis')[row,ids],vec('var')[row,ids].sqrt(),noise)
        logphi,logmix,(norm2,along)=self.log_density(s,x,W)
        post=torch.softmax(W.log()[:,None]+logphi,dim=-1)
        beta,zeta=-vec('b')/self.alpha,-vec('K')/self.alpha
        axis,var=vec('axis'),vec('var')
        qcoef=zeta/var
        g=scalar('gamma')[:,None]+(beta*axis).sum(-1)[:,None]*along+.5*(
            qcoef[...,-1][:,None]*norm2+(qcoef[...,0]-qcoef[...,-1])[:,None]*along.square()-zeta.sum(-1)[:,None])
        t=self.ts[:,None].expand(self.nodes,nr).reshape(r,1,1)
        emean=(-vec('b')-vec('xi')/t)
        ec=(emean*axis).sum(-1)
        ecoef=(-vec('K')-var/t+self.sigma**2)/var
        ep,eo=ecoef[...,0],ecoef[...,-1]
        esquare=ec.square()[:,None]+2*(ec*ep)[:,None]*along+eo.square()[:,None]*norm2+(ep.square()-eo.square())[:,None]*along.square()
        mu_axis=(vec('xi')*axis).sum(-1)
        ecoeff=ec[:,None]-eo[:,None]*mu_axis[:,None]+(ep-eo)[:,None]*along
        eta_bar=x*(post*eo[:,None]).sum(-1)[...,None]+torch.bmm(post*ecoeff,axis)
        G=(post*g).sum(-1)
        vg=(post*(g-G[:,:,None]).square()).sum(-1)
        ve=(post*esquare).sum(-1)-eta_bar.square().sum(-1)
        spatial=self.sigma**2*(post*g*(logphi-logmix[:,:,None])).sum(-1)
        if self.proposal_mode=='spatial':
            weight=(logmix-proposal_logp).exp()
        else:
            weight=W[row,ids]/(selected_prior*W.sum(-1)[:,None])
        R=(weight*(spatial-.5*ve-.5*self.alpha*vg)).mean(-1)*W.sum(-1)
        result=(R.reshape(self.nodes,nr)*self.tw[:,None]).sum()
        # Relative importance ESS is diagnostic only: no weight clipping.
        ess=(torch.ones(r,device=self.device) if self.proposal_mode=='all_components'
            else weight.sum(-1).square()/(self.samples*weight.square().sum(-1)))
        return result,ess.detach().min()

    def evaluate(self,p,q,corrected=True,gradient=False):
        with torch.no_grad():
            c,cr=self.oracle(q/p)
            pair=(p*c).sum()
            hs=self.sigma**2*(p*(p.log()-1)).sum()
            recvlog=q.log()-q.sum(1).log()[:,None]-(self.b/self.b.sum()).log()[None]
            hr=self.sigma**2*(q*recvlog).sum()
            value=pair+(hr if corrected else hs)
            gp=c-q/p*cr+(0 if corrected else self.sigma**2*p.log())
            gq=cr+(self.sigma**2*recvlog if corrected else 0)
            R=p.new_zeros(());ess=p.new_ones(())
        if corrected:
            for k,start in enumerate(range(0,self.n,self.chunk)):
                stop=min(self.n,start+self.chunk)
                pp=p[start:stop].detach().requires_grad_(gradient)
                qq=q[start:stop].detach().requires_grad_(gradient)
                with torch.set_grad_enabled(gradient):
                    v,e=self.correction_chunk(pp,qq,start,stop,self.proposals[k])
                    if gradient:
                        dp,dq=torch.autograd.grad(v,(pp,qq))
                        gp[start:stop]+=dp;gq[start:stop]+=dq
                R+=v.detach();ess=torch.minimum(ess,e)
            value+=R
        details=dict(pair=float(pair),source_entropy=float(hs),receiver_entropy=float(hr),
            spatial_correction=float(R),objective=float(value),min_relative_ess=float(ess))
        return value,gp,gq,details

    def changes(self,p,q):
        b0=self.P@self.target_positions/self.a[:,None]
        bn=p@self.target_positions/self.a[:,None]
        return dict(P_relative_TV=float((p-self.P).abs().sum()/(2*self.a.sum())),
            Q_relative_TV=float((q-self.Q).abs().sum()/(2*self.b.sum())),
            mean_barycenter_shift=float((self.a*(bn-b0).norm(dim=-1)).sum()/self.a.sum()),
            mean_abs_log_ratio_change=float((self.P*((q/p).log()-(self.Q/self.P).log()).abs()).sum()/self.a.sum()),
            row_relative_l1=float((p.sum(1)-self.a).abs().sum()/self.a.sum()),
            column_relative_l1=float((q.sum(0)-self.b).abs().sum()/self.b.sum()))


def optimize(obj,corrected,steps,out):
    p,q=obj.P.clone(),obj.Q.clone();step=1.;history=[];begin=time.perf_counter()
    status='iteration_budget';rmin=obj.cfg['pair']['ratio_min'];rmax=obj.cfg['pair']['ratio_max']
    initial=obj.evaluate(p,q,corrected)[3]
    for it in range(1,steps+1):
        value,gp,gq,detail=obj.evaluate(p,q,corrected,gradient=True)
        gp-=((p*gp).sum(1)/obj.a)[:,None]
        gq-=((q*gq).sum(0)/obj.b)[None]
        lp,lq=p.log(),q.log();trial=step*2 if it>1 else step
        for bt in range(30):
            try:
                nlp,nlq,projection=kl_project(lp-trial*gp,lq-trial*gq,obj.a,obj.b,rmin,rmax)
            except OracleError as error:
                if error.status != 'primal_projection_not_converged':
                    raise
                trial*=.5
                continue
            pn,qn=nlp.exp(),nlq.exp()
            vn,_,_,dn=obj.evaluate(pn,qn,corrected)
            linear=(gp*(pn-p)+gq*(qn-q)).sum()
            kl=(pn*(nlp-lp)-pn+p+qn*(nlq-lq)-qn+q).sum()
            if bool(torch.isfinite(vn) & (vn<=value+linear+kl/trial+1e-12*(1+value.abs())) & (vn<=value+1e-12*(1+value.abs()))):
                break
            trial*=.5
        else:
            status='line_search_failed';break
        mapping=float(((pn-p).abs().sum()/obj.a.sum()+(qn-q).abs().sum()/obj.b.sum())/trial)
        p,q,step=pn,qn,trial
        record=dict(iteration=it,seconds=time.perf_counter()-begin,mapping=mapping,step=step,
            backtracks=bt,**dn,**obj.changes(p,q))
        history.append(record)
        if it%5==0:
            torch.save(dict(P=p.cpu(),Q=q.cpu()),out.with_suffix('.pt'))
        json_write(out.with_suffix('.json'),dict(corrected=corrected,status='running',initial=initial,history=history))
        print(json.dumps(dict(corrected=corrected,**record)),flush=True)
        if mapping<1e-3:
            status='converged';break
    torch.save(dict(P=p.cpu(),Q=q.cpu()),out.with_suffix('.pt'))
    result=dict(corrected=corrected,status=status,initial=initial,history=history,
        final=obj.evaluate(p,q,corrected)[3],changes=obj.changes(p,q),seconds=time.perf_counter()-begin)
    json_write(out.with_suffix('.json'),result)
    return p,q,result


def main():
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--block',default='interval_0.npz')
    a.add_argument('--nodes',type=int,default=16);a.add_argument('--points',type=int,default=32)
    a.add_argument('--chunk',type=int,default=4);a.add_argument('--steps',type=int,default=40)
    a.add_argument('--seed',type=int,default=7300);a.add_argument('--check-only',action='store_true')
    a.add_argument('--proposal',choices=['spatial','reparameterized','all_components'],default='reparameterized')
    a.add_argument('--tag',default='reparameterized')
    args=a.parse_args();run=Path(args.run).resolve()
    out=Path('../Results/local_coupling_posterior')/run.name/Path(args.block).stem/args.tag;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();obj=PosteriorObjective(run,args.block,args.nodes,args.points,args.seed,args.chunk,args.proposal)
    # Check the full posterior derivative on one row chunk against central differences.
    stop=min(args.chunk,obj.n);pp=obj.P[:stop].clone().requires_grad_();qq=obj.Q[:stop].clone().requires_grad_()
    v,_=obj.correction_chunk(pp,qq,0,stop,obj.proposals[0]);dp,dq=torch.autograd.grad(v,(pp,qq))
    gen=torch.Generator(device=obj.device).manual_seed(937)
    vp=pp.detach()*torch.randn(pp.shape,device=obj.device,dtype=pp.dtype,generator=gen)
    vq=qq.detach()*torch.randn(qq.shape,device=obj.device,dtype=qq.dtype,generator=gen)
    analytic=float((dp*vp+dq*vq).sum());checks=[]
    with torch.no_grad():
        for h in [1e-3,1e-4,1e-5]:
            plus=obj.correction_chunk(pp+h*vp,qq+h*vq,0,stop,obj.proposals[0])[0]
            minus=obj.correction_chunk(pp-h*vp,qq-h*vq,0,stop,obj.proposals[0])[0]
            fd=float((plus-minus)/(2*h));checks.append(dict(h=h,analytic=analytic,finite_difference=fd,error=abs(fd-analytic)))
    if checks[-1]['error']>1e-6*(1+abs(analytic)):
        raise RuntimeError(f'Posterior gradient check failed: {checks}')
    metadata=dict(run=str(run),block=args.block,nodes=args.nodes,points=args.points,seed=args.seed,
        shape=list(obj.P.shape),path_sha256=file_hash(run/'pair_path.pt'),gradient_checks=checks,
        setup_seconds=time.perf_counter()-start,proposal_mode=args.proposal,scope='Full coupling arrays; frozen path networks; discrete posterior objective; no Matching retraining')
    json_write(out/'metadata.json',metadata);print(json.dumps(metadata),flush=True)
    if args.check_only:
        t=time.perf_counter();d=obj.evaluate(obj.P,obj.Q,True,True)[3]
        print(json.dumps(dict(gradient_seconds=time.perf_counter()-t,**d)),flush=True);return
    results={}
    for corrected,name in [(False,'original_refined'),(True,'posterior_refined')]:
        p,q,r=optimize(obj,corrected,args.steps,out/name);results[name]=(p,q,r)
    # Held-out proposal samples and finer time rule, with no clipping or retraining.
    del obj
    validation=[]
    for seed in [args.seed+100,args.seed+101]:
        test=PosteriorObjective(run,args.block,2*args.nodes,2*args.points,seed,args.chunk,args.proposal)
        rows={'baseline':test.evaluate(test.P,test.Q,True)[3]}
        for name,(p,q,_) in results.items():rows[name]=test.evaluate(p,q,True)[3]
        validation.append(dict(seed=seed,results=rows));del test
    json_write(out/'validation.json',validation)
    print(json.dumps(dict(validation=validation,seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':main()

"""Exact Gaussian Cren versus sigma expansion, fixed normalized path shapes.
No training. Uniform endpoint ratios/distance choices predeclared below.
"""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.runtime import configure,load_config,json_write
from src.wfr_heuristic import WFRBridgePath
from src.pair_cost import finite_pair_integrand
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate(alpha,sigma,ratio,shape,nodes,device):
    distance=.5*np.sqrt(alpha);dim=2
    q,w=np.polynomial.legendre.leggauss(nodes)
    t=torch.tensor((q+1)/2,dtype=torch.float64,device=device)[:,None]
    weights=torch.tensor(w/2,dtype=t.dtype,device=device)
    delta=torch.zeros(nodes,dim,dtype=t.dtype,device=device);delta[:,0]=distance
    ell=torch.full_like(t,np.log(ratio));tau=t*(1-t);dtau=1-2*t
    h=WFRBridgePath(alpha,sigma)(t,torch.zeros_like(delta),delta,ell)
    Z=shape*tau;dotZ=shape*dtau;kappa=tau*Z.exp();dotkappa=Z.exp()*(dtau+tau*dotZ)
    var=(sigma**2*kappa).expand_as(delta);dotvar=(sigma**2*dotkappa).expand_as(delta)
    s=dict(t=t,tau=tau,delta=delta,log_r=ell,xi=h['xi'],dotxi=h['dotxi'],mass=h['mass'],gamma=h['gamma'],
        var=var,Z=Z.expand_as(delta),dotZ=dotZ.expand_as(delta),h_var=torch.full_like(delta,shape),
        h_mu=(h['xi']-t*delta)/tau,h_mass=(h['mass'].log()-t*ell)/tau,
        b=-alpha*h['dotxi']/(alpha+var),K=-alpha*(dotvar-sigma**2)/(2*alpha+var),dvar_minus_sigma2=dotvar-sigma**2)
    integrate=lambda x:float((x[:,0]*weights).sum())
    exact=.5*distance**2-dim*sigma**2/2*ratio+integrate(finite_pair_integrand(s,sigma,alpha))
    m=h['mass'];v2=h['dotxi'].square().sum(-1,keepdim=True)
    J0=integrate(m*(v2/2+alpha*h['gamma'].square()/2))
    # Explicit finite part of the only divergent coefficient.
    J1=integrate(m*(dim*(dotkappa-1).square()/(8*kappa)-kappa*v2/(2*alpha))-dim*ratio/(2*(1-t)))
    J2=integrate(m*(kappa.square()*v2/(2*alpha**2)-dim*(dotkappa-1).square()/(16*alpha)))
    return dict(alpha=alpha,sigma=sigma,beta=sigma**2/alpha,distance=distance,ratio=ratio,shape=shape,
        C_exact=exact,J0=J0,J1=J1,J2=J2,correction=exact-J0,
        remainder_after_sigma2=exact-J0-sigma**2*J1,remainder_after_sigma4=exact-J0-sigma**2*J1-sigma**4*J2)


def main():
    device=configure(load_config(ROOT/'configs/gene_tight_bound.yaml'))
    out=ROOT/'../Results/standalone_path_diagnostic/sigma_scaling';out.mkdir(parents=True,exist_ok=True)
    rows=[];quad_errors=[]
    for alpha in (2.,10.):
        for ratio in (1.,4.):
            for shape in (0.,-1.):
                for sigma in (.03125,.0625,.125,.25,.5,1.,2.):
                    a=evaluate(alpha,sigma,ratio,shape,128,device)
                    b=evaluate(alpha,sigma,ratio,shape,256,device)
                    quad_errors.append(abs(a['C_exact']-b['C_exact']));rows.append(b)
    collapse=[]
    for ratio in (1.,4.):
        for beta in (.001,.01,.1,.5):
            a=evaluate(2.,np.sqrt(2*beta),ratio,0.,256,device)
            b=evaluate(10.,np.sqrt(10*beta),ratio,0.,256,device)
            collapse.append(abs(a['C_exact']/2-b['C_exact']/10))
    slopes=[]
    for alpha in (2.,10.):
        for ratio in (1.,4.):
            subset=[r for r in rows if r['alpha']==alpha and r['ratio']==ratio and r['shape']==0 and r['sigma']<=.125]
            x=np.log([r['sigma'] for r in subset])
            slopes.append(dict(alpha=alpha,ratio=ratio,
                correction_order=float(np.polyfit(x,np.log(np.abs([r['correction'] for r in subset])),1)[0]),
                residual_order=float(np.polyfit(x,np.log(np.abs([r['remainder_after_sigma2'] for r in subset])),1)[0])))
    summary=dict(dimension=2,normalized_distance=.5,slopes=slopes,quadrature_max_error=max(quad_errors),
        nondimensional_collapse_max_error=max(collapse),rows=rows,
        interpretation='Fixed normalized mean/mass and covariance shapes. No global optimum or non-Gaussian gap assertion.')
    json_write(out/'summary.json',summary)
    with (out/'values.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for alpha in (2.,10.):
        for ratio in (1.,4.):
            subset=[r for r in rows if r['alpha']==alpha and r['ratio']==ratio and r['shape']==0]
            x=np.array([r['sigma'] for r in subset]);label=f'alpha={alpha:g}, r={ratio:g}'
            axes[0].loglog(x,np.abs([r['correction'] for r in subset]),'o-',label=label)
            axes[1].loglog(x,np.abs([r['remainder_after_sigma2'] for r in subset]),'o-',label=label)
            alt=[r for r in rows if r['alpha']==alpha and r['ratio']==ratio and r['shape']==-1]
            axes[2].semilogx(x,[(a['C_exact']-b['C_exact'])/a['sigma']**2 for a,b in zip(subset,alt)],'o-',label=label)
    for ax in axes:ax.set_xlabel('sigma');ax.grid(alpha=.2)
    axes[0].set_title('Leading correction: order sigma^2');axes[0].set_ylabel('|Cren - J0|');axes[0].legend(fontsize=8)
    axes[1].set_title('After subtracting sigma^2 term');axes[1].set_ylabel('|Cren - J0 - sigma^2 J1|')
    axes[2].set_title('Fixed covariance shape change');axes[2].set_ylabel('(Cbridge - Cmodified) / sigma^2')
    fig.savefig(out/'scaling.png',dpi=180);fig.savefig(out/'scaling.pdf');plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()

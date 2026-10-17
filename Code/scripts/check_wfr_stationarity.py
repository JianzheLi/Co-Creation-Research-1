"""First variations of Cren at WFR + Brownian bridge; no model training.

Fixed endpoints imply the subtraction is constant. Hence Cren's derivative
must equal the derivative of the analytically minimized raw running energy.
Test that identity with endpoint-compatible Gaussian perturbations.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.runtime import configure,load_config,json_write
from src.wfr_heuristic import WFRBridgePath
from src.pair_cost import finite_pair_integrand

ROOT=Path(__file__).resolve().parents[1]


def objective(coordinates,distance,ratio,nodes,device):
    alpha,sigma,dimension=10.,.25,2
    q,w=np.polynomial.legendre.leggauss(nodes)
    t=torch.tensor((q+1)/2,dtype=torch.float64,device=device)[:,None]
    weight=torch.tensor(w/2,dtype=torch.float64,device=device)
    delta=torch.zeros(nodes,dimension,dtype=t.dtype,device=device);delta[:,0]=distance
    ell=torch.full_like(t,np.log(ratio));tau=t*(1-t);dtau=1-2*t
    base=WFRBridgePath(alpha,sigma)(t,torch.zeros_like(delta),delta,ell)
    hm=tau.square()*(2*t-1);dhm=2*tau*dtau*(2*t-1)+2*tau.square()
    h=tau.square();dh=2*tau*dtau
    mu=base['xi']+coordinates[0]*hm*delta
    dmu=base['dotxi']+coordinates[0]*dhm*delta
    mass=base['mass']*torch.exp(coordinates[1]*h)
    gamma=base['gamma']+coordinates[1]*dh
    Z=tau*coordinates[2:][None,:];dZ=dtau*coordinates[2:][None,:]
    var=sigma**2*tau*Z.exp();dvar=sigma**2*Z.exp()*(dtau+tau*dZ)
    state=dict(t=t,tau=tau,delta=delta,log_r=ell,xi=mu,dotxi=dmu,mass=mass,gamma=gamma,
        var=var,Z=Z,dotZ=dZ,h_var=Z/tau,h_mu=(mu-t*delta)/tau,h_mass=(mass.log()-t*ell)/tau,
        b=-alpha*dmu/(alpha+var),K=-alpha*(dvar-sigma**2)/(2*alpha+var),dvar_minus_sigma2=dvar-sigma**2)
    cren=.5*distance**2-dimension*sigma**2/2*ratio+(finite_pair_integrand(state,sigma,alpha)[:,0]*weight).sum()
    energy=mass*(alpha/2*gamma.square()+alpha/2*(dmu.square()/(alpha+var)).sum(-1,keepdim=True)
        +alpha/4*((dvar-sigma**2).square()/(var*(2*alpha+var))).sum(-1,keepdim=True))
    truncated=(energy[:,0]*weight).sum() # Value diverges as quadrature resolves t=1, derivatives finite.
    analytic_mu=(mass*alpha/(alpha+var[:,:1])*(dmu*(dhm*delta)).sum(-1,keepdim=True))[:,0]
    return cren,truncated,(analytic_mu*weight).sum()


def main():
    device=configure(load_config(ROOT/'configs/gene_tight_bound.yaml'))
    rows=[]
    for d,r in [(0.,1.),(1.,1.),(.8,1.13),(1.,4.),(2.,.1)]:
        rows_by_nodes=[]
        for nodes in (64,128,256):
            a=torch.zeros(4,dtype=torch.float64,device=device,requires_grad=True)
            finite,raw,analytic=objective(a,d,r,nodes,device)
            gf=torch.autograd.grad(finite,a,retain_graph=True)[0]
            gr=torch.autograd.grad(raw,a)[0]
            torch.testing.assert_close(gf,gr,atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(gf[0],analytic,atol=1e-10,rtol=1e-8)
            direction=-gf.detach()/gf.detach().norm()
            step=.001
            lowered=objective(step*direction,d,r,nodes,device)[0]
            assert lowered<finite
            rows_by_nodes.append(dict(nodes=nodes,cost=float(finite.detach()),first_variation=gf.detach().cpu().tolist(),
                raw_and_renormalized_gradient_max_error=float((gf-gr).abs().max()),
                descent_step=step,cost_change=float((lowered-finite).detach())))
        rows.append(dict(distance=d,ratio=r,quadrature=rows_by_nodes))
    out=ROOT/'../Results/standalone_path_diagnostic'
    json_write(out/'wfr_stationarity.json',dict(alpha=10.,sigma=.25,dimension=2,
        coordinates=['mean: mu_WFR + a*tau^2*(2t-1)*delta','mass: m_WFR*exp(a*tau^2)',
                     'parallel variance: sigma^2*tau*exp(a*tau)','perpendicular variance: sigma^2*tau*exp(a*tau)'],
        baseline_controls='Analytically minimal compatible controls, same as original RUOT',rows=rows))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

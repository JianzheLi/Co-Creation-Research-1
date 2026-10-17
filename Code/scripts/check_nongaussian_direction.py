"""Moment-preserving fourth-Hermite density perturbation with feasible controls.

Keep u fixed; solve the perturbed Fokker-Planck equation exactly for g. This
constructs a feasible non-Gaussian improvement, without fitting another model.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.runtime import configure,load_config,json_write
from src.wfr_heuristic import WFRBridgePath

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def check(alpha,distance,ratio,nodes,device):
    sigma=.25
    tx,tw=np.polynomial.legendre.leggauss(128)
    t=torch.tensor((tx+1)/2,device=device)[:,None];wt=torch.tensor(tw/2,device=device)
    gx,gw=np.polynomial.hermite.hermgauss(nodes)
    a,b=np.meshgrid(gx*np.sqrt(2),gx*np.sqrt(2),indexing='ij')
    z=torch.tensor(np.stack([a.ravel(),b.ravel()],axis=-1),device=device)
    weights=torch.tensor(np.outer(gw,gw).ravel()/np.pi,device=device)
    delta=torch.zeros(len(t),2,device=device,dtype=t.dtype);delta[:,0]=distance
    s=WFRBridgePath(alpha,sigma)(t,torch.zeros_like(delta),delta,torch.full_like(t,np.log(ratio)))
    var=s['var'];dotvar=s['dotvar'];mu_dot=s['dotxi'];mass=s['mass']
    drift=alpha*mu_dot/(alpha+var)
    A=alpha*(dotvar-sigma**2)/(var*(2*alpha+var))
    K=-A*var
    y=var.sqrt()[:,None,:]*z[None,:,:]
    u=drift[:,None,:]+A[:,None,:]*y
    g=s['gamma'][:,None,:]+((drift[:,None,:]*y).sum(-1,keepdim=True)+.5*A[:,None,:]*(y.square().sum(-1,keepdim=True)-2*var[:,None,:]))/alpha
    tau=t*(1-t);chi=16*tau.square();dotchi=32*tau*(1-2*t)
    zz=z[None,:,0:1];H4=zz**4-6*zz**2+3;Hp=4*zz**3-12*zz;Hpp=12*zz**2-12
    h=chi[:,None,:]*H4
    transport=(-mu_dot[:,None,0:1]+u[:,:,0:1]+sigma**2*y[:,:,0:1]/var[:,None,:])/var.sqrt()[:,None,:]-.5*zz*dotvar[:,None,:]/var[:,None,:]
    q=dotchi[:,None,:]*H4+chi[:,None,:]*Hp*transport-.5*sigma**2*chi[:,None,:]*Hpp/var[:,None,:]
    linear=h*(.5*u.square().sum(-1,keepdim=True)+.5*alpha*g.square())+alpha*g*q
    integrate=lambda v:((v[:,:,0]*weights).sum(-1)*mass[:,0]*wt).sum()
    derivative=integrate(linear)
    expected=(-3/alpha*(mass[:,0]*chi[:,0]*K[:,0].square()*wt).sum())
    torch.testing.assert_close(derivative,expected,atol=1e-10,rtol=1e-9)
    assert derivative<0
    # A small constructive step: since H4 >= -6 and chi <= 1, f_eps is positive.
    second=integrate(.5*alpha*q.square())
    eps=min(1e-3,float(-derivative/(4*second)))
    difference=integrate(eps*linear+.5*alpha*eps**2*q.square()/(1+eps*h))
    assert difference<0 and eps<1/6
    # Zeroth, first and second density moments are preserved by Hermite orthogonality.
    moments=torch.stack([torch.ones_like(z[:,0]),z[:,0],z[:,1],z[:,0]**2,z[:,1]**2,z[:,0]*z[:,1]])
    moment_error=float((moments*(H4[0,:,0]*weights)).sum(-1).abs().max())
    assert moment_error<1e-10
    return dict(alpha=alpha,distance=distance,ratio=ratio,spatial_nodes_per_dimension=nodes,
        derivative_formula=float(expected),derivative_feasible_control=float(derivative),
        epsilon=eps,finite_cost_change=float(difference),moment_orthogonality_error=moment_error)


def main():
    device=configure(load_config(ROOT/'configs/gene_tight_bound.yaml'))
    rows=[check(alpha,d,r,n,device) for alpha in (2.,10.) for d,r in ((0.,1.),(1.,1.),(1.,4.)) for n in (12,24)]
    json_write(ROOT/'../Results/standalone_path_diagnostic/nongaussian_direction.json',dict(
        density='f_eps=f_G*(1+eps*chi(t)*H4((x_i-mu_i)/sqrt(s_i))), chi=16*t^2*(1-t)^2, H4=z^4-6z^2+3',
        derivative='-3/alpha integral m*chi*(A_ii*s_i)^2 dt',
        interpretation='Explicit feasible non-Gaussian descent, preserving mass, mean, covariance. Not a trained non-Gaussian optimum.',rows=rows))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

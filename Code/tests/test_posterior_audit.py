"""Validate compatible-control interpolation, used only in the diagnostic."""
import torch
from src.pair_path import PairPath,spectral_apply
from scripts.audit_posterior_corrections import compatible_coefficients


def test_interpolated_controls_satisfy_gaussian_fokker_planck():
    torch.manual_seed(81)
    p=PairPath(dimension=5,width=8,layers=2,parameterization='axial_distance',cost_objective='tight_logmoment_upper_bound').double()
    with torch.no_grad():
        for net in (p.mu_net,p.var_net,p.mass_net):
            net.last.weight.normal_(0,.12)
    t=torch.tensor([[.0001],[.17],[.53],[.86],[.9999]],dtype=torch.float64)
    s=p(t,torch.randn(5,5,dtype=t.dtype),torch.randn(5,1,dtype=t.dtype))
    y=spectral_apply(s,s['var'].sqrt(),torch.randn_like(s['delta']))
    invy=spectral_apply(s,1/s['var'],y)
    lhs=s['gamma']+(s['dotxi']*invy).sum(-1,keepdim=True)+.5*((invy*spectral_apply(s,s['dotvar'],invy)).sum(-1,keepdim=True)-(s['dotvar']/s['var']).sum(-1,keepdim=True))
    for theta in (0.,.25,.5,.75,1.):
        drift,dv,beta,zeta=compatible_coefficients(s,theta,p.alpha)
        u=drift+spectral_apply(s,dv/s['var'],y)
        g=s['gamma']+(beta*y).sum(-1,keepdim=True)+.5*((y*spectral_apply(s,zeta/s['var'],y)).sum(-1,keepdim=True)-zeta.sum(-1,keepdim=True))
        rhs=-(dv/s['var']).sum(-1,keepdim=True)+(u*invy).sum(-1,keepdim=True)+p.sigma**2/2*(invy.square().sum(-1,keepdim=True)-(1/s['var']).sum(-1,keepdim=True))+g
        torch.testing.assert_close(lhs,rhs,atol=1e-8,rtol=1e-10)


def test_gaussian_entropy_boundary_control_variate():
    # For Gaussian-compatible controls E_phi g = gamma; integrating m gamma
    # gives r-1, independently of the spatially varying growth terms.
    import numpy as np
    x,w=np.polynomial.legendre.leggauss(64)
    t=torch.tensor((x+1)/2,dtype=torch.float64)[:,None]
    torch.manual_seed(82)
    p=PairPath(dimension=2,width=8,layers=2,parameterization='axial_distance',cost_objective='tight_logmoment_upper_bound').double()
    with torch.no_grad():
        p.mass_net.last.weight.normal_(0,.2)
    delta=torch.tensor([[1.,-.3]],dtype=t.dtype).expand(len(t),-1)
    for r in (.1,.7,2.,10.):
        s=p(t,delta,torch.full_like(t,np.log(r)))
        integral=(s['mass'][:,0]*s['gamma'][:,0]*torch.tensor(w/2,dtype=t.dtype)).sum()
        torch.testing.assert_close(integral,torch.tensor(r-1,dtype=t.dtype),atol=1e-10,rtol=1e-10)

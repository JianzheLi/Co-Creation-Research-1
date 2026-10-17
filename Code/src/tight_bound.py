"""Posterior-free positive log-moment bound and Gaussian integration.

r(log r)_+^2 is convex, and r(log r)_-^2 <= 4/e^2. Hence
E_p log^2(p/q) <= sum a E_phi [log(phi/q)_+^2] + 4/e^2.
The axial Gaussian integral uses an analytic truncated-normal fourth moment
along the axis and one generalized Gauss-Laguerre rule perpendicular to it.
"""
from functools import lru_cache
import math
import numpy as np
import torch
from scipy.special import roots_genlaguerre, gammaln


@lru_cache(maxsize=32)
def radial_rule(dimension, nodes, device, dtype):
    if dimension == 1:
        x,w=np.array([0.]),np.array([1.])
    else:
        shape=(dimension-1)/2
        x,w=roots_genlaguerre(nodes,shape-1)
        x=2*x
        w=w*np.exp(-gammaln(shape))
    return torch.tensor(x,device=device,dtype=dtype),torch.tensor(w,device=device,dtype=dtype)


def normal_positive_quadratic_second(a,b,c):
    """E[(a Z^2+b Z+c)_+^2], Z standard normal; arbitrary broadcasting."""
    def primitive(x):
        x=x.clamp(-38,38)
        density=torch.exp(-x.square()/2)/math.sqrt(2*math.pi)
        cdf=.5*torch.erfc(-x/math.sqrt(2))
        m1=-density
        m2=cdf-x*density
        m3=-(x.square()+2)*density
        m4=3*cdf-(x.pow(3)+3*x)*density
        return a.square()*m4+2*a*b*m3+(b.square()+2*a*c)*m2+2*b*c*m1+c.square()*cdf
    full=3*a.square()+b.square()+2*a*c+c.square()
    # A linear polynomial is handled separately, avoiding ill-conditioned roots.
    quadratic=a.abs()>1e-10
    safe_a=torch.where(quadratic,a,torch.ones_like(a))
    discriminant=b.square()-4*a*c
    root=discriminant.clamp_min(1e-30).sqrt()
    x1=(-b-root)/(2*safe_a); x2=(-b+root)/(2*safe_a)
    between=primitive(torch.maximum(x1,x2))-primitive(torch.minimum(x1,x2))
    result=torch.where(a>0,full-between,between)
    result=torch.where(discriminant>0,result,torch.where(a>0,full,torch.zeros_like(full)))
    nonzero_b=b.abs()>1e-12
    safe_b=torch.where(nonzero_b,b,torch.ones_like(b))
    x=(-c/safe_b).clamp(-38,38)
    density=torch.exp(-x.square()/2)/math.sqrt(2*math.pi)
    cdf=.5*torch.erfc(-x/math.sqrt(2))
    linear_left=b.square()*(cdf-x*density)-2*b*c*density+c.square()*cdf
    linear=torch.where(b>0,b.square()+c.square()-linear_left,linear_left)
    linear=torch.where(nonzero_b,linear,c.clamp_min(0).square())
    return torch.where(quadratic,result,linear).clamp_min(0)


def log_ratio_moments(s,sigma):
    relative=(1-s['t'])*s['Z'].exp()
    displacement=s['mean_displacement']
    mean=.5*((-torch.log1p(-s['t'])-s['Z']+relative-1).sum(-1,keepdim=True)
             +displacement.square().sum(-1,keepdim=True)/(sigma**2*s['t']))
    # The displacement is parallel to the path axis.
    variance=(displacement.square().sum(-1,keepdim=True)*s['var'][:,:1]/(sigma**4*s['t'].square())
              +.5*(relative-1).square().sum(-1,keepdim=True))
    return mean,mean.square()+variance


def positive_log_moment(s,sigma,nodes=32):
    dimension=s['delta'].shape[-1]
    radial,weights=radial_rule(dimension,nodes,s['t'].device,s['t'].dtype)
    relative=(1-s['t'])*s['Z'].exp()
    displacement=s['mean_displacement']
    parallel=(displacement*s['axis']).sum(-1,keepdim=True)
    a=.5*(relative[:,:1]-1)
    b=parallel*s['var'][:,:1].sqrt()/(sigma**2*s['t'])
    c=.5*((-torch.log1p(-s['t'])-s['Z']).sum(-1,keepdim=True)
           +displacement.square().sum(-1,keepdim=True)/(sigma**2*s['t']))
    if dimension>1:
        c=c+.5*(relative[:,1:2]-1)*radial[None,:]
    return (normal_positive_quadratic_second(a,b,c)*weights).sum(-1,keepdim=True)

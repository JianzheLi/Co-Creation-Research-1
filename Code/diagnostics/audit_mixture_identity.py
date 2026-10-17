"""Independent continuous-endpoint check; no project control/cost functions used.

One source at 0, emitting Y~N(0,v), f_t^Y=m(t,Y)N(tY,sigma²t(1-t)),
m(t,Y)=exp(t beta Y+t(1-t)h). The joint tilted distribution is Gaussian,
so posterior regression and all spatial moments are evaluated exactly by
Gauss-Hermite quadrature. Only the time integration is approximate.
"""
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
device = "cuda"


def check(alpha, sigma, n):
    variance, beta, h = 1.4, 0.35, 0.2
    nodes, weights = np.polynomial.legendre.leggauss(n)
    z, w = np.polynomial.hermite.hermgauss(8)
    q = torch.tensor((nodes + 1) / 2, device=device)[:, None, None]
    t = 1 - (1 - q).square()
    tw = torch.tensor(weights, device=device)[:, None, None] * (1 - q)
    z = torch.tensor(z * math.sqrt(2), device=device)
    hw = torch.tensor(np.outer(w, w) / math.pi, device=device)[None]
    s = sigma**2 * t * (1-t)
    mass = torch.exp(t*(1-t)*h + 0.5*t.square()*beta**2*variance)
    mass1 = math.exp(0.5*beta**2*variance)
    Y = t*beta*variance + math.sqrt(variance)*z[None, :, None]
    y = s.sqrt()*z[None, None, :]
    X = t*Y+y
    covx = t.square()*variance+s
    post_var = variance*s/covx
    post_mean = t*beta*variance + t*variance/covx*(X-t.square()*beta*variance)
    ey = X-t*post_mean
    eYy = X*post_mean-t*(post_mean.square()+post_var)
    ey2 = ey.square()+t.square()*post_var
    A = alpha/(alpha+s)
    H = 2*alpha/((1-t)*(2*alpha+s))
    c = -sigma**2*t/(2*alpha+s)
    U = A*post_mean-H*ey
    g = beta*Y+(1-2*t)*h + Y*y/(alpha+s)+c*(y.square()/s-1)
    G = beta*post_mean+(1-2*t)*h + eYy/(alpha+s)+c*(ey2/s-1)
    eta = -s/(alpha+s)*Y + sigma**2*t/(2*alpha+s)*y
    eta_bar = -s/(alpha+s)*post_mean + sigma**2*t/(2*alpha+s)*ey
    logomega = -0.5*(torch.log(2*math.pi*post_var)+(Y-post_mean).square()/post_var)
    F = (y-(1-t)*Y).square()/(2*(1-t))

    def integral(value):
        return (tw*mass*hw*value).sum()

    action = integral(0.5*U.square()+0.5*alpha*G.square())
    logmass = t*(1-t)*h + 0.5*t.square()*beta**2*variance
    mass_term = (tw*0.5*sigma**2*mass1*torch.expm1(logmass-math.log(mass1))/(1-t)).sum()
    cren = 0.5*variance-0.5*sigma**2*mass1 + integral(0.5*eta.square()+0.5*alpha*g.square()+g*F)+mass_term
    entropy = -0.5*sigma**2*math.log(2*math.pi*math.e*variance)
    fixed = 0.5*sigma**2*mass1*math.log(2*math.pi*math.e*sigma**2)
    K = integral(g*logomega)
    V = integral((eta-eta_bar).square())
    J = 0.5*integral((g-G).square())
    rhs = cren+entropy+fixed+sigma**2*K-0.5*V-alpha*J
    return dict(alpha=alpha, sigma=sigma, time_nodes=n, action=action.item(),
                decomposed_action=rhs.item(), absolute_error=abs(action.item()-rhs.item()))


rows = [check(a,s,n) for a,s in [(2.,.25),(2.,1.),(10.,.25),(10.,1.)]
        for n in [64,128,256]]
result = dict(device=device, spatial_quadrature="8x8 exact polynomial Gaussian moments",
              cases=rows, max_error_256=max(r['absolute_error'] for r in rows if r['time_nodes']==256))
Path('tmp/math_audit_mixture_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))

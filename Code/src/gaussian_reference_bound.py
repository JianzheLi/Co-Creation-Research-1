"""A Gaussian Brownian-bridge-mixture reference, fixed from observed data.

q_t=N(x0, [sigma^2 t(1-t)+s_data t^2] I).
s_data is the average (per coordinate) observed independent-endpoint
displacement second moment, equally weighted across model intervals.
It is derived once, not fitted or exposed as a tunable coefficient.
The reference and moments are shared by the alpha-scale and diffusion-scale
upper bounds. Their Young coefficients are selected by the cost/control
callers. All moments below are closed form.
"""
import numpy as np
import torch
from .pair_path import spectral_apply


def observed_second_moment(snapshots):
    moments = []
    for source, target in zip(snapshots[:-1], snapshots[1:]):
        x, y = source.x, target.x
        a, b = source.weights/source.mass, target.weights/target.mass
        v = (np.sum(a*np.sum(x*x, axis=1))+np.sum(b*np.sum(y*y, axis=1))
             -2*np.dot(a@x, b@y))/x.shape[1]
        moments.append(v)
    result = float(np.mean(moments))
    if not np.isfinite(result) or result <= 0:
        raise ValueError('Gaussian reference needs a positive observed displacement second moment')
    return result


def moments(state, sigma):
    var, mu, qv = state['var'], state['mean_displacement'], state['reference_variance']
    relative = var/qv
    mean = .5*((qv.log()-var.log()+relative-1).sum(-1, keepdim=True)
               +mu.square().sum(-1, keepdim=True)/qv)
    variance = (mu*spectral_apply(state, var, mu)).sum(-1, keepdim=True)/qv.square()
    variance = variance+.5*(relative-1).square().sum(-1, keepdim=True)
    cross = state['gamma']*mean
    cross = cross+(state['growth_linear']*spectral_apply(state, var, mu/qv)).sum(-1, keepdim=True)
    cross = cross+.5*(state['growth_quadratic']*(relative-1)).sum(-1, keepdim=True)
    return mean, mean.square()+variance, cross

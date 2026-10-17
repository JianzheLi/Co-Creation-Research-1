import itertools
import math
import numpy as np
import pytest
import torch
from src.pair_path import PairPath, controls, spectral_apply
from src.pair_cost import finite_moments, finite_pair_integrand, CostOracle
from src.gaussian_reference_bound import moments, observed_second_moment
from src.matching import make_targets
from src.data import Snapshot


def state(dimension, alpha, sigma, objective='gaussian_reference_upper_bound'):
    torch.manual_seed(71)
    path = PairPath(dimension=dimension, alpha=alpha, sigma=sigma, width=8, layers=2,
        parameterization='axial_distance', cost_objective=objective,
        reference_second_moment=.3).double()
    with torch.no_grad():
        for net in (path.mu_net, path.var_net, path.mass_net):
            net.last.weight.normal_(0, .06)
    t = torch.tensor([[.05], [.5], [.95]], dtype=torch.float64)
    delta = torch.randn(3, dimension, dtype=torch.float64)
    return path, path(t, delta, torch.tensor([[-.4], [.5], [1.]], dtype=torch.float64))


@pytest.mark.parametrize('objective', ['gaussian_reference_upper_bound','diffusion_full_square_upper_bound'])
@pytest.mark.parametrize('dimension,alpha,sigma', [(2,10,.25), (10,2,1), (2,2,.25), (10,10,1)])
def test_control_stationarity_compatibility_and_matching(dimension, alpha, sigma, objective):
    path, s = state(dimension, alpha, sigma, objective)
    targets = make_targets(s, torch.randn_like(s['xi']), sigma=sigma, alpha=alpha)
    x = targets['x'].detach().requires_grad_()
    u, g = controls(s, x, alpha=alpha)
    y = x-s['xi']
    invy = spectral_apply(s, 1/s['var'], y)
    score = -invy
    ell = -.5*(y*invy).sum(-1, keepdim=True)+.5*x.square().sum(-1, keepdim=True)/s['reference_variance']
    growth_weight = alpha+(sigma**2 if objective == 'diffusion_full_square_upper_bound' else alpha)
    potential = growth_weight*g+sigma**2*ell
    gradient = torch.autograd.grad(potential.sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(u, gradient, rtol=1e-10, atol=1e-10)
    div = sum(torch.autograd.grad(u[:, k].sum(), x, retain_graph=True)[0][:, k:k+1] for k in range(dimension))
    rhs = -div-(u*score).sum(-1, keepdim=True)+sigma**2/2*(score.square().sum(-1, keepdim=True)-(1/s['var']).sum(-1, keepdim=True))+g
    lhs = s['gamma']+(s['dotxi']*invy).sum(-1, keepdim=True)+.5*((invy*spectral_apply(s, s['dotvar'], invy)).sum(-1, keepdim=True)-(s['dotvar']/s['var']).sum(-1, keepdim=True))
    torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(targets['target_g'], g)
    torch.testing.assert_close(targets['target_v_scaled'], targets['sqrt_wf']*(u-sigma**2/2*score))
    for t in (1e-10, 1-1e-10):
        endpoint = path(torch.full_like(s['t'], t), s['delta'], s['log_r'])
        assert torch.isfinite(finite_pair_integrand(endpoint, sigma, alpha)).all()


@pytest.mark.parametrize('objective', ['gaussian_reference_upper_bound','diffusion_full_square_upper_bound'])
def test_gaussian_moments_independent_quadrature(objective):
    path, s = state(2, 2, 1, objective)
    z, w = np.polynomial.hermite.hermgauss(6)
    indices = list(itertools.product(range(6), repeat=2))
    noise = torch.tensor([[z[i], z[j]] for i,j in indices])*math.sqrt(2)
    weights = torch.tensor([w[i]*w[j]/math.pi for i,j in indices])
    for j in range(3):
        sj = {k:v[j:j+1] if torch.is_tensor(v) else v for k,v in s.items()}
        x = sj['xi']+spectral_apply(sj, sj['var'].sqrt(), noise)
        u, g = controls(sj, x, alpha=path.alpha)
        ell = -.5*(noise.square().sum(-1, keepdim=True)+sj['var'].log().sum(-1, keepdim=True))
        ell = ell+.5*(x.square().sum(-1, keepdim=True)/sj['reference_variance']+2*sj['reference_variance'].log())
        actual = moments(sj, path.sigma)
        for result, integrand in zip(actual, (ell, ell.square(), g*ell)):
            torch.testing.assert_close(result.squeeze(), (weights*integrand[:,0]).sum(), rtol=1e-10, atol=1e-10)
        _, parts = finite_pair_integrand(sj,path.sigma,path.alpha,return_terms=True)
        lam = path.sigma**2 if objective == 'diffusion_full_square_upper_bound' else path.alpha
        correction = .5*lam*(g+path.sigma**2/lam*ell).square()+2*path.sigma**4/(lam*math.e**2)
        torch.testing.assert_close(parts['bound_correction'].squeeze(), sj['mass'].squeeze()*(weights*correction[:,0]).sum(),rtol=1e-10,atol=1e-10)
        fm = finite_moments(sj, path.sigma, path.alpha)
        F = (x-sj['delta']).square().sum(-1, keepdim=True)/(2*(1-sj['t']))
        residual = u+(x-sj['delta'])/(1-sj['t'])
        for key, integrand in [('E_res',residual.square().sum(-1)), ('E_g2',g[:,0].square()), ('E_gF',(g*F)[:,0])]:
            torch.testing.assert_close(fm[key].squeeze(), (weights*integrand).sum(), rtol=1e-10, atol=1e-10)


def test_data_reference_matches_all_independent_pairs():
    x = np.array([[0.,1.],[2.,3.]])
    y = np.array([[1.,2.],[3.,0.],[-1.,2.]])
    a,b = np.array([.2,.8]), np.array([.3,.5,.2])
    actual = observed_second_moment([Snapshot(0,x,a,2), Snapshot(1,y,b,3)])
    expected = np.sum(a[:,None]*b[None,:]*np.sum((x[:,None]-y[None,:])**2,axis=-1))/2
    assert abs(actual-expected)<1e-12


@pytest.mark.parametrize('objective', ['gaussian_reference_upper_bound','diffusion_full_square_upper_bound'])
def test_cost_and_ratio_derivatives_converge(objective):
    path, s = state(2, 2, 1, objective)
    a = CostOracle(path, 64).tensor_query(s['delta'], s['log_r'].exp())
    b = CostOracle(path, 128).tensor_query(s['delta'], s['log_r'].exp())
    for left, right in zip(a,b):
        torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)

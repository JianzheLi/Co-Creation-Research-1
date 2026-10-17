"""Independent geometry, PDE, and renormalization checks for distance paths."""
import math

import numpy as np
import pytest
import torch

from src.matching import make_targets
from src.pair_cost import CostOracle, finite_moments, finite_pair_integrand
from src.pair_path import PairPath, controls
from src.runtime import load_config, make_path


def make_axial(dimension=3, device="cpu", mode="quadratic_minimal"):
    torch.manual_seed(125)
    path = PairPath(dimension=dimension, width=8, layers=2,
                    parameterization="axial_distance", control_mode=mode).double().to(device)
    with torch.no_grad():
        for net in (path.mu_net, path.var_net, path.mass_net):
            net.last.weight.normal_(0, 0.04)
            net.last.bias.normal_(0, 0.03)
        path.var_net.last.bias[1] = 0.3  # Exercise genuinely anisotropic paths.
    return path


def matrix(s, key="var"):
    """Explicit dense covariance for independent reference calculations."""
    eig, e = s[key], s["axis"]
    perpendicular = eig[:, -1:]
    eye = torch.eye(e.shape[-1], dtype=e.dtype, device=e.device)
    return (perpendicular[..., None] * eye
            + (eig[:, :1] - perpendicular)[..., None] * e[..., :, None] * e[..., None, :])


@pytest.mark.parametrize("dimension", [1, 2, 5, 10])
def test_endpoint_covariance_slopes_and_coincident_endpoints(dimension):
    path = make_axial(dimension)
    delta = torch.randn(4, dimension, dtype=torch.float64)
    delta[0] = 0
    ell = torch.linspace(-1, 1, 4, dtype=torch.float64)[:, None]
    eye = torch.eye(dimension, dtype=torch.float64).expand(4, -1, -1)
    for t in (0., 1.):
        s = path(torch.full_like(ell, t), delta, ell)
        torch.testing.assert_close(s["xi"], t * delta)
        torch.testing.assert_close(s["mass"], (t * ell).exp())
        torch.testing.assert_close(matrix(s), torch.zeros_like(eye))
        torch.testing.assert_close(matrix(s, "dotvar"), (1 - 2*t) * path.sigma**2 * eye)
        assert torch.isfinite(finite_pair_integrand(s)).all()
    middle = path(torch.full_like(ell, 0.4), delta, ell)
    assert torch.all(torch.linalg.eigvalsh(matrix(middle)) > 0)
    torch.testing.assert_close(matrix(middle)[0], middle["var"][0, 0] * eye[0])
    torch.testing.assert_close(middle["xi"][0], delta[0])
    # The remainder in Sigma = sigma^2 * t I + O(t^2) is bounded at both ends.
    for eps in (1e-4, 1e-6):
        for t in (eps, 1-eps):
            s = path(torch.full_like(ell, t), delta, ell)
            assert ((matrix(s) - path.sigma**2 * eps * eye) / eps**2).abs().max() < 3
    if dimension > 1:
        zero = path(torch.full_like(ell, 0.4), torch.zeros_like(delta), ell)
        near = path(torch.full_like(ell, 0.4), delta * 1e-8, ell)
        torch.testing.assert_close(matrix(near), matrix(zero), atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("dimension", [2, 3, 10])
def test_rotation_reflection_cost_derivatives_and_matching(device, dimension):
    path = make_axial(dimension, device)
    delta = torch.randn(6, dimension, dtype=torch.float64, device=device)
    delta[0] = 0
    t = torch.linspace(0.1, 0.9, 6, dtype=torch.float64, device=device)[:, None]
    ell = torch.linspace(-0.5, 1, 6, dtype=torch.float64, device=device)[:, None]
    noise, x0 = torch.randn_like(delta), torch.randn_like(delta)
    Q = torch.linalg.qr(torch.randn(dimension, dimension, dtype=torch.float64, device=device)).Q
    oracle = CostOracle(path, nodes=32, condition_chunk=3)
    original_cost = oracle.tensor_query(delta, ell.exp())
    for sign in (1, -1):
        R = Q.clone()
        R[:, 0] *= sign
        s = path(t, delta, ell, x0=x0)
        sr = path(t, delta @ R.T, ell, x0=x0 @ R.T)
        for field in ("xi", "dotxi", "b"):
            torch.testing.assert_close(sr[field], s[field] @ R.T, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(matrix(sr), R @ matrix(s) @ R.T, atol=1e-12, rtol=1e-11)
        rotated_cost = oracle.tensor_query(delta @ R.T, ell.exp())
        for a, b in zip(original_cost, rotated_cost):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        targets = make_targets(s, noise)
        rotated_targets = make_targets(sr, noise @ R.T)
        for field in ("x", "target_v_scaled", "target_s_scaled"):
            torch.testing.assert_close(rotated_targets[field], targets[field] @ R.T, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(rotated_targets["target_g"], targets["target_g"], atol=1e-10, rtol=1e-10)
    u, g = controls(s, targets["x"])
    score = -torch.linalg.solve(matrix(s), (targets["x"] - s["xi"])[..., None]).squeeze(-1)
    torch.testing.assert_close(targets["target_s_scaled"], targets["sqrt_ws"] * score)
    torch.testing.assert_close(targets["target_v_scaled"], targets["sqrt_wf"] * (u - path.sigma**2 / 2 * score))
    torch.testing.assert_close(targets["target_g"], g)
    h = 1e-4
    r = ell.exp()
    cm = oracle.tensor_query(delta, r - h, 0)
    cp = oracle.tensor_query(delta, r + h, 0)
    torch.testing.assert_close(original_cost[1], (cp-cm)/(2*h), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(original_cost[2], (cp-2*original_cost[0]+cm)/h**2, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("mode", ["quadratic_minimal", "uniform_growth"])
def test_full_covariance_pde_time_derivatives_and_parameter_gradient(mode):
    path = make_axial(mode=mode)
    t = torch.tensor([[0.2], [0.7]], dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([[0.4, -0.8, 1.2], [-1., 0.6, 0.3]], dtype=torch.float64)
    ell = torch.tensor([[0.3], [-0.7]], dtype=torch.float64)
    s = path(t, delta, ell)
    for value, derivative in (("xi", "dotxi"), ("var", "dotvar")):
        actual = torch.cat([torch.autograd.grad(s[value][:, j].sum(), t, retain_graph=True)[0]
                            for j in range(3)], -1)
        torch.testing.assert_close(actual, s[derivative], atol=1e-11, rtol=1e-11)
    x = (s["xi"] + 0.1 * torch.randn_like(delta)).detach().requires_grad_()
    cov = matrix(s)
    y = x - s["xi"]
    logf = s["mass"].log().squeeze(-1) - 0.5 * (torch.linalg.slogdet(cov)[1]
           + (y * torch.linalg.solve(cov, y[..., None]).squeeze(-1)).sum(-1))
    partial = torch.autograd.grad(logf.sum(), t, retain_graph=True)[0]
    score = torch.autograd.grad(logf.sum(), x, create_graph=True)[0]
    lap_logf = sum(torch.autograd.grad(score[:, j].sum(), x, retain_graph=True)[0][:, j:j+1] for j in range(3))
    u, g = controls(s, x, mode=mode)
    divu = sum(torch.autograd.grad(u[:, j].sum(), x, retain_graph=True)[0][:, j:j+1] for j in range(3))
    rhs = -divu - (u * score).sum(-1, keepdim=True) + path.sigma**2 / 2 * (lap_logf + score.square().sum(-1, keepdim=True)) + g
    torch.testing.assert_close(partial, rhs, atol=1e-9, rtol=1e-9)
    loss = finite_pair_integrand(s, mode=mode).mean()
    loss.backward()
    parameter = path.var_net.last.bias
    expected, original, h = parameter.grad[1].item(), parameter[1].item(), 1e-6
    values = []
    with torch.no_grad():
        for offset in (-h, h):
            parameter[1] = original + offset
            values.append(finite_pair_integrand(path(t, delta, ell), mode=mode).mean().item())
        parameter[1] = original
    assert expected == pytest.approx((values[1] - values[0]) / (2*h), rel=1e-6, abs=1e-8)


def test_sampling_full_covariance_and_analytic_moments():
    path = make_axial()
    t = torch.tensor([[0.43]], dtype=torch.float64)
    delta = torch.tensor([[0.7, -0.8, 0.5]], dtype=torch.float64)
    s = path(t, delta, torch.tensor([[math.log(1.7)]], dtype=torch.float64))
    assert abs(s["var"][0, 0] - s["var"][0, 1]) > 1e-4
    torch.manual_seed(27)
    noise = torch.randn(200_000, 3, dtype=torch.float64)
    targets = make_targets(s, noise)
    x = targets["x"]
    torch.testing.assert_close(torch.cov(x.T), matrix(s)[0], rtol=0.04, atol=1e-4)
    u, g = controls(s, x)
    v = (delta - x) / (1 - t)
    F = (x - delta).square().sum(-1, keepdim=True) / (2 * (1 - t))
    empirical = ((u-v).square().sum(-1).mean(), g.square().mean(), (g*F).mean())
    for actual, expected in zip(empirical, finite_moments(s).values()):
        assert actual.item() == pytest.approx(expected.item(), rel=0.015, abs=0.001)


def test_finite_cost_matches_subtracted_terminal_logarithm():
    path = make_axial().freeze()
    delta = torch.tensor([[0.5, -0.7, 0.3]], dtype=torch.float64)
    ell = torch.tensor([[math.log(1.6)]], dtype=torch.float64)
    finite = CostOracle(path, 128)(delta, [1.6], derivatives=0)[0]
    nodes, weights = np.polynomial.legendre.leggauss(256)
    estimates = []
    for cutoff in (1e-2, 1e-4, 1e-6):
        L = -math.log(cutoff)
        z = torch.tensor((nodes+1)*L/2, dtype=torch.float64)[:, None]
        t = -torch.expm1(-z)
        s = path(t, delta.expand(len(t), -1), ell.expand(len(t), -1))
        # Mean derivative lies in the parallel eigenspace. Variance energy is
        # a trace over all n eigenvalues, including n-1 transverse directions.
        energy = 0.5*s["mass"]*(path.alpha*s["gamma"].square()
                 + s["dotxi"].square().sum(-1, keepdim=True)/(1+s["var"][:, :1]/path.alpha)
                 + (s["dvar_minus_sigma2"].square()/(4*s["var"]+2*s["var"].square()/path.alpha)).sum(-1, keepdim=True))
        integrated = np.dot(weights, (energy*(-z).exp()).numpy().reshape(-1))*L/2
        estimates.append(integrated-path.dimension*path.sigma**2*1.6/2*L)
    assert abs(estimates[-1]-finite) < 5e-5
    assert abs(estimates[-1]-finite) < abs(estimates[0]-finite)


def test_new_config_and_legacy_checkpoint_architecture():
    new = load_config("configs/mouse_ruot_distance_minibatch.yaml")
    old = load_config("configs/mouse_ruot_tanh_minibatch.yaml")
    path = make_path(new, "cpu")
    assert path.parameterization == "axial_distance"
    assert path.mu_net.hidden[0].in_features == 3
    assert path.mu_net.last.out_features == 1
    assert path.var_net.last.out_features == 2
    assert make_path(old, "cpu").parameterization == "diagonal"
    assert new["matching"] == old["matching"]
    assert new["coupling"] == old["coupling"]

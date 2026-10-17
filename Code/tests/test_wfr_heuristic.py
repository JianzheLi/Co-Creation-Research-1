import math

import torch
import pytest

from src.wfr_heuristic import WFRBridgePath, WFRBrownianBridgePath, wfr_static_cost


@pytest.mark.parametrize('alpha', [2.0, 10.0])
def test_static_wfr_cost_uses_alpha_in_action_and_geometry(alpha):
    P = torch.tensor([1.0], dtype=torch.float64)
    Q = torch.tensor([4.0], dtype=torch.float64)
    kappa_same_point = torch.tensor([1.0], dtype=torch.float64)
    kappa_separated = torch.tensor([0.8], dtype=torch.float64)
    torch.testing.assert_close(wfr_static_cost(P, Q, kappa_same_point, alpha), torch.tensor([2*alpha], dtype=torch.float64))
    torch.testing.assert_close(wfr_static_cost(P, Q, kappa_separated, alpha), torch.tensor([3.6*alpha], dtype=torch.float64))


@pytest.mark.parametrize('alpha', [2.0, 10.0])
def test_integrated_dirac_action_matches_static_wfr_cost(alpha):
    import numpy as np
    nodes, weights = np.polynomial.legendre.leggauss(64)
    t = torch.tensor((nodes+1)/2, dtype=torch.float64)[:, None]
    delta = torch.tensor([[1.2, -.5]], dtype=torch.float64).expand(len(t), -1)
    r = torch.full_like(t, 1.7)
    path = WFRBridgePath(alpha=alpha)
    state = path(t, torch.zeros_like(delta), delta, r.log())
    rate = .5*state['mass']*(state['dotxi'].square().sum(-1,keepdim=True)+alpha*state['gamma'].square())
    integrated = (rate.flatten()*torch.tensor(weights/2)).sum()
    exact = wfr_static_cost(torch.tensor(1.), torch.tensor(1.7,dtype=torch.float64),
                            (delta[0].norm()/(2*math.sqrt(alpha))).cos(),alpha)
    torch.testing.assert_close(integrated,exact)


def test_wfr_bridge_hits_endpoints_and_uses_bridge_variance():
    alpha, sigma = 10.0, 0.25
    path = WFRBridgePath(alpha, sigma)
    x0 = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    delta = torch.tensor([[1.4, 0.6]], dtype=torch.float64)
    log_r = torch.tensor([[math.log(2.5)]], dtype=torch.float64)
    t = torch.tensor([[1e-5], [0.37], [1 - 1e-5]], dtype=torch.float64)
    repeated_x0 = x0.expand(len(t), -1)
    repeated_delta = delta.expand(len(t), -1)
    repeated_log_r = log_r.expand(len(t), -1)
    state = path(t, repeated_x0, repeated_delta, repeated_log_r)
    torch.testing.assert_close(state["xi"][0], x0[0] + t[0] * delta[0], atol=2e-5, rtol=0)
    torch.testing.assert_close(state["xi"][-1], x0[0] + delta[0], atol=2e-5, rtol=0)
    torch.testing.assert_close(state["mass"][0], torch.ones(1, dtype=torch.float64), atol=2e-5, rtol=0)
    torch.testing.assert_close(state["mass"][-1], torch.tensor([2.5], dtype=torch.float64), atol=2e-4, rtol=0)
    torch.testing.assert_close(state["var"], sigma ** 2 * t * (1 - t))
    assert torch.all(state["mass"] > 0)
    assert torch.isfinite(state["gamma"]).all()


def test_wfr_bridge_samples_full_gaussian_and_weighted_flow_labels():
    path = WFRBridgePath(alpha=10.0, sigma=0.25)
    x0 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    delta = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    log_r = torch.tensor([[math.log(1.7)]], dtype=torch.float64)
    t = torch.tensor([[0.4]], dtype=torch.float64)
    noise = torch.tensor([[1.2, -0.7]], dtype=torch.float64)
    targets = path.conditional_samples(t, x0, delta, log_r, noise)
    state = path(t, x0, delta, log_r)
    torch.testing.assert_close(targets["x"], state["xi"] + state["var"].sqrt() * noise)
    torch.testing.assert_close(targets["target_s_scaled"], -noise)
    expected_v_scaled = (state["tau"].sqrt() * state["dotxi"]
                         + 0.5 * path.sigma * (1 - 2 * t) * noise)
    torch.testing.assert_close(targets["target_v_scaled"], expected_v_scaled)


def test_brownian_bridge_path_uses_linear_mean_and_log_linear_mass():
    alpha, sigma = 2.0, 0.5
    path = WFRBrownianBridgePath(alpha, sigma)
    x0 = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    delta = torch.tensor([[1.4, 0.6]], dtype=torch.float64)
    log_r = torch.tensor([[math.log(2.5)]], dtype=torch.float64)
    t = torch.tensor([[0.2], [0.6]], dtype=torch.float64)
    state = path(t, x0.expand(2, -1), delta.expand(2, -1), log_r.expand(2, -1))
    torch.testing.assert_close(state["xi"], x0 + t * delta)
    torch.testing.assert_close(state["mass"], torch.exp(t * log_r))
    torch.testing.assert_close(state["gamma"], log_r.expand(2, -1))
    torch.testing.assert_close(state["var"], sigma**2 * t * (1 - t))
    torch.testing.assert_close(state["dotxi"], delta.expand(2, -1))


def test_brownian_bridge_samples_full_gaussian_and_probability_flow_target():
    path = WFRBrownianBridgePath(alpha=2.0, sigma=0.5)
    x0 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    delta = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    log_r = torch.tensor([[math.log(1.7)]], dtype=torch.float64)
    t = torch.tensor([[0.4]], dtype=torch.float64)
    noise = torch.tensor([[1.2, -0.7]], dtype=torch.float64)
    state = path(t, x0, delta, log_r)
    targets = path.conditional_samples(t, x0, delta, log_r, noise)
    torch.testing.assert_close(targets["x"], state["xi"] + state["var"].sqrt() * noise)
    torch.testing.assert_close(targets["target_s_scaled"], -noise)
    expected = state["tau"].sqrt() * delta + 0.5 * path.sigma * (1 - 2 * t) * noise
    torch.testing.assert_close(targets["target_v_scaled"], expected)


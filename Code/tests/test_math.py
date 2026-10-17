import math
import numpy as np
import pytest
import torch
from torch import nn
from scipy.special import logsumexp

from src.pair_path import PairPath, controls, exprel
from src.pair_cost import CostOracle, finite_moments, finite_pair_integrand
from src.coupling import HermiteCost, OracleError, solve_dual
from src.matching import make_targets, matching_loss
from src.sample import rollout
from src.evaluate import evaluate_snapshot, particle_mass_and_weights
from src.data import load_gene, PairIndex


def path_fixture(perturbed=True, mode="quadratic_minimal"):
    torch.manual_seed(123)
    path = PairPath(width=8, layers=2, control_mode=mode).double()
    if perturbed:
        with torch.no_grad():
            for net in (path.mu_net, path.var_net, path.mass_net):
                net.last.weight.normal_(0, 0.03)
                net.last.bias.normal_(0, 0.02)
    return path


def conditions(n=7):
    torch.manual_seed(11)
    return (torch.linspace(0.1, 0.9, n).reshape(-1, 1).double().requires_grad_(),
            torch.randn(n, 2, dtype=torch.float64),
            torch.linspace(-1.5, 1.5, n).reshape(-1, 1).double())


def test_endpoints_and_variance_slopes():
    path = path_fixture()
    _, delta, ell = conditions()
    for t, endpoint in ((0.0, False), (1.0, True)):
        state = path(torch.full_like(ell, t), delta, ell)
        torch.testing.assert_close(state["xi"], delta if endpoint else torch.zeros_like(delta))
        torch.testing.assert_close(state["mass"], ell.exp() if endpoint else torch.ones_like(ell))
        torch.testing.assert_close(state["var"], torch.zeros_like(delta))
        torch.testing.assert_close(state["dotvar"], torch.full_like(delta, (-1 if endpoint else 1) * path.sigma ** 2))
        assert torch.isfinite(finite_pair_integrand(state)).all()


def test_time_jvp_matches_autograd_and_finite_difference():
    path = path_fixture()
    t, delta, ell = conditions()
    state = path(t, delta, ell)
    for value, derivative in (("xi", "dotxi"), ("var", "dotvar")):
        grad = torch.cat([torch.autograd.grad(state[value][:, j].sum(), t, retain_graph=True)[0]
                          for j in range(2)], dim=1)
        torch.testing.assert_close(grad, state[derivative], rtol=1e-10, atol=1e-10)
        h = 1e-5
        finite = (path(t + h, delta, ell)[value] - path(t - h, delta, ell)[value]) / (2 * h)
        torch.testing.assert_close(finite, state[derivative], rtol=1e-6, atol=1e-7)
    dlogm = torch.autograd.grad(state["mass"].log().sum(), t)[0]
    torch.testing.assert_close(dlogm, state["gamma"], rtol=1e-10, atol=1e-10)


def test_integrand_parameter_gradient_and_exprel():
    path = path_fixture()
    t, delta, ell = conditions(3)
    loss = finite_pair_integrand(path(t, delta, ell)).mean()
    loss.backward()
    parameter = path.mass_net.last.weight
    analytic = parameter.grad[0, 0].item()
    original = parameter[0, 0].item()
    h = 1e-6
    with torch.no_grad():
        parameter[0, 0] = original + h
        plus = finite_pair_integrand(path(t, delta, ell)).mean().item()
        parameter[0, 0] = original - h
        minus = finite_pair_integrand(path(t, delta, ell)).mean().item()
        parameter[0, 0] = original
    assert analytic == pytest.approx((plus - minus) / (2 * h), rel=1e-6, abs=1e-7)
    z = torch.tensor([0, 1e-9, -1e-9, 0.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(exprel, (z,), eps=1e-7, atol=1e-6)


def test_analytic_space_moments_against_gaussian_mc():
    path = path_fixture()
    state = path(torch.tensor([[0.43]], dtype=torch.float64), torch.tensor([[0.7, -0.3]], dtype=torch.float64),
                 torch.tensor([[math.log(1.7)]], dtype=torch.float64))
    torch.manual_seed(27)
    noise = torch.randn(250_000, 2, dtype=torch.float64)
    x = state["xi"] + state["var"].sqrt() * noise
    u, g = controls(state, x)
    x1 = state["delta"]
    v = (x1 - x) / (1 - state["t"])
    F = (x - x1).square().sum(-1, keepdim=True) / (2 * (1 - state["t"]))
    empirical = ((u - v).square().sum(-1).mean(), g.square().mean(), (g * F).mean())
    exact = finite_moments(state)
    for actual, expected in zip(empirical, exact.values()):
        assert actual.item() == pytest.approx(expected.item(), rel=0.012, abs=0.001)


def test_pde_compatibility_and_minimal_control_energy():
    path = path_fixture()
    t, delta, ell = conditions()
    state = path(t, delta, ell)
    torch.manual_seed(4)
    x = (state["xi"] + state["var"].sqrt() * torch.randn_like(delta)).detach().requires_grad_()
    u, g = controls(state, x)
    y, var = x - state["xi"], state["var"]
    score = -y / var
    partial_logf = state["gamma"] + (y * state["dotxi"] / var
                      + 0.5 * (y.square() / var - 1) * state["dotvar"] / var).sum(-1, keepdim=True)
    divu = sum(torch.autograd.grad(u[:, j].sum(), x, retain_graph=True)[0][:, j:j+1] for j in range(2))
    rhs = -divu - (u * score).sum(-1, keepdim=True) + 0.5 * path.sigma ** 2 * (score.square() - 1 / var).sum(-1, keepdim=True) + g
    torch.testing.assert_close(partial_logf, rhs, rtol=1e-9, atol=1e-9)
    minimal = 0.5 * state["mass"] * (path.alpha * state["gamma"].square()
             + (state["dotxi"].square() / (1 + var / path.alpha)
             + state["dvar_minus_sigma2"].square() / (4 * var + 2 * var.square() / path.alpha)).sum(-1, keepdim=True))
    uniform = 0.5 * state["mass"] * (path.alpha * state["gamma"].square()
             + (state["dotxi"].square() + state["dvar_minus_sigma2"].square() / (4 * var)).sum(-1, keepdim=True))
    assert torch.all(minimal <= uniform + 1e-12)


def test_uniform_brownian_bridge_reference_and_ratio_derivatives():
    path = path_fixture(False, "uniform_growth")
    delta = np.array([[1, 2], [-0.2, 0.7]])
    oracle = CostOracle(path)
    cost = oracle(delta, np.ones(2), derivatives=0)
    np.testing.assert_allclose(cost, 0.5 * np.square(delta).sum(1) - path.dimension * path.sigma ** 2 / 2, atol=1e-12)
    r = np.array([0.7, 1.9])
    c, cr, crr = oracle(delta, r)
    h = 1e-4
    cm, cp = oracle(delta, r - h, 0), oracle(delta, r + h, 0)
    np.testing.assert_allclose(cr, (cp - cm) / (2 * h), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(crr, (cp - 2 * c + cm) / h ** 2, rtol=2e-6, atol=2e-6)
    assert not np.allclose(CostOracle(path_fixture(False))(delta, np.ones(2), 0), cost)


def test_finite_cost_equals_renormalized_energy_limit():
    path = path_fixture().freeze()
    delta = torch.tensor([[0.5, -0.7]], dtype=torch.float64)
    ell = torch.tensor([[math.log(1.6)]], dtype=torch.float64)
    finite = CostOracle(path, 128)(delta, [1.6], derivatives=0)[0]
    nodes, weights = np.polynomial.legendre.leggauss(256)
    estimates = []
    for cutoff in (1e-2, 1e-4, 1e-6):
        # z=-log(1-t) resolves the terminal logarithm with fixed smooth nodes.
        L = -math.log(cutoff)
        z = torch.tensor((nodes + 1) * L / 2, dtype=torch.float64).reshape(-1, 1)
        t = -torch.expm1(-z)
        s = path(t, delta.expand(len(t), -1), ell.expand(len(t), -1))
        energy = 0.5 * s["mass"] * (10 * s["gamma"].square()
                 + (s["dotxi"].square() / (1 + s["var"] / 10)
                 + s["dvar_minus_sigma2"].square() / (4 * s["var"] + 2 * s["var"].square() / 10)).sum(-1, keepdim=True))
        integrated = np.dot(weights, (energy * (-z).exp()).detach().numpy().reshape(-1)) * L / 2
        estimates.append(integrated - path.dimension * path.sigma ** 2 * 1.6 / 2 * L)
    assert abs(estimates[-1] - finite) < 3e-5
    assert abs(estimates[-1] - finite) < abs(estimates[0] - finite)


def convex_oracle(offset):
    def oracle(r):
        return offset + 20 * (1 + r - 2 * np.sqrt(r)), 20 * (1 - 1 / np.sqrt(r)), 10 / r ** 1.5
    return oracle


def test_dual_coupling_marginals_fenchel_and_nonconvergence():
    offset = np.array([[0.1, 0.3, 0.2], [0.4, 0.05, 0.2]])
    oracle = convex_oracle(offset)
    a, b = np.array([0.4, 0.6]), np.array([0.3, 0.5, 0.4])
    result = solve_dual(a, b, oracle, epsilon=0.2)
    assert result.status == "converged"
    np.testing.assert_allclose(result.P.sum(1), a, rtol=1e-5)
    np.testing.assert_allclose(result.Q.sum(0), b, rtol=1e-8)
    assert np.all((result.r >= 0.1) & (result.r <= 10))
    assert np.allclose(result.Q, result.P * result.r)
    interior = (result.r > 0.1) & (result.r < 10)
    np.testing.assert_allclose(oracle(result.r)[1][interior], np.broadcast_to(result.lam, result.r.shape)[interior], atol=1e-9)
    assert result.history[-1]["gap_is_certificate"] is False
    failed = solve_dual(a, b, oracle, epsilon=0.2, max_outer_iterations=1)
    assert failed.status == "max_outer_iterations"
    with pytest.raises(OracleError, match="infeasible_ratio_range"):
        solve_dual(a, b * 20, oracle)


def test_single_hermite_query_and_nonconvex_rejection():
    offset = np.array([[0.1, 0.2], [0.3, 0.4]])
    grid = np.geomspace(0.1, 10, 129)
    values = convex_oracle(offset[..., None])(grid)
    table = HermiteCost(grid, values[0], np.broadcast_to(values[1], values[0].shape))
    r = np.array([[0.15, 0.8], [2.2, 8.5]])
    direct, tab = convex_oracle(offset)(r), table(r)
    np.testing.assert_allclose(tab[0], direct[0], atol=1e-5)
    np.testing.assert_allclose(tab[1], direct[1], atol=1e-4)
    lam = np.array([-100, 5])
    R, star, _ = table.conjugate(lam)
    np.testing.assert_allclose(table(R)[0] + star, lam * R, atol=1e-12)
    assert np.all(R[:, 0] == 0.1)
    result = solve_dual([0.4, 0.6], [0.5, 0.7], table, epsilon=0.2)
    assert result.status == "converged"
    with pytest.raises(OracleError, match="cost_oracle_nonconvex"):
        HermiteCost(grid, -np.square(grid)[None, :], -2 * grid[None, :])


def test_scaled_matching_labels_equal_raw_controls():
    path = path_fixture()
    t, delta, ell = conditions()
    state = path(t, delta, ell)
    torch.manual_seed(9)
    targets = make_targets(state, torch.randn_like(delta))
    u, g = controls(state, targets["x"])
    raw_score = -(targets["x"] - state["xi"]) / state["var"]
    torch.testing.assert_close(targets["target_g"], g)
    torch.testing.assert_close(targets["target_s_scaled"], targets["sqrt_ws"] * raw_score)
    torch.testing.assert_close(targets["target_v_scaled"], targets["sqrt_wf"] * (u - 0.5 * path.sigma ** 2 * raw_score))
    prediction = (torch.zeros_like(delta), torch.zeros_like(ell), torch.zeros_like(delta))
    loss, _ = matching_loss(prediction, targets, torch.ones(len(t)))
    double_mass = dict(targets, mass=2 * targets["mass"])
    doubled, _ = matching_loss(prediction, double_mass, torch.ones(len(t)))
    torch.testing.assert_close(doubled, 2 * loss)


def test_flow_matching_loss_keeps_original_local_time_weight():
    target = dict(target_v_scaled=torch.zeros((2, 2)), target_g=torch.zeros((2, 1)),
                  target_s_scaled=torch.zeros((2, 2)), sqrt_ws=torch.zeros((2, 1)),
                  mass=torch.ones((2, 1)), sqrt_wf=torch.tensor([[0.1], [0.5]]))
    prediction = (torch.ones((2, 2)), torch.zeros((2, 1)), torch.zeros((2, 2)))
    loss, parts = matching_loss(prediction, target, torch.ones(2))
    torch.testing.assert_close(parts["flow"], torch.tensor(0.26))
    torch.testing.assert_close(loss, torch.tensor(0.26))


def test_scaled_flow_supervision_remains_finite_near_endpoints():
    path = path_fixture()
    _, delta, ell = conditions(2)
    t = torch.tensor([[1e-30], [1 - 1e-15]], dtype=torch.float64)
    state = path(t, delta[:2], ell[:2])
    targets = make_targets(state, torch.ones_like(delta[:2]))
    assert torch.isfinite(targets["target_v_scaled"]).all()


def test_mixture_regression_matches_explicit_mass_posterior():
    # Fixed x: expectation of the condition gradient equals regression against
    # the explicit mass-weighted posterior. Posterior is used ONLY in this test.
    means = torch.tensor([[-1.0, 0], [0.7, 1], [1.3, -0.4]], dtype=torch.float64)
    var = torch.tensor([[0.2, 0.3], [0.4, 0.2], [0.3, 0.5]], dtype=torch.float64)
    P = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    mass = torch.tensor([0.7, 1.9, 1.2], dtype=torch.float64)
    x = torch.tensor([0.3, 0.1], dtype=torch.float64)
    density = torch.exp(-0.5 * ((x - means).square() / var).sum(1)) / (2 * math.pi * var.prod(1).sqrt())
    posterior = P * mass * density
    posterior /= posterior.sum()
    labels = torch.tensor([[0.1, 0.2, 0.3, 0.5, -0.4], [0.3, 0.8, -0.2, 0.4, 0.2], [-0.1, 0.7, 0.4, -0.3, 0.9]], dtype=torch.float64)
    prediction = torch.randn(5, dtype=torch.float64, requires_grad=True)
    conditional_loss = (posterior * (prediction - labels).square().sum(1)).sum()
    mixture_target = (posterior[:, None] * labels).sum(0)
    mixture_loss = (prediction - mixture_target).square().sum()
    grad1 = torch.autograd.grad(conditional_loss, prediction)[0]
    grad2 = torch.autograd.grad(mixture_loss, prediction)[0]
    torch.testing.assert_close(grad1, grad2, rtol=1e-12, atol=1e-12)


def test_pot_zero_distance_mass_scaling_and_known_distances():
    x = np.array([[0.0, 0], [1.0, 0], [0.0, 2]])
    logw = np.full(3, -math.log(3))
    baseline = evaluate_snapshot(x, logw, x, n_initial_observed=3)
    assert baseline["W1_normalized"] == pytest.approx(0, abs=1e-12)
    assert baseline["W2_normalized"] == pytest.approx(0, abs=1e-12)
    assert baseline["mass_abs_error"] < 1e-12
    scaled = evaluate_snapshot(x, logw + math.log(2), x, n_initial_observed=3)
    assert scaled["mass_pred"] == pytest.approx(2)
    assert scaled["W1_normalized"] == pytest.approx(baseline["W1_normalized"])
    known = evaluate_snapshot(np.array([[0.0, 0]]), [0], np.array([[1.0, 0], [3.0, 0]]), mass_true=1)
    assert known["W1_normalized"] == pytest.approx(2)
    assert known["W2_normalized"] == pytest.approx(math.sqrt(5))


def test_particle_weight_evaluation_is_log_stable_and_rejects_mass_overflow():
    mass, weights, log_mass = particle_mass_and_weights(np.array([700.0, 699.0]))
    assert log_mass == pytest.approx(700 + math.log1p(math.exp(-1)))
    assert np.isfinite(mass)
    np.testing.assert_allclose(weights.sum(), 1.0)
    np.testing.assert_allclose(weights, [1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))])
    with pytest.raises(FloatingPointError, match="exceeds float64 range"):
        particle_mass_and_weights(np.array([710.0, 0.0]))


def test_continuous_fixed_particle_rollout_carries_raw_mass():
    class ConstantFeedback(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
        def forward(self, t, x):
            return torch.ones_like(x), torch.full_like(t, math.log(2)), torch.zeros_like(x)
    initial = np.zeros((4, 2))
    increments = np.zeros((40, 4, 2))
    result = rollout(ConstantFeedback(), initial, n_particles=4, steps_per_unit=10,
                     initial_x=initial, normal_increments=increments, trace_indices=[0])
    np.testing.assert_allclose(result["positions"][-1], 4, atol=2e-6)
    np.testing.assert_allclose(np.exp(logsumexp(result["log_weights"], axis=1)), [1, 2, 4, 8, 16], rtol=1e-6)
    assert result["positions"].shape == (5, 4, 2)
    np.testing.assert_allclose(result["trace_times"], np.arange(41) / 10)


def test_rollout_can_capture_an_interior_time_without_restarting_the_path():
    class ConstantFeedback(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, t, x):
            return torch.ones_like(x), torch.full_like(t, math.log(2)), torch.zeros_like(x)

    initial = np.zeros((3, 2), dtype=np.float32)
    result = rollout(
        ConstantFeedback(), initial, times=(0, 1), n_particles=3,
        steps_per_unit=10, initial_x=initial, stochastic=False,
        capture_times=(0.5,),
    )
    np.testing.assert_allclose(result["capture_positions"][0], 0.5, atol=1e-7)
    np.testing.assert_allclose(
        np.exp(result["capture_log_weights"][0]).sum(), math.sqrt(2), rtol=1e-6
    )
    np.testing.assert_allclose(result["positions"][-1], 1.0, atol=1e-7)


def test_rollout_action_uses_score_corrected_drift_and_raw_particle_mass():
    class ConstantFeedback(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
        def forward(self, t, x):
            return torch.ones_like(x), torch.full_like(t, math.log(2)), torch.ones_like(x)

    n, steps, sigma, alpha = 4, 10, 0.25, 10
    increments = np.zeros((4 * steps, n, 2))
    result = rollout(ConstantFeedback(), np.zeros((n, 2)), n_particles=n,
                     steps_per_unit=steps, sigma=sigma,
                     initial_x=np.zeros((n, 2)), normal_increments=increments, action_alpha=alpha)
    drift = 1 + 0.5 * sigma ** 2
    growth = math.log(2)
    dt = 1 / steps
    kinetic_rate = 0.5 * 2 * drift ** 2
    growth_rate = alpha / 2 * growth ** 2
    expected_kinetic = dt * sum(math.exp(growth * step * dt) * kinetic_rate for step in range(4 * steps))
    expected_growth = dt * sum(math.exp(growth * step * dt) * growth_rate for step in range(4 * steps))
    assert result["positions"][-1, 0, 0] == pytest.approx(4 * drift)
    assert result["action"]["kinetic"] == pytest.approx(expected_kinetic)
    assert result["action"]["growth"] == pytest.approx(expected_growth)
    assert result["action"]["total"] == pytest.approx(expected_kinetic + expected_growth)
    assert len(result["action"]["intervals"]) == 4


def test_official_data_pair_index_and_subsampling_masses():
    full = load_gene()
    assert [len(s.x) for s in full] == [400, 442, 530, 690, 969]
    assert PairIndex(full).size == 1445370
    small = load_gene(max_points=4)
    np.testing.assert_allclose([s.mass for s in small], [s.mass for s in full])
    index = PairIndex(small)
    actual = index.delta(np.arange(index.size)).numpy()
    expected = np.concatenate([(b.x[None, :, :] - a.x[:, None, :]).reshape(-1, 2) for a, b in zip(small[:-1], small[1:])])
    np.testing.assert_array_equal(actual, expected)

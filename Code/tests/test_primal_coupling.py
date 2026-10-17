import numpy as np
import pytest
import torch
from scipy.optimize import minimize, LinearConstraint

from src.primal_coupling_gpu import kl_project, perspective_value_gradient, solve_primal_gpu
from src.coupling import HermiteCost, OracleError, solve_dual


def test_perspective_gradient_matches_autograd():
    p = torch.tensor([[.2, .3], [.1, .4]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[.3, .2], [.2, .5]], dtype=torch.float64, requires_grad=True)
    r = q/p
    c = 20*(1+r-2*r.sqrt())
    cr = 20*(1-r.rsqrt())
    f, gp, gq = perspective_value_gradient(p, q, p.log(), c, cr, .0625)
    expected = torch.autograd.grad(f, (p, q))
    torch.testing.assert_close(gp, expected[0])
    torch.testing.assert_close(gq, expected[1])


def test_kl_joint_projection_matches_constrained_reference():
    u = np.array([[.01, 4.], [.2, 1.]])
    v = np.array([[3., .04], [.5, .02]])
    a, b = np.array([.4, .6]), np.array([.55, .45])
    lo, hi = .5, 2.
    lp, lq, _ = kl_project(torch.tensor(np.log(u)), torch.tensor(np.log(v)),
                           torch.tensor(a), torch.tensor(b), lo, hi, tolerance=1e-11)
    p, q = lp.exp().numpy(), lq.exp().numpy()
    np.testing.assert_allclose(p.sum(1), a, atol=1e-10)
    np.testing.assert_allclose(q.sum(0), b, atol=1e-10)
    assert (q >= lo*p-1e-12).all() and (q <= hi*p+1e-12).all()
    matrix = np.zeros((4, 8))
    matrix[0, :2] = 1; matrix[1, 2:4] = 1
    matrix[2, [4, 6]] = 1; matrix[3, [5, 7]] = 1
    cone = np.concatenate((-lo*np.eye(4), np.eye(4)), axis=1)
    cone2 = np.concatenate((hi*np.eye(4), -np.eye(4)), axis=1)
    reference = np.r_[u.flatten(), v.flatten()]
    objective = lambda x: np.sum(x*np.log(x/reference)-x+reference)
    solved = minimize(objective, np.full(8, .25), jac=lambda x: np.log(x/reference),
        bounds=[(1e-12, None)]*8, constraints=[LinearConstraint(matrix, np.r_[a,b], np.r_[a,b]),
        LinearConstraint(cone, 0, np.inf), LinearConstraint(cone2, 0, np.inf)],
        method="SLSQP", options=dict(ftol=1e-12, maxiter=1000))
    assert solved.success
    assert abs(objective(np.r_[p.flatten(), q.flatten()])-solved.fun) < 1e-8


def test_direct_primal_agrees_with_dual_reference():
    offset = np.array([[.1, .2, .3], [.4, .2, .1]])
    class AnalyticCost:
        def __call__(self, r):
            return self.value(r), 20*(1-r.rsqrt())
        def value(self, r):
            return torch.tensor(offset, dtype=r.dtype, device=r.device)+20*(1+r-2*r.sqrt())
    a, b = np.array([.4, .6]), np.array([.3, .5, .4])
    actual = solve_primal_gpu(a, b, AnalyticCost(), epsilon=.2, device="cpu",
                              stationarity_tolerance=1e-5, projection_tolerance=1e-5)
    assert actual.status == "converged"
    grid = np.geomspace(.1, 10., 1025)
    cost = offset[..., None]+20*(1+grid-2*np.sqrt(grid))
    slopes = np.broadcast_to(20*(1-1/np.sqrt(grid)), cost.shape).copy()
    expected = solve_dual(a, b, HermiteCost(grid, cost, slopes), epsilon=.2,
                          marginal_tolerance=1e-8, root_logmass_tolerance=1e-10, max_outer_iterations=10000)
    assert expected.status == "converged"
    np.testing.assert_allclose(actual.P, expected.P, atol=3e-5, rtol=1e-3)
    np.testing.assert_allclose(actual.Q, expected.Q, atol=3e-5, rtol=1e-3)
    objectives = [row["objective"] for row in actual.history]
    assert np.diff(objectives).max() < 1e-10
    assert actual.history[-1]["projected_gradient_relative_l1"] < 1e-5
    assert actual.history[0]["projection_tolerance"] == 1e-5
    assert actual.history[0]["row_relative_tv"] < 1e-9
    assert len(actual.history) > 1  # Feasibility alone must not stop optimization.


def test_primal_solver_can_resume_from_feasible_iterate():
    class AnalyticCost:
        def __call__(self, r):
            return self.value(r), 20*(1-r.rsqrt())
        def value(self, r):
            return 20*(1+r-2*r.sqrt())
    a, b = np.array([.4, .6]), np.array([.3, .5, .4])
    partial = solve_primal_gpu(a, b, AnalyticCost(), epsilon=.2, device="cpu",
                               max_iterations=3, stationarity_tolerance=1e-8)
    assert partial.status == "max_outer_iterations"
    resumed = solve_primal_gpu(a, b, AnalyticCost(), epsilon=.2, device="cpu",
                               stationarity_tolerance=1e-5, initial_log_P=partial.log_P,
                               initial_log_Q=partial.log_P+np.log(partial.r),
                               initial_step=2*partial.history[-1]["step_size"])
    reference = solve_primal_gpu(a, b, AnalyticCost(), epsilon=.2, device="cpu",
                                 stationarity_tolerance=1e-5)
    assert resumed.status == "converged"
    np.testing.assert_allclose(resumed.P, reference.P, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(resumed.Q, reference.Q, atol=1e-7, rtol=1e-5)


def test_nonmonotone_fallback_accepts_bounded_ascent_at_meaningful_step(monkeypatch):
    class Cost:
        def value(self, ratio):
            penalty = (ratio.log().abs() > 1e-8).to(ratio.dtype)
            return 1000. + penalty

        def __call__(self, ratio):
            return self.value(ratio), torch.zeros_like(ratio)

    candidate = torch.tensor([[.4, .1], [.1, .4]], dtype=torch.float64)
    candidate_q = torch.tensor([[.1, .4], [.4, .1]], dtype=torch.float64)

    def fixed_projection(log_p, log_q, a, b, ratio_min, ratio_max, *args, **kwargs):
        return candidate.to(log_p.device).log(), candidate_q.to(log_q.device).log(), 1

    monkeypatch.setattr("src.primal_coupling_gpu.kl_project", fixed_projection)
    result = solve_primal_gpu(
        np.array([.5, .5]), np.array([.5, .5]), Cost(), epsilon=.2, device="cpu",
        max_iterations=1, stationarity_tolerance=3e-3, allow_nonmonotone_fallback=True,
        nonmonotone_min_step_fraction=2**-12)

    record = result.history[0]
    assert record["line_search_nonmonotone_fallback"]
    assert record["line_search_nonmonotone_step_fraction"] >= 2**-12
    assert record["line_search_majorization_fallback"]
    assert record["objective_change"] > 0
    assert record["relative_objective_change"] <= 3e-3
    np.testing.assert_allclose(result.P, candidate.numpy(), atol=1e-12)
    np.testing.assert_allclose(result.Q, candidate_q.numpy(), atol=1e-12)


def test_majorized_tiny_ascent_needs_no_tighter_projection_pass(monkeypatch):
    class Cost:
        def value(self, ratio):
            penalty = (ratio.log().abs() > 1e-8).to(ratio.dtype)
            return 1000. + 0.001 * penalty

        def __call__(self, ratio):
            return self.value(ratio), torch.zeros_like(ratio)

    candidate = torch.tensor([[.4, .1], [.1, .4]], dtype=torch.float64)
    candidate_q = torch.tensor([[.1, .4], [.4, .1]], dtype=torch.float64)
    calls = 0

    def fixed_projection(log_p, log_q, *args, **kwargs):
        nonlocal calls
        calls += 1
        return candidate.log(), candidate_q.log(), 1

    monkeypatch.setattr("src.primal_coupling_gpu.kl_project", fixed_projection)
    result = solve_primal_gpu(
        np.array([.5, .5]), np.array([.5, .5]), Cost(), epsilon=1e-6, device="cpu",
        max_iterations=1, stationarity_tolerance=3e-3, projection_tolerance=5e-5,
        allow_nonmonotone_fallback=True, nonmonotone_min_step_fraction=2**-12)

    record = result.history[0]
    assert calls == 1
    assert record["line_search_small_increase_fallback"]
    assert record["objective_change"] > 0
    assert record["relative_objective_change"] <= 3e-3**2
    assert record["step_size"] == 1.0
    assert record["projection_tolerance"] == 5e-5
    np.testing.assert_allclose(result.P, candidate.numpy(), atol=1e-12)
    np.testing.assert_allclose(result.Q, candidate_q.numpy(), atol=1e-12)


def test_tighter_retry_accepts_bounded_ascent_at_meaningful_step(monkeypatch):
    class Cost:
        def value(self, ratio):
            penalty = (ratio.log().abs() > 1e-8).to(ratio.dtype)
            return 1000. + penalty

        def __call__(self, ratio):
            return self.value(ratio), torch.zeros_like(ratio)

    candidate = torch.tensor([[.4, .1], [.1, .4]], dtype=torch.float64)
    candidate_q = torch.tensor([[.1, .4], [.4, .1]], dtype=torch.float64)
    calls = 0

    def fail_first_precision_then_return_bounded_ascent(log_p, log_q, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 12:
            raise OracleError("primal_projection_not_converged", {})
        return candidate.log(), candidate_q.log(), 1

    monkeypatch.setattr("src.primal_coupling_gpu.kl_project",
                        fail_first_precision_then_return_bounded_ascent)
    result = solve_primal_gpu(
        np.array([.5, .5]), np.array([.5, .5]), Cost(), epsilon=.2, device="cpu",
        max_iterations=1, stationarity_tolerance=3e-3,
        allow_nonmonotone_fallback=True, nonmonotone_min_step_fraction=2**-12)

    record = result.history[0]
    assert calls == 13
    assert record["line_search_majorization_fallback"]
    assert record["line_search_nonmonotone_fallback"]
    assert record["objective_change"] > 0
    assert record["relative_objective_change"] <= 3e-3
    assert record["step_size"] >= 1.0
    np.testing.assert_allclose(result.P, candidate.numpy(), atol=1e-12)
    np.testing.assert_allclose(result.Q, candidate_q.numpy(), atol=1e-12)


def test_fallback_mode_does_not_accept_numerically_inert_subfloor_step(monkeypatch):
    class Cost:
        def value(self, ratio):
            penalty = torch.tensor([[1., 0.], [0., 1.]], dtype=ratio.dtype,
                                   device=ratio.device)
            return 1000. + penalty

        def __call__(self, ratio):
            return self.value(ratio), torch.zeros_like(ratio)

    calls = 0

    def projection_that_only_returns_noop_below_floor(log_p, log_q, *args, **kwargs):
        nonlocal calls
        calls += 1
        # In each line-search precision pass, force the first three trials
        # (steps 1, 1/2, 1/4) to fail projection. Smaller trials return the
        # current feasible coupling, whose objective is unchanged despite a
        # nonzero gradient. The solver must not accept that subfloor no-op.
        pass_starts = (1, 13, 25)
        if any(start <= calls < start + 3 for start in pass_starts):
            raise OracleError("primal_projection_not_converged", {})
        p = torch.full_like(log_p, .25)
        return p.log(), p.log(), 1

    monkeypatch.setattr("src.primal_coupling_gpu.kl_project", projection_that_only_returns_noop_below_floor)
    with pytest.raises(OracleError) as error:
        solve_primal_gpu(
            np.array([.5, .5]), np.array([.5, .5]), Cost(), epsilon=.2, device="cpu",
            max_iterations=1, stationarity_tolerance=3e-3, allow_nonmonotone_fallback=True,
            nonmonotone_min_step_fraction=.25)
    assert error.value.status == "primal_line_search_failed"
    assert calls >= 25

import numpy as np
import pytest
import torch
from torch import nn

from src.cost_distillation import (CostNetwork, DistilledCostOracle, loss_on_grid,
                                  reduce_squared_error, check_deployment_curvature,
                                  sample_continuous_conditions, sample_boundary_indices)
from src.neural_coupling import NeuralPairCost
from src.coupling_gpu import solve_dual_gpu
from src.pair_path import PairPath


def forbidden(*args, **kwargs):
    raise AssertionError("path forward/quadrature must not run during student queries")


def test_strict_loss_keeps_worst_point_visible_and_differentiable():
    error = torch.zeros(4096, dtype=torch.float64)
    error[0] = 2.
    error.requires_grad_(True)
    loss = reduce_squared_error(error, "mean_plus_max")
    torch.testing.assert_close(loss, torch.tensor(4+4/4096, dtype=torch.float64))
    loss.backward()
    assert error.grad[0] >= 4
    assert reduce_squared_error(error.detach(), "mean") == 4/4096


def test_strict_loss_has_finite_zero_error_gradient():
    error = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    loss = reduce_squared_error(error, "mean_plus_max")
    loss.backward()
    assert loss == 0 and torch.isfinite(error.grad).all() and error.grad.abs().max() == 0


def test_continuous_sampler_is_uniform_in_physical_ratio_and_not_on_grid():
    torch.manual_seed(819)
    d, r = sample_continuous_conditions(20000, 3., .1, 10., torch.zeros((), dtype=torch.float64))
    assert 0 <= d.min() and d.max() <= 3.
    assert .1 <= r.min() and r.max() <= 10.
    assert abs(float(r.mean())-5.05) < .08
    assert abs(float(d.mean())-1.5) < .03
    d_grid_index = d/3*128
    assert bool((d_grid_index != d_grid_index.round()).all())
    d2, r2 = sample_continuous_conditions(20000, 3., .1, 10., d)
    assert not torch.equal(d, d2) and not torch.equal(r, r2)


def test_configurable_max_error_weight():
    error = torch.tensor([1., 2.], dtype=torch.float64)
    assert reduce_squared_error(error, "mean_plus_max", 1.) == 6.5
    assert reduce_squared_error(error, "mean_plus_max", 10.) == 42.5


def test_dense_boundary_subsampling_preserves_corners_and_batch_size():
    boundary = torch.arange(100, dtype=torch.int64)
    corners = torch.tensor([0, 20, 70, 99])
    picked = sample_boundary_indices(boundary, corners, 16)
    assert len(picked) == 16
    assert torch.isin(corners, picked).all() and torch.isin(picked, boundary).all()
    torch.testing.assert_close(sample_boundary_indices(boundary, corners, 100), boundary)
    assert len(sample_boundary_indices(boundary, corners, 0)) == 0


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_student_autodiff_without_teacher(device, monkeypatch):
    torch.manual_seed(83)
    path = PairPath(width=4, layers=1, parameterization="axial_distance").double().to(device)
    model = CostNetwork(3., 0.1, 10., width=16, layers=2).double().to(device)
    monkeypatch.setattr(np.polynomial.legendre, "leggauss", forbidden)
    oracle = DistilledCostOracle(path, model, 0.1, 10., 3., condition_chunk=3)
    monkeypatch.setattr(path, "forward", forbidden)
    delta = torch.tensor([[0., 0.], [0.3, 0.4], [-0.4, 0.3]], dtype=torch.float64, device=device)
    r = torch.tensor([0.4, 0.9, 1.8], dtype=torch.float64, device=device)
    cost, slope, curvature = oracle.tensor_query(delta, r, 2)
    h = 1e-4
    minus, plus = oracle.tensor_query(delta, r-h, 0), oracle.tensor_query(delta, r+h, 0)
    torch.testing.assert_close(slope, (plus-minus)/(2*h), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(curvature, (plus-2*cost+minus)/h**2, rtol=1e-4, atol=1e-6)
    assert cost.device.type == device
    with pytest.raises(ValueError, match="outside distilled"):
        oracle.tensor_query([[4., 0.]], 1.)
    with pytest.raises(ValueError, match="outside distilled"):
        oracle.tensor_query([[1., 0.]], 20.)


@pytest.mark.parametrize("reduction", ["mean", "mean_plus_max"])
def test_value_and_derivative_supervision_backpropagate(reduction):
    model = CostNetwork(3., 0.1, 10., width=8, layers=2).double()
    inputs = torch.randn(8, 2, dtype=torch.float64)
    loss, *_ = loss_on_grid(model, inputs, torch.ones(8, 1, dtype=torch.float64),
                            torch.zeros(8, 1, dtype=torch.float64), reduction)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_derivative_supervision_is_in_ratio_not_log_ratio():
    model = CostNetwork(3., 0.1, 10., width=8, layers=1).double()
    model.net = nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.net.weight.copy_(torch.tensor([[0., 1.]], dtype=torch.float64))
    z = torch.tensor([[0.2, -0.8], [-0.4, 0.7]], dtype=torch.float64)
    r = (z[:, 1:2]*model.input_scale[1]+model.input_center[1]).exp()
    loss, _, slope_loss = loss_on_grid(model, z, z[:, 1:2], 1/(r*model.input_scale[1]))
    assert loss.item() == 0
    assert slope_loss.item() == 0


def test_curvature_supervision_uses_physical_ratio_chain_rule():
    model = CostNetwork(3., .1, 10., width=8, layers=1).double()
    model.net = nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.net.weight.copy_(torch.tensor([[0., 1.]], dtype=torch.float64))
    z = torch.tensor([[.2, -.8], [-.4, .7]], dtype=torch.float64)
    r = (z[:, 1:2]*model.input_scale[1]+model.input_center[1]).exp()
    slope = 1/(r*model.input_scale[1])
    curvature = -1/(r.square()*model.input_scale[1])
    loss, _, _, curvature_loss = loss_on_grid(model, z, z[:, 1:2], slope, curvatures=curvature)
    assert loss < 1e-25 and curvature_loss < 1e-25
    loss, *_ = loss_on_grid(model, z, z[:, 1:2], slope, curvatures=curvature+.1)
    loss.backward()
    assert torch.isfinite(model.net.weight.grad).all() and model.net.weight.grad.abs().max() > 0


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_continuous_coupling_is_entirely_student_based(device, monkeypatch):
    # Closed-form network substitute makes solver correctness independent of
    # stochastic fitting, while the production student oracle and builder run.
    class AnalyticStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros((), dtype=torch.float64))
        def forward(self, d, ell):
            r = ell.exp()
            return d.square()/2 + 20*(1+r-2*r.sqrt()) + self.dummy*0
    path = PairPath(width=4, layers=1, parameterization="axial_distance").double().to(device)
    student = DistilledCostOracle(path, AnalyticStudent().to(device), 0.1, 10., 3.)
    monkeypatch.setattr(path, "forward", forbidden)
    monkeypatch.setattr("src.coupling_gpu.TorchHermiteCost.__init__", forbidden)
    report = check_deployment_curvature(student, dict(distance_grid_nodes=3, ratio_grid_nodes=5, validation_samples=9))
    assert report["passed"] and report["negative_or_zero_count"] == 0
    assert report["grid_shape"] == [9, 17]
    delta = np.array([[[0.1, 0.], [0.7, 0.]], [[-0.4, 0.], [0.2, 0.]]])
    table = NeuralPairCost(student, delta)
    assert table.check_curvature(17, 1e-5)["min_curvature"] > 0
    # Analytic inverse, including both boundaries, and conjugate derivative.
    lam = torch.tensor([[-50., 0.], [10., 30.]], dtype=torch.float64, device=device)
    r, star, rprime, residual = table.conjugate(lam)
    expected = torch.where(lam >= 20, 10., (20/(20-lam)).square()).clamp(.1, 10.)
    torch.testing.assert_close(r, expected, atol=1e-9, rtol=1e-9)
    cost = table(r)[0]
    torch.testing.assert_close(star, lam*r-cost)
    torch.testing.assert_close(rprime, torch.where((r > .1) & (r < 10), r**1.5/10, 0))
    assert residual.max() < 1e-9
    result = solve_dual_gpu([0.4, 0.6], [0.5, 0.7], table, epsilon=0.2,
                            marginal_tolerance=0.002, device=device)
    assert result.status == "converged"
    np.testing.assert_allclose(result.P.sum(1), [0.4, 0.6], rtol=0.004)
    np.testing.assert_allclose(result.Q.sum(0), [0.5, 0.7], rtol=1e-7)


@pytest.mark.parametrize("sampling", ["grid", "continuous"])
@pytest.mark.parametrize("curvature_weight", [0., 1.])
def test_fixed_steps_finish_even_when_accuracy_report_fails(tmp_path, monkeypatch, sampling, curvature_weight):
    import json
    from types import SimpleNamespace
    import src.cost_distillation as module
    from src.runtime import json_write
    path = PairPath(width=4, layers=1, parameterization="axial_distance").double()
    teacher_calls = []
    class Teacher:
        nodes, condition_chunk = 4, 4096
        def tensor_query(self, delta, r, derivatives):
            teacher_calls.append((delta.clone(), r.clone()))
            r = r.flatten()
            return (2+r.square(), 2*r, torch.full_like(r, 2))[:derivatives+1]
    monkeypatch.setattr(module, "configure", lambda config: torch.device("cpu"))
    monkeypatch.setattr(module, "init_run", lambda run, config: run)
    monkeypatch.setattr(module, "load_path", lambda *args: path)
    monkeypatch.setattr(module, "snapshots_for", lambda config: [
        SimpleNamespace(x=np.array([[0., 0.]])), SimpleNamespace(x=np.array([[1., 0.]]))])
    monkeypatch.setattr("src.solve_couplings.select_quadrature", lambda *args: Teacher())
    (tmp_path/"pair_path.pt").write_bytes(b"provenance")
    json_write(tmp_path/"status.json", {"pair": {"status": "completed"}})
    config = dict(seed=1, pair=dict(ratio_min=.1, ratio_max=10.), coupling=dict(condition_chunk=4096),
                  distillation=dict(distance_grid_nodes=3, ratio_grid_nodes=3, hidden_width=4,
                                    hidden_layers=1, batch_size=9, training_steps=2, learning_rate=.001,
                                    validation_samples=3, query_atol=1e-15, query_rtol=1e-15, sampling=sampling,
                                    curvature_weight=curvature_weight))
    module.train_distillation(config, tmp_path)
    validation = json.loads((tmp_path/"distillation_validation.json").read_text())
    assert validation["passed"] is False
    assert validation["training_complete"] and validation["completed_steps"] == 2
    checkpoint = torch.load(tmp_path/"cost_network.pt", weights_only=False)
    assert checkpoint["validated"] is False
    assert checkpoint["stopping_rule"] == "fixed_steps"
    grid = torch.load(tmp_path/"distillation_grid.pt", weights_only=False)
    assert (grid["curvature"] is not None) == bool(curvature_weight)
    assert ("second_derivative" in validation["supervised_targets"]) == bool(curvature_weight)
    if sampling == "continuous":
        # Fresh batches have nine distinct distances, unlike the 3x3 grid.
        fresh = [(d, r) for d, r in teacher_calls if len(d) == 9 and len(d[:, 0].unique()) == 9]
        assert len(fresh) == 2 and not torch.equal(fresh[0][0], fresh[1][0])
    else:
        # Exactly one label-grid call, one validation call and four benchmark
        # calls; the optimization loop never calls the teacher in grid mode.
        assert len(teacher_calls) == 6
    saved = (tmp_path/"cost_network.pt").read_bytes()
    history = (tmp_path/"distillation_history.jsonl").read_bytes()
    module.train_distillation(config, tmp_path)
    assert (tmp_path/"cost_network.pt").read_bytes() == saved
    assert (tmp_path/"distillation_history.jsonl").read_bytes() == history


def test_continuous_cost_reports_nonconvexity_without_modifying_weights():
    class NonconvexStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
        def forward(self, d, ell):
            r = ell.exp()
            return r.square()/2 + self.amplitude*torch.sin(5*r) + d.square()/2
    path = PairPath(width=4, layers=1, parameterization="axial_distance").double()
    model = NonconvexStudent()
    oracle = DistilledCostOracle(path, model, .1, 10., 3.)
    report = check_deployment_curvature(oracle, dict(distance_grid_nodes=3, ratio_grid_nodes=33, validation_samples=9))
    assert not report["passed"] and report["negative_or_zero_count"] > 0
    cost = NeuralPairCost(oracle, np.zeros((2, 2, 2)))
    before = model.amplitude.detach().clone()
    with pytest.warns(RuntimeWarning, match="monotonicity warning"):
        report = cost.check_curvature(129, 1e-5)
    assert report["min_curvature"] < 0
    assert cost.warning_records["domain_probe"]["occurrences"] == 1
    torch.testing.assert_close(before, model.amplitude, rtol=0, atol=0)


def test_nonpositive_newton_curvature_warns_and_bisects_to_valid_root():
    class QuarticStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros((), dtype=torch.float64))
        def forward(self, d, ell):
            r = ell.exp()
            return r**4/4-r.square()/2 + self.dummy*0
    path = PairPath(width=4, layers=1, parameterization="axial_distance").double()
    oracle = DistilledCostOracle(path, QuarticStudent(), .1, 10., 3.)
    cost = NeuralPairCost(oracle, np.zeros((1, 1, 2)))
    cost._warm_ratio = torch.tensor([[.2]], dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="continuous_ratio_inverse"):
        r, star, rprime, residual = cost.conjugate(torch.zeros((1, 1), dtype=torch.float64))
    torch.testing.assert_close(r, torch.ones_like(r), atol=1e-9, rtol=0)
    torch.testing.assert_close(star, torch.full_like(star, .25))
    torch.testing.assert_close(rprime, torch.full_like(rprime, .5))
    assert residual.max() <= 1e-9


def test_warning_policy_does_not_suppress_nonfinite_cost():
    from src.coupling import OracleError
    class BadStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros((), dtype=torch.float64))
        def forward(self, d, ell):
            return ell.exp()*float("nan") + self.dummy*0
    path = PairPath(width=4, layers=1, parameterization="axial_distance").double()
    oracle = DistilledCostOracle(path, BadStudent(), .1, 10., 3.)
    with pytest.raises(OracleError) as exc:
        NeuralPairCost(oracle, np.zeros((1, 1, 2)))
    assert exc.value.status == "cost_oracle_nonfinite"


def test_bisection_uses_only_cost_and_first_derivative(monkeypatch):
    class QuadraticStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros((), dtype=torch.float64))
        def forward(self, d, ell):
            return ell.exp().square()/2 + d.square()/2 + self.dummy*0
    path = PairPath(width=4, layers=1, parameterization="axial_distance").double()
    oracle = DistilledCostOracle(path, QuadraticStudent(), .1, 10., 3.)
    original_grad = torch.autograd.grad
    def first_order_only(*args, **kwargs):
        assert not kwargs.get("create_graph", False), "second derivative graph requested"
        return original_grad(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "grad", first_order_only)
    cost = NeuralPairCost(oracle, np.zeros((1, 2, 2)), root_method="bisection")
    r, star, rprime, residual = cost.conjugate(torch.tensor([[.8, 1.3]], dtype=torch.float64))
    torch.testing.assert_close(r, torch.tensor([[.8, 1.3]], dtype=torch.float64), atol=1e-9, rtol=0)
    assert rprime is None and residual.max() <= 1e-9
    result = solve_dual_gpu([1.], [.4, .6], cost, epsilon=.2, device="cpu",
                            max_outer_iterations=2, root_method="bisection")
    assert result.history[-1]["column_relative_l1"] < 1e-8
    assert result.history[-1]["inverse_derivative_residual"] <= 1e-9
    assert result.history[-1]["root_method"] == "bisection"
    # Batched device-to-host checks must retain both public-query guards.
    from src.coupling import OracleError
    for query in (cost, cost.value):
        with pytest.raises(ValueError, match="outside distilled"):
            query(torch.tensor([[-1., 1.]], dtype=torch.float64))
        with pytest.raises(OracleError) as exc:
            query(torch.tensor([[float("nan"), 1.]], dtype=torch.float64))
        assert exc.value.status == "cost_oracle_nonfinite"

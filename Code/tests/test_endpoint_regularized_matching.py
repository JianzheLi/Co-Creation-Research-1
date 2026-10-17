import math
import numpy as np
import torch
from torch import nn

from src.matching import Feedback, make_targets, matching_loss, matching_loss_weights
from src.pair_path import PairPath
from src.sample import rollout
from src.wfr_heuristic import WFRBridgePath, WFRBrownianBridgePath


def _path_state(sigma):
    torch.manual_seed(23)
    path = PairPath(dimension=3, sigma=sigma, alpha=2.0, width=12, layers=2,
                    parameterization="axial_distance").double()
    t = torch.tensor([[0.2], [0.55], [0.83]], dtype=torch.float64)
    delta = torch.tensor([[0.8, -0.3, 0.2], [-0.4, 0.9, 0.1], [0.5, 0.2, -0.7]], dtype=torch.float64)
    log_r = torch.tensor([[-0.4], [0.2], [0.7]], dtype=torch.float64)
    x0 = torch.tensor([[0.1, 0.3, 0.2], [0.7, -0.1, 0.5], [-0.2, 0.1, 0.4]], dtype=torch.float64)
    return path, path(t, delta, log_r, create_graph=False, x0=x0)


def test_velocity_teacher_is_split_into_finite_and_scaled_singular_parts():
    path, state = _path_state(0.5)
    noise = torch.tensor([[0.3, -0.5, 1.2], [-0.8, 0.1, 0.4], [0.6, 0.2, -0.9]], dtype=torch.float64)
    old = make_targets(state, noise, path.sigma, path.alpha)
    new = make_targets(state, noise, path.sigma, path.alpha,
                       field_parameterization="endpoint_regularized")
    sqrt_tau = state["tau"].sqrt()

    torch.testing.assert_close(new["target_v_regular"]
                               + new["target_v_singular_scaled"] / sqrt_tau,
                               old["target_v_scaled"] / sqrt_tau)
    torch.testing.assert_close(new["target_v_scaled"], old["target_v_scaled"])
    torch.testing.assert_close(new["target_s_scaled"], old["target_s_scaled"])
    assert torch.equal(new["sqrt_wf"], torch.ones_like(new["sqrt_wf"]))
    assert torch.equal(new["sqrt_ws"], torch.ones_like(new["sqrt_ws"]))


def test_endpoint_regularized_loss_supervises_both_velocity_heads_without_tau_weights():
    _, state = _path_state(0.5)
    noise = torch.randn((3, 3), dtype=torch.float64)
    targets = make_targets(state, noise, 0.5, 2.0,
                           field_parameterization="endpoint_regularized")
    targets = {key: value.float() for key, value in targets.items()}
    prediction = tuple(torch.randn_like(targets[key]) for key in (
        "target_v_regular", "target_v_singular_scaled", "target_g", "target_s_scaled"))
    factor = torch.tensor([1.0, 0.5, 2.0])
    loss, parts = matching_loss(prediction, targets, factor)
    mass_factor = factor * targets["mass"].float().squeeze(-1)
    flow = ((prediction[0] - targets["target_v_regular"].float()).square().sum(-1)
            + (prediction[1] - targets["target_v_singular_scaled"].float()).square().sum(-1))
    growth = (prediction[2] - targets["target_g"].float()).square().squeeze(-1)
    score = (prediction[3] - targets["target_s_scaled"].float()).square().sum(-1)
    torch.testing.assert_close(loss, (mass_factor.squeeze(-1) * (flow + growth + score)).mean())
    torch.testing.assert_close(parts["flow"], (mass_factor.squeeze(-1) * flow).mean())


def test_sigma_balanced_weights_normalize_singular_flow_and_scale_growth_by_action_coefficient():
    config = {"model_sigma": 0.5, "growth_penalty": 1225.0,
              "matching": {"field_parameterization": "endpoint_regularized",
                           "loss_weighting": "endpoint_sigma_balanced"}}
    weights = matching_loss_weights(config)
    assert weights == {"flow_regular": 1.0, "flow_singular": 4.0,
                       "flow": 1.0, "growth": 1225.0, "score": 1.0}

    _, state = _path_state(0.5)
    noise = torch.randn((3, 3), dtype=torch.float64)
    targets = make_targets(state, noise, 0.5, 2.0,
                           field_parameterization="endpoint_regularized")
    prediction = tuple(torch.randn_like(targets[key]) for key in (
        "target_v_regular", "target_v_singular_scaled", "target_g", "target_s_scaled"))
    factor = torch.tensor([1.0, 0.5, 2.0])
    loss, parts = matching_loss(prediction, targets, factor, weights)
    mass_factor = factor * targets["mass"].float().squeeze(-1)
    regular = ((prediction[0] - targets["target_v_regular"]).square().sum(-1)
               * mass_factor).mean()
    singular = ((prediction[1] - targets["target_v_singular_scaled"]).square().sum(-1)
                * mass_factor).mean()
    growth = ((prediction[2] - targets["target_g"]).square().squeeze(-1)
              * mass_factor).mean()
    score = ((prediction[3] - targets["target_s_scaled"]).square().sum(-1)
             * mass_factor).mean()
    torch.testing.assert_close(parts["flow_regular"], regular)
    torch.testing.assert_close(parts["flow_singular"], 4.0 * singular)
    torch.testing.assert_close(parts["growth"], 1225.0 * growth)
    torch.testing.assert_close(parts["score"], score)
    torch.testing.assert_close(loss, regular + 4.0 * singular + 1225.0 * growth + score)


def test_growth_loss_multiplier_scales_only_growth_weight_and_changes_matching_version():
    base = {"model_sigma": 0.5, "growth_penalty": 1800.0,
            "matching": {"field_parameterization": "endpoint_regularized",
                         "absorb_sigma_in_singular_velocity": True,
                         "loss_weighting": "endpoint_explicit_sigma"}}
    doubled = {**base, "matching": {**base["matching"], "growth_loss_multiplier": 2.0}}
    base_weights = matching_loss_weights(base)
    doubled_weights = matching_loss_weights(doubled)
    assert base_weights == {"flow_regular": 1.0, "flow_singular": 1.0,
                            "flow": 1.0, "growth": 1800.0, "score": 1.0}
    assert doubled_weights == {**base_weights, "growth": 3600.0}

    from src.matching import matching_version
    assert matching_version(base) == 8
    assert matching_version(doubled) == 9


def test_absorbing_sigma_into_singular_velocity_exactly_replaces_inverse_sigma_squared_loss():
    for sigma in (0.1, 0.25, 0.5, 1.0):
        path, state = _path_state(sigma)
        noise = torch.randn((3, 3), dtype=torch.float64)
        old_targets = make_targets(state, noise, sigma, 2.0,
                                  field_parameterization="endpoint_regularized")
        new_targets = make_targets(state, noise, sigma, 2.0,
                                  field_parameterization="endpoint_regularized",
                                  absorb_sigma_in_singular_velocity=True)
        old_targets = {key: value.float() for key, value in old_targets.items()}
        new_targets = {key: value.float() for key, value in new_targets.items()}
        torch.testing.assert_close(
            sigma * new_targets["target_v_singular_scaled"],
            old_targets["target_v_singular_scaled"], rtol=2e-6, atol=1e-7)

        old_prediction = tuple(torch.randn_like(old_targets[key]) for key in (
            "target_v_regular", "target_v_singular_scaled", "target_g", "target_s_scaled"))
        new_prediction = (old_prediction[0], old_prediction[1] / sigma,
                          old_prediction[2], old_prediction[3])
        factor = torch.tensor([1.0, 0.5, 2.0])
        old_config = {"model_sigma": sigma, "growth_penalty": 2.0, "matching": {
            "field_parameterization": "endpoint_regularized",
            "loss_weighting": "endpoint_sigma_balanced"}}
        new_config = {"model_sigma": sigma, "growth_penalty": 2.0, "matching": {
            "field_parameterization": "endpoint_regularized",
            "absorb_sigma_in_singular_velocity": True,
            "loss_weighting": "endpoint_explicit_sigma"}}
        old_loss, old_parts = matching_loss(
            old_prediction, old_targets, factor, matching_loss_weights(old_config))
        new_loss, new_parts = matching_loss(
            new_prediction, new_targets, factor, matching_loss_weights(new_config))
        torch.testing.assert_close(new_loss, old_loss, rtol=2e-6, atol=1e-6)
        torch.testing.assert_close(new_parts["flow_singular"], old_parts["flow_singular"],
                                   rtol=2e-6, atol=1e-6)


def test_endpoint_regularized_feedback_uses_shared_two_head_flow_and_decodes_exactly():
    model = Feedback(dimension=3, width=16, layers=2, network_variant="gated_residual",
                     field_parameterization="endpoint_regularized", sigma=0.5,
                     endpoint_guard=0.01)
    t = torch.tensor([[0.2], [1.55], [2.0]])
    x = torch.randn((3, 3))
    encoded = model.matching_fields(t, x)
    assert len(encoded) == 4
    assert model.flow.regular_head is not model.flow.singular_head
    local = t - torch.floor(t)
    sqrt_tau = (local * (1.0 - local)).clamp_min(0.01 * 0.99).sqrt()
    physical = model(t, x)
    torch.testing.assert_close(physical[0], encoded[0] + encoded[1] / sqrt_tau)
    torch.testing.assert_close(physical[1], encoded[2])
    torch.testing.assert_close(physical[2], encoded[3] / (0.5 * sqrt_tau))


def test_sigma_absorbed_singular_velocity_decoder_preserves_physical_velocity():
    sigma = 0.25
    old_model = Feedback(dimension=3, width=16, layers=2,
                         field_parameterization="endpoint_regularized", sigma=sigma)
    new_model = Feedback(dimension=3, width=16, layers=2,
                         field_parameterization="endpoint_regularized", sigma=sigma,
                         absorb_sigma_in_singular_velocity=True)
    t = torch.tensor([[0.2], [0.55], [0.83]])
    regular = torch.randn((3, 3))
    singular_old = sigma * torch.randn((3, 3))
    singular_new = singular_old / sigma
    growth, score = torch.randn((3, 1)), torch.randn((3, 3))
    old_fields = old_model._decode_fields(t, (regular, singular_old, growth, score))
    new_fields = new_model._decode_fields(t, (regular, singular_new, growth, score))
    for old, new in zip(old_fields, new_fields):
        torch.testing.assert_close(old, new)


def test_endpoint_regularized_feedback_can_use_four_independent_networks():
    model = Feedback(dimension=3, width=16, layers=2, network_variant="plain",
                     field_parameterization="endpoint_regularized", independent_velocity_networks=True,
                     sigma=0.5)
    t = torch.tensor([[0.2], [1.55], [2.0]])
    x = torch.randn((3, 3))
    fields = model.matching_fields(t, x)
    assert model.flow is None
    assert model.flow_regular is not model.flow_singular
    assert model.flow_regular is not model.growth
    assert model.flow_regular is not model.score
    assert model.flow_singular is not model.growth
    assert model.flow_singular is not model.score
    assert all(value.shape == x.shape for value in (fields[0], fields[1], fields[3]))
    assert fields[2].shape == (len(x), 1)


def test_regular_velocity_and_normalized_score_supervision_remain_finite_as_sigma_shrinks():
    path_big, state_big = _path_state(0.5)
    path_small = PairPath(dimension=3, sigma=1e-4, alpha=2.0, width=12, layers=2,
                          parameterization="axial_distance").double()
    path_small.load_state_dict(path_big.state_dict())
    t = state_big["t"]
    state_small = path_small(t, state_big["delta"], state_big["log_r"],
                             create_graph=False, x0=state_big["xi"] - state_big["mean_displacement"])
    noise = torch.tensor([[0.3, -0.5, 1.2], [-0.8, 0.1, 0.4], [0.6, 0.2, -0.9]], dtype=torch.float64)
    big = make_targets(state_big, noise, 0.5, 2.0,
                       field_parameterization="endpoint_regularized")
    small = make_targets(state_small, noise, 1e-4, 2.0,
                         field_parameterization="endpoint_regularized")
    assert torch.isfinite(small["target_v_regular"]).all()
    assert torch.isfinite(small["target_v_singular_scaled"]).all()
    assert torch.isfinite(small["target_s_scaled"]).all()
    torch.testing.assert_close(small["target_v_regular"], big["target_v_regular"], rtol=5e-2, atol=3e-2)
    torch.testing.assert_close(small["target_s_scaled"], big["target_s_scaled"], rtol=2e-3, atol=2e-3)
    assert small["target_v_regular"].abs().mean() > 0.1
    assert small["target_s_scaled"].abs().mean() > 0.1


def test_wfr_bridge_teachers_use_the_same_two_head_velocity_decomposition():
    t = torch.tensor([[0.22], [0.61], [0.84]], dtype=torch.float64)
    x0 = torch.tensor([[0.1, -0.2], [0.4, 0.3], [-0.5, 0.2]], dtype=torch.float64)
    delta = torch.tensor([[0.7, 0.2], [-0.3, 0.8], [0.1, -0.6]], dtype=torch.float64)
    log_r = torch.tensor([[0.1], [-0.4], [0.5]], dtype=torch.float64)
    noise = torch.tensor([[0.3, -0.5], [-0.8, 0.1], [0.6, 0.2]], dtype=torch.float64)
    for path in (WFRBridgePath(alpha=2.0, sigma=0.5),
                 WFRBrownianBridgePath(alpha=2.0, sigma=0.5)):
        targets = path.conditional_samples(t, x0, delta, log_r, noise)
        sqrt_tau = t.mul(1 - t).sqrt()
        torch.testing.assert_close(
            targets["target_v_regular"] + targets["target_v_singular_scaled"] / sqrt_tau,
            targets["target_v_scaled"] / sqrt_tau)
        torch.testing.assert_close(targets["target_s_scaled"], -noise)


def test_endpoint_regularized_ode_integrates_the_integrable_endpoint_singularity():
    class ConstantEncodedFeedback(nn.Module):
        field_parameterization = "endpoint_regularized"

        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def matching_fields(self, t, x):
            regular = x.new_tensor([0.7, -0.3]).expand_as(x)
            singular_scaled = x.new_tensor([0.2, -0.1]).expand_as(x)
            growth = torch.full_like(t, math.log(2.0))
            score_scaled = torch.zeros_like(x)
            return regular, singular_scaled, growth, score_scaled

    initial = np.zeros((3, 2), dtype=np.float32)
    result = rollout(ConstantEncodedFeedback(), initial, n_particles=len(initial),
                     steps_per_unit=20, initial_x=initial, stochastic=False,
                     trace_indices=[0])
    expected = 4 * (np.array([0.7, -0.3]) + math.pi * np.array([0.2, -0.1]))
    np.testing.assert_allclose(result["positions"][-1], np.tile(expected, (len(initial), 1)), atol=3e-6)
    np.testing.assert_allclose(np.exp(result["log_weights"][-1]).sum(), 16.0, rtol=1e-6)
    trace_times = result["trace_times"]
    assert len(trace_times) == 4 * 20 + 1
    assert np.all(np.diff(trace_times) > 0)
    np.testing.assert_allclose(trace_times[[0, 20, 40, 60, 80]], [0, 1, 2, 3, 4], atol=1e-12)


def test_sigma_absorbed_singular_velocity_is_decoded_in_ode_and_sde_rollouts():
    class ConstantAbsorbedFeedback(nn.Module):
        field_parameterization = "endpoint_regularized"
        absorb_sigma_in_singular_velocity = True
        sigma = 0.25

        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def matching_fields(self, t, x):
            regular = torch.zeros_like(x)
            singular_scaled = x.new_tensor([0.2, -0.1]).expand_as(x)
            growth = torch.zeros_like(t)
            score_scaled = x.new_tensor([0.4, 0.3]).expand_as(x)
            return regular, singular_scaled, growth, score_scaled

    steps = 20
    initial = np.zeros((1, 2), dtype=np.float32)
    model = ConstantAbsorbedFeedback()
    ode = rollout(model, initial, n_particles=1, steps_per_unit=steps,
                  initial_x=initial, stochastic=False)
    expected_ode = 4 * math.pi * model.sigma * np.array([0.2, -0.1])
    np.testing.assert_allclose(ode["positions"][-1, 0], expected_ode, atol=3e-6)

    zero_increments = np.zeros((4 * steps, 1, 2), dtype=np.float32)
    sde = rollout(model, initial, n_particles=1, steps_per_unit=steps,
                  sigma=model.sigma, initial_x=initial, stochastic=True,
                  normal_increments=zero_increments)
    expected_sde = 4 * math.pi * model.sigma * (np.array([0.2, -0.1])
                                                 + 0.5 * np.array([0.4, 0.3]))
    np.testing.assert_allclose(sde["positions"][-1, 0], expected_sde, atol=3e-6)


def test_endpoint_regularized_sde_preserves_physical_brownian_variance_under_time_change():
    class ZeroEncodedFeedback(nn.Module):
        field_parameterization = "endpoint_regularized"

        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def matching_fields(self, t, x):
            return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(t), torch.zeros_like(x)

    steps = 20
    dt = 1 / steps
    # A constant standardized Brownian increment of +1 at every solver step.
    uniform_time_increments = np.full((steps * 4, 1, 2), math.sqrt(dt), dtype=np.float32)
    initial = np.zeros((1, 2), dtype=np.float32)
    result = rollout(ZeroEncodedFeedback(), initial, n_particles=1,
                     steps_per_unit=steps, sigma=0.5, initial_x=initial,
                     normal_increments=uniform_time_increments, trace_indices=[0])

    s_nodes = np.arange(steps + 1) / steps
    t_nodes = np.sin(0.5 * math.pi * s_nodes) ** 2
    expected_per_interval = np.sqrt(np.diff(t_nodes)).sum()
    np.testing.assert_allclose(result["positions"][-1],
                               np.full((1, 2), 4 * 0.5 * expected_per_interval),
                               rtol=1e-6, atol=1e-6)
    assert len(result["trace_times"]) == steps * 4 + 1
    np.testing.assert_allclose(result["trace_times"][[0, steps, 2 * steps, 3 * steps, 4 * steps]],
                               [0, 1, 2, 3, 4], atol=1e-12)

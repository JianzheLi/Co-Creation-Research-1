import math

import pytest
import torch
from torch import nn

from src.matching import (Feedback, feedback_architecture,
                          FeedbackMLP, validate_feedback_architecture)


def test_tanh_hidden_layers_keep_feedback_bounded_for_extreme_inputs():
    torch.manual_seed(7)
    model = Feedback(dimension=2, width=16, layers=3, activation="tanh")
    probes = torch.tensor([[0.0, 0.0], [1.0, -2.0], [1e6, -1e6], [-1e6, 1e6]])
    times = torch.tensor([[0.0], [0.5], [1e6], [-1e6]])

    outputs = model(times, probes)
    for subnetwork, output in zip((model.flow, model.growth, model.score), outputs):
        final = subnetwork.net[-1]
        bound = final.bias.abs() + final.weight.abs().sum(dim=1)
        assert torch.isfinite(output).all()
        assert (output.abs() <= bound + 1e-6).all()


def test_feedback_architecture_is_saved_and_checked():
    config = {"matching": {"feedback_activation": "tanh"}}
    architecture = feedback_architecture(config)
    assert architecture == {
        "hidden_activation": "tanh",
        "last_hidden_activation": "tanh",
        "output_activation": "linear",
        "network_variant": "plain",
        "growth_network_variant": "plain",
        "score_network_variant": "plain",
        "field_parameterization": "raw",
        "independent_velocity_networks": False,
        "absorb_sigma_in_singular_velocity": False,
    }
    validate_feedback_architecture({"feedback_architecture": architecture}, config)
    with pytest.raises(RuntimeError, match="different network architecture"):
        validate_feedback_architecture({
            "feedback_architecture": {**architecture, "network_variant": "residual"}
        }, config)


def test_feedback_rejects_unknown_activation():
    with pytest.raises(ValueError, match="unsupported feedback activation"):
        Feedback(activation="gelu")


def test_leaky_relu_hidden_layers_keep_negative_gradients_and_linear_output():
    model = Feedback(dimension=2, width=16, layers=3, activation="leaky_relu")
    hidden_activations = [layer for layer in model.growth.net if isinstance(layer, nn.LeakyReLU)]
    assert len(hidden_activations) == 3
    assert all(layer.negative_slope == 0.01 for layer in hidden_activations)
    assert isinstance(model.growth.net[-1], nn.Linear)

    probe = torch.tensor([-2.0], requires_grad=True)
    hidden_activations[0](probe).sum().backward()
    assert probe.grad.item() == pytest.approx(0.01)


def test_leaky_relu_architecture_records_fixed_slope():
    config = {"matching": {"feedback_activation": "leaky_relu"}}
    architecture = feedback_architecture(config)
    assert architecture["hidden_activation"] == "leaky_relu"
    assert architecture["last_hidden_activation"] == "leaky_relu"
    assert architecture["leaky_relu_negative_slope"] == pytest.approx(0.01)
    validate_feedback_architecture({"feedback_architecture": architecture}, config)


@pytest.mark.parametrize("variant,width,layers", [
    ("plain", 400, 6),
    ("plain_output_norm", 400, 6),
    ("plain", 339, 8),
    ("residual", 400, 6),
    ("residual_scaled", 400, 6),
    ("residual_branch_scaled", 400, 6),
    ("prenorm_residual", 400, 6),
    ("prenorm_residual_unbounded", 400, 6),
    ("prenorm_residual_linear_skip", 400, 6),
    ("prenorm_bounded_branch_residual", 400, 6),
    ("gated", 300, 6),
    ("gated_residual", 300, 6),
    ("fourier", 420, 6),
])
def test_matching_network_variants_have_independent_equal_shape_outputs(variant, width, layers):
    torch.manual_seed(13)
    model = Feedback(dimension=100, width=width, layers=layers, network_variant=variant, time_scale=4.0)
    assert model.flow is not model.growth and model.flow is not model.score and model.growth is not model.score
    assert model.flow.net[0].weight.data_ptr() != model.score.net[0].weight.data_ptr()
    times = torch.rand(3, 1)
    points = torch.randn(3, 100)
    values = model(times, points)
    assert [tuple(value.shape) for value in values] == [(3, 100), (3, 1), (3, 100)]
    assert all(torch.isfinite(value).all() for value in values)


def test_matching_variant_parameter_counts_are_in_expected_range():
    models = [
        Feedback(100, 400, 6),
        Feedback(100, 400, 6, network_variant="plain_output_norm"),
        Feedback(100, 339, 8),
        Feedback(100, 400, 6, network_variant="residual"),
        Feedback(100, 400, 6, network_variant="residual_scaled"),
        Feedback(100, 400, 6, network_variant="residual_branch_scaled"),
        Feedback(100, 400, 6, network_variant="prenorm_residual"),
        Feedback(100, 400, 6, network_variant="prenorm_residual_unbounded"),
        Feedback(100, 400, 6, network_variant="prenorm_residual_linear_skip"),
        Feedback(100, 400, 6, network_variant="prenorm_bounded_branch_residual"),
        Feedback(100, 300, 6, network_variant="gated"),
        Feedback(100, 300, 6, network_variant="gated_residual"),
        Feedback(100, 420, 6, network_variant="fourier"),
    ]
    counts = [sum(p.numel() for p in model.parameters()) for model in models]
    baseline = counts[0]
    assert counts == [2_609_001, 2_609_001, 2_592_534, 2_609_001, 2_609_001, 2_609_001, 2_609_001, 2_609_001,
                      2_629_302, 2_609_001,
                      2_861_301, 2_861_301, 2_875_521]
    assert all(abs(count / baseline - 1) < 0.11 for count in counts[1:])


def test_depth_scaled_residual_preserves_identity_skip_when_learned_branch_is_zero():
    model = FeedbackMLP(2, 2, width=3, layers=3, activation="leaky_relu",
                        network_variant="residual_branch_scaled")
    with torch.no_grad():
        model.net[0].weight.copy_(torch.eye(3))
        model.net[0].bias.zero_()
        for layer_index in (2, 4):
            model.net[layer_index].weight.zero_()
            model.net[layer_index].bias.zero_()
        model.net[-1].weight.zero_()
        model.net[-1].weight[0, 1] = 1.0
        model.net[-1].weight[1, 2] = 1.0
        model.net[-1].bias.zero_()
    t = torch.tensor([[0.5]])
    x = torch.tensor([[2.0, 3.0]])
    output = model(t, x)
    torch.testing.assert_close(output, x)
    assert model.residual_branch_scale == pytest.approx(1 / (2 ** 0.5))


def test_prenorm_residual_readout_is_bounded_for_large_finite_inputs():
    torch.manual_seed(22)
    model = FeedbackMLP(2, 2, width=16, layers=3, activation="leaky_relu",
                        network_variant="prenorm_residual")
    time = torch.tensor([[0.5], [0.8]])
    points = torch.tensor([[1e8, -1e8], [-5e7, 7e7]])
    output = model(time, points)
    final = model.net[-1]
    bound = torch.linalg.matrix_norm(final.weight, ord=2) * (16 ** 0.5) + final.bias.norm()
    assert torch.isfinite(output).all()
    assert (output.norm(dim=-1) <= bound + 1e-3).all()


def test_plain_output_norm_bounds_every_endpoint_regularized_field_without_new_parameters():
    torch.manual_seed(31)
    model = Feedback(dimension=2, width=16, layers=3, activation="leaky_relu",
                     network_variant="plain_output_norm", field_parameterization="endpoint_regularized",
                     independent_velocity_networks=True)
    baseline = Feedback(dimension=2, width=16, layers=3, activation="leaky_relu",
                        network_variant="plain", field_parameterization="endpoint_regularized",
                        independent_velocity_networks=True)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters())
    t = torch.tensor([[0.2], [2.7]])
    x = torch.tensor([[1e8, -1e8], [-5e7, 7e7]])
    outputs = model.matching_fields(t, x)
    subnetworks = (model.flow_regular, model.flow_singular, model.growth, model.score)
    for subnetwork, output in zip(subnetworks, outputs):
        head = subnetwork.net[-1]
        bound = torch.linalg.matrix_norm(head.weight, ord=2) * math.sqrt(16) + head.bias.norm()
        assert torch.isfinite(output).all()
        assert (output.norm(dim=-1) <= bound + 1e-3).all()


def test_plain_output_affine_norm_learns_scale_and_shift_for_all_four_fields():
    torch.manual_seed(32)
    kwargs = dict(dimension=2, width=16, layers=3, activation="leaky_relu",
                  field_parameterization="endpoint_regularized",
                  independent_velocity_networks=True)
    model = Feedback(**kwargs, network_variant="plain_output_affine_norm",
                     growth_network_variant="plain_output_affine_norm",
                     score_network_variant="plain_output_affine_norm")
    non_affine = Feedback(**kwargs, network_variant="plain_output_norm",
                          growth_network_variant="plain_output_norm",
                          score_network_variant="plain_output_norm")

    fields = (model.flow_regular, model.flow_singular, model.growth, model.score)
    assert all(isinstance(field.output_norm, nn.LayerNorm) for field in fields)
    assert all(field.output_norm.elementwise_affine for field in fields)
    assert sum(p.numel() for p in model.parameters()) - sum(
        p.numel() for p in non_affine.parameters()) == 8 * kwargs["width"]

    t = torch.tensor([[0.2], [2.7]])
    x = torch.tensor([[0.25, -0.75], [-1.5, 0.5]])
    sum(value.square().mean() for value in model.matching_fields(t, x)).backward()
    assert all(field.output_norm.weight.grad is not None for field in fields)
    assert all(field.output_norm.bias.grad is not None for field in fields)


def test_affine_output_norm_architecture_is_recorded_in_checkpoint_metadata():
    config = {"matching": {"feedback_activation": "leaky_relu",
                           "network_variant": "plain_output_affine_norm",
                           "growth_network_variant": "plain_output_affine_norm",
                           "score_network_variant": "plain_output_affine_norm"}}
    architecture = feedback_architecture(config)
    assert architecture["network_variant"] == "plain_output_affine_norm"
    assert architecture["growth_network_variant"] == "plain_output_affine_norm"
    assert architecture["score_network_variant"] == "plain_output_affine_norm"
    validate_feedback_architecture({"feedback_architecture": architecture}, config)


def test_prenorm_residual_unbounded_readout_keeps_pre_norm_branches_without_output_cap():
    model = FeedbackMLP(2, 2, width=16, layers=3, activation="leaky_relu",
                        network_variant="prenorm_residual_unbounded")
    assert isinstance(model.output_norm, nn.Identity)
    assert len(model.residual_norms) == 2
    assert model.residual_branch_scale == pytest.approx(1 / (2 ** 0.5))


def test_prenorm_residual_linear_skip_preserves_a_direct_linear_input_path():
    model = FeedbackMLP(2, 2, width=4, layers=2, activation="leaky_relu",
                        network_variant="prenorm_residual_linear_skip")
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.zero_()
        model.input_skip.weight.copy_(torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    t = torch.tensor([[0.5]])
    x = torch.tensor([[2.0, 3.0]])
    torch.testing.assert_close(model(t, x), x)


def test_bounded_residual_branch_preserves_identity_and_caps_each_learned_update():
    model = FeedbackMLP(2, 2, width=4, layers=3, activation="leaky_relu",
                        network_variant="prenorm_bounded_branch_residual")
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].bias.zero_()
        model.net[0].weight[1, 1] = 1.0
        model.net[0].weight[2, 2] = 1.0
        for index in (2, 4):
            model.net[index].weight.zero_()
            model.net[index].bias.zero_()
        model.net[-1].weight.zero_()
        model.net[-1].weight[0, 1] = 1.0
        model.net[-1].weight[1, 2] = 1.0
        model.net[-1].bias.zero_()
    x = torch.tensor([[2.0, 3.0]])
    torch.testing.assert_close(model(torch.tensor([[0.5]]), x), x)
    assert len(model.residual_output_norms) == 2
    assert all(not list(norm.parameters()) for norm in model.residual_output_norms)


def test_score_output_norm_can_be_selected_without_changing_other_fields():
    config = {"matching": {"feedback_activation": "leaky_relu",
                           "network_variant": "plain",
                           "score_network_variant": "plain_output_norm"}}
    architecture = feedback_architecture(config)
    assert architecture["network_variant"] == "plain"
    assert architecture["score_network_variant"] == "plain_output_norm"
    validate_feedback_architecture({"feedback_architecture": architecture}, config)
    model = Feedback(dimension=2, width=16, layers=3, activation="leaky_relu",
                     score_network_variant="plain_output_norm")
    assert isinstance(model.flow.output_norm, nn.Identity)
    assert isinstance(model.growth.output_norm, nn.Identity)
    assert isinstance(model.score.output_norm, nn.LayerNorm)


def test_growth_architecture_can_remain_plain_while_velocity_and_score_use_prenorm():
    config = {"matching": {"feedback_activation": "leaky_relu",
                           "network_variant": "prenorm_residual",
                           "growth_network_variant": "plain"}}
    architecture = feedback_architecture(config)
    assert architecture["network_variant"] == "prenorm_residual"
    assert architecture["growth_network_variant"] == "plain"
    assert architecture["score_network_variant"] == "prenorm_residual"
    validate_feedback_architecture({"feedback_architecture": architecture}, config)
    model = Feedback(dimension=2, width=16, layers=3, activation="leaky_relu",
                     network_variant="prenorm_residual", growth_network_variant="plain")
    assert isinstance(model.flow.output_norm, nn.LayerNorm)
    assert isinstance(model.growth.output_norm, nn.Identity)
    assert isinstance(model.score.output_norm, nn.LayerNorm)



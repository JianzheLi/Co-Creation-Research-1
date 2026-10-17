from pathlib import Path
import json
import math
import time

import numpy as np
import torch
from torch import nn
from .pair_path import spectral_apply
from .runtime import configure, file_hash, init_run, load_path, log_json, snapshots_for, update_stage


LEAKY_RELU_NEGATIVE_SLOPE = 0.01
FEEDBACK_ACTIVATIONS = {"silu", "tanh", "leaky_relu"}


def _feedback_output_norm(width, network_variant):
    if network_variant == "plain_output_affine_norm":
        return nn.LayerNorm(width, elementwise_affine=True)
    if network_variant in {"prenorm_residual", "plain_output_norm"}:
        return nn.LayerNorm(width, elementwise_affine=False)
    return nn.Identity()


def _feedback_activation(name):
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=LEAKY_RELU_NEGATIVE_SLOPE)
    raise ValueError(f"unsupported feedback activation: {name}")


class FeedbackMLP(nn.Module):
    def __init__(self, dimension, n_output, width=400, layers=6, activation="silu",
                 last_hidden_activation=None, network_variant="plain", time_scale=1.0):
        super().__init__()
        if activation not in FEEDBACK_ACTIVATIONS:
            raise ValueError(f"unsupported feedback activation: {activation}")
        if network_variant not in {"plain", "residual", "residual_scaled", "residual_branch_scaled",
                                   "prenorm_residual", "prenorm_residual_unbounded",
                                   "prenorm_residual_linear_skip",
                                   "prenorm_bounded_branch_residual",
                                   "plain_output_norm",
                                   "plain_output_affine_norm",
                                   "gated", "gated_residual", "fourier"}:
            raise ValueError(f"unsupported feedback network variant: {network_variant}")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        last_hidden_activation = last_hidden_activation or activation
        if last_hidden_activation not in FEEDBACK_ACTIVATIONS:
            raise ValueError(f"unsupported last hidden activation: {last_hidden_activation}")
        self.network_variant = network_variant
        self.time_scale = float(time_scale)
        # The legacy residual_scaled variant scales its identity skip. The
        # residual_branch_scaled variant instead preserves the identity path
        # and scales only each learned residual update. Its factor is fixed by
        # depth (no additional tuning parameter).
        self.residual_scale = 1.0 / max(layers - 1, 1)
        self.residual_branch_scale = 1.0 / math.sqrt(max(layers - 1, 1))
        self.residual_norms = nn.ModuleList(
            [nn.LayerNorm(width, elementwise_affine=False) for _ in range(max(layers - 1, 0))]
            if network_variant in {"prenorm_residual", "prenorm_residual_unbounded",
                                   "prenorm_residual_linear_skip", "prenorm_bounded_branch_residual"} else [])
        self.residual_output_norms = nn.ModuleList(
            [nn.LayerNorm(width, elementwise_affine=False) for _ in range(max(layers - 1, 0))]
            if network_variant == "prenorm_bounded_branch_residual" else [])
        self.output_norm = _feedback_output_norm(width, network_variant)
        input_width = dimension + 1 + (8 if network_variant == "fourier" else 0)
        self.input_skip = (nn.Linear(input_width, n_output, bias=False)
                           if network_variant == "prenorm_residual_linear_skip" else None)
        modules = []
        for k in range(layers):
            hidden_activation = last_hidden_activation if k == layers - 1 else activation
            modules += [nn.Linear(input_width if k == 0 else width, width), _feedback_activation(hidden_activation)]
        modules.append(nn.Linear(width, n_output))
        self.net = nn.Sequential(*modules)
        self.gates = nn.ModuleList([nn.Linear(width, width) for _ in range(max(layers - 1, 0))]
                                   if network_variant in {"gated", "gated_residual"} else [])

    def forward(self, global_t, x):
        if self.network_variant == "fourier":
            frequencies = global_t.new_tensor((1.0, 2.0, 4.0, 8.0))
            phase = (2.0 * math.pi / self.time_scale) * global_t * frequencies
            time_input = torch.cat((global_t, phase.sin(), phase.cos()), dim=-1)
        else:
            time_input = global_t
        hidden = torch.cat((time_input, x), dim=-1)
        for k in range((len(self.net) - 1) // 2):
            layer_input = hidden
            if k > 0 and self.network_variant in {"prenorm_residual", "prenorm_residual_unbounded",
                                                   "prenorm_residual_linear_skip",
                                                   "prenorm_bounded_branch_residual"}:
                layer_input = self.residual_norms[k - 1](hidden)
            transformed = self.net[2 * k](layer_input)
            if k > 0:
                if self.network_variant == "residual":
                    transformed = transformed + hidden
                elif self.network_variant == "residual_scaled":
                    transformed = transformed + self.residual_scale * hidden
                elif self.network_variant == "residual_branch_scaled":
                    transformed = hidden + self.residual_branch_scale * transformed
                elif self.network_variant in {"prenorm_residual", "prenorm_residual_unbounded",
                                               "prenorm_residual_linear_skip"}:
                    transformed = hidden + self.residual_branch_scale * transformed
                elif self.network_variant == "prenorm_bounded_branch_residual":
                    transformed = hidden + self.residual_branch_scale * self.residual_output_norms[k - 1](transformed)
                elif self.network_variant in {"gated", "gated_residual"}:
                    gate = torch.sigmoid(self.gates[k - 1](hidden))
                    if self.network_variant == "gated":
                        transformed = gate * transformed + (1.0 - gate) * hidden
                    else:
                        transformed = hidden + gate * transformed
            hidden = self.net[2 * k + 1](transformed)
        output = self.net[-1](self.output_norm(hidden))
        if self.input_skip is not None:
            output = output + self.input_skip(torch.cat((time_input, x), dim=-1))
        return output


class FeedbackTwoHeadMLP(nn.Module):
    """One velocity trunk with separate regular and endpoint-singular heads."""
    def __init__(self, dimension, width=400, layers=6, activation="silu",
                 last_hidden_activation=None, network_variant="plain", time_scale=1.0):
        super().__init__()
        if activation not in FEEDBACK_ACTIVATIONS or (last_hidden_activation is not None
                                                       and last_hidden_activation not in FEEDBACK_ACTIVATIONS):
            raise ValueError("unsupported feedback activation")
        if network_variant not in {"plain", "residual", "residual_scaled", "residual_branch_scaled",
                                   "prenorm_residual", "prenorm_residual_unbounded",
                                   "prenorm_residual_linear_skip",
                                   "prenorm_bounded_branch_residual",
                                   "plain_output_norm",
                                   "plain_output_affine_norm",
                                   "gated", "gated_residual", "fourier"}:
            raise ValueError(f"unsupported feedback network variant: {network_variant}")
        self.network_variant = network_variant
        self.time_scale = float(time_scale)
        # Match FeedbackMLP's fixed-depth residual scaling exactly.
        self.residual_scale = 1.0 / max(layers - 1, 1)
        self.residual_branch_scale = 1.0 / math.sqrt(max(layers - 1, 1))
        self.residual_norms = nn.ModuleList(
            [nn.LayerNorm(width, elementwise_affine=False) for _ in range(max(layers - 1, 0))]
            if network_variant in {"prenorm_residual", "prenorm_residual_unbounded",
                                   "prenorm_residual_linear_skip", "prenorm_bounded_branch_residual"} else [])
        self.residual_output_norms = nn.ModuleList(
            [nn.LayerNorm(width, elementwise_affine=False) for _ in range(max(layers - 1, 0))]
            if network_variant == "prenorm_bounded_branch_residual" else [])
        self.output_norm = _feedback_output_norm(width, network_variant)
        last_hidden_activation = last_hidden_activation or activation
        input_width = dimension + 1 + (8 if network_variant == "fourier" else 0)
        self.input_skip_regular = (nn.Linear(input_width, dimension, bias=False)
                                   if network_variant == "prenorm_residual_linear_skip" else None)
        self.input_skip_singular = (nn.Linear(input_width, dimension, bias=False)
                                    if network_variant == "prenorm_residual_linear_skip" else None)
        modules = []
        for k in range(layers):
            hidden_activation = last_hidden_activation if k == layers - 1 else activation
            modules += [nn.Linear(input_width if k == 0 else width, width), _feedback_activation(hidden_activation)]
        self.net = nn.Sequential(*modules)
        self.regular_head = nn.Linear(width, dimension)
        self.singular_head = nn.Linear(width, dimension)
        self.gates = nn.ModuleList([nn.Linear(width, width) for _ in range(max(layers - 1, 0))]
                                   if network_variant in {"gated", "gated_residual"} else [])

    def forward(self, global_t, x):
        if self.network_variant == "fourier":
            frequencies = global_t.new_tensor((1.0, 2.0, 4.0, 8.0))
            phase = (2.0 * math.pi / self.time_scale) * global_t * frequencies
            time_input = torch.cat((global_t, phase.sin(), phase.cos()), dim=-1)
        else:
            time_input = global_t
        hidden = torch.cat((time_input, x), dim=-1)
        for k in range(len(self.net) // 2):
            layer_input = hidden
            if k > 0 and self.network_variant in {"prenorm_residual", "prenorm_residual_unbounded",
                                                   "prenorm_residual_linear_skip",
                                                   "prenorm_bounded_branch_residual"}:
                layer_input = self.residual_norms[k - 1](hidden)
            transformed = self.net[2 * k](layer_input)
            if k > 0:
                if self.network_variant == "residual":
                    transformed = transformed + hidden
                elif self.network_variant == "residual_scaled":
                    transformed = transformed + self.residual_scale * hidden
                elif self.network_variant == "residual_branch_scaled":
                    transformed = hidden + self.residual_branch_scale * transformed
                elif self.network_variant in {"prenorm_residual", "prenorm_residual_unbounded",
                                               "prenorm_residual_linear_skip"}:
                    transformed = hidden + self.residual_branch_scale * transformed
                elif self.network_variant == "prenorm_bounded_branch_residual":
                    transformed = hidden + self.residual_branch_scale * self.residual_output_norms[k - 1](transformed)
                elif self.network_variant in {"gated", "gated_residual"}:
                    gate = torch.sigmoid(self.gates[k - 1](hidden))
                    if self.network_variant == "gated":
                        transformed = gate * transformed + (1.0 - gate) * hidden
                    else:
                        transformed = hidden + gate * transformed
            hidden = self.net[2 * k + 1](transformed)
        hidden = self.output_norm(hidden)
        regular = self.regular_head(hidden)
        singular = self.singular_head(hidden)
        if self.input_skip_regular is not None:
            skip_input = torch.cat((time_input, x), dim=-1)
            regular = regular + self.input_skip_regular(skip_input)
            singular = singular + self.input_skip_singular(skip_input)
        return regular, singular


class Feedback(nn.Module):
    """Networks for finite/singular flow, growth G, and score S."""
    def __init__(self, dimension=2, width=400, layers=6, activation="silu", last_hidden_activation=None,
                 network_variant="plain", time_scale=1.0, field_parameterization="raw",
                 sigma=0.25, endpoint_guard=1e-5, independent_velocity_networks=False,
                 absorb_sigma_in_singular_velocity=False, growth_network_variant=None,
                 score_network_variant=None):
        super().__init__()
        if field_parameterization not in {"raw", "endpoint_regularized"}:
            raise ValueError(f"unsupported field parameterization: {field_parameterization}")
        if field_parameterization == "endpoint_regularized" and sigma <= 0:
            raise ValueError("endpoint_regularized fields require positive sigma")
        if not 0 < endpoint_guard < 0.5:
            raise ValueError("endpoint_guard must be in (0,0.5)")
        self.activation = activation
        self.last_hidden_activation = last_hidden_activation or activation
        self.network_variant = network_variant
        growth_network_variant = growth_network_variant or network_variant
        self.growth_network_variant = growth_network_variant
        score_network_variant = score_network_variant or network_variant
        self.score_network_variant = score_network_variant
        self.time_scale = float(time_scale)
        self.field_parameterization = field_parameterization
        self.sigma = float(sigma)
        self.endpoint_guard = float(endpoint_guard)
        self.independent_velocity_networks = bool(independent_velocity_networks)
        self.absorb_sigma_in_singular_velocity = bool(absorb_sigma_in_singular_velocity)
        if self.independent_velocity_networks and field_parameterization != "endpoint_regularized":
            raise ValueError("independent velocity networks require endpoint_regularized parameterization")
        if self.absorb_sigma_in_singular_velocity and field_parameterization != "endpoint_regularized":
            raise ValueError("sigma-normalized singular flow requires endpoint_regularized parameterization")
        if field_parameterization == "endpoint_regularized" and self.independent_velocity_networks:
            # Four independent MLPs: finite flow, endpoint-singular flow,
            # growth, and score. Their targets have distinct time behavior.
            self.flow = None
            self.flow_regular = FeedbackMLP(dimension, dimension, width, layers, activation,
                                            last_hidden_activation, network_variant, time_scale)
            self.flow_singular = FeedbackMLP(dimension, dimension, width, layers, activation,
                                             last_hidden_activation, network_variant, time_scale)
        elif field_parameterization == "endpoint_regularized":
            # Retain the original shared-trunk version for old checkpoints.
            self.flow = FeedbackTwoHeadMLP(dimension, width, layers, activation, last_hidden_activation,
                                           network_variant, time_scale)
            self.flow_regular = self.flow_singular = None
        else:
            self.flow = FeedbackMLP(dimension, dimension, width, layers, activation, last_hidden_activation,
                                    network_variant, time_scale)
            self.flow_regular = self.flow_singular = None
        self.growth = FeedbackMLP(dimension, 1, width, layers, activation, last_hidden_activation,
                                  growth_network_variant, time_scale)
        self.score = FeedbackMLP(dimension, dimension, width, layers, activation, last_hidden_activation,
                                 score_network_variant, time_scale)

    def matching_fields(self, t, x):
        """Return the fields directly fitted by Matching.

        In endpoint_regularized mode the flow networks return the finite part
        and a regularized singular coefficient; the score head returns
        sigma*sqrt(tau)*Score. These targets stay finite for bridge paths.
        """
        if self.field_parameterization == "endpoint_regularized":
            if self.independent_velocity_networks:
                flow_regular = self.flow_regular(t, x)
                flow_singular_scaled = self.flow_singular(t, x)
            else:
                flow_regular, flow_singular_scaled = self.flow(t, x)
            return flow_regular, flow_singular_scaled, self.growth(t, x), self.score(t, x)
        return self.flow(t, x), self.growth(t, x), self.score(t, x)

    def _decode_fields(self, t, fields):
        if self.field_parameterization == "endpoint_regularized":
            flow_regular, flow_singular_scaled, growth, score_scaled = fields
        else:
            flow, growth, score = fields
            return flow, growth, score
        local_t = t - torch.floor(t)
        tau = local_t * (1.0 - local_t)
        min_tau = self.endpoint_guard * (1.0 - self.endpoint_guard)
        sqrt_tau = tau.clamp_min(min_tau).sqrt()
        singular_scale = self.sigma if self.absorb_sigma_in_singular_velocity else 1.0
        flow = flow_regular + singular_scale * flow_singular_scaled / sqrt_tau
        score = score_scaled / (self.sigma * sqrt_tau)
        return flow, growth, score

    def forward(self, t, x):
        return self._decode_fields(t, self.matching_fields(t, x))


MATCHING_VERSION = 5


def matching_version(config):
    matching = config.get("matching", {})
    if float(matching.get("growth_loss_multiplier", 1.0)) != 1.0:
        return 9
    if (matching.get("field_parameterization") == "endpoint_regularized"
            and matching.get("absorb_sigma_in_singular_velocity", False)):
        return 8
    if matching.get("loss_weighting", "none") == "endpoint_sigma_balanced":
        return 7
    # Version 6 splits velocity supervision into its finite and singular parts.
    return 6 if matching.get("field_parameterization") == "endpoint_regularized" else MATCHING_VERSION


def make_targets(state, noise, sigma=0.25, alpha=10, validate=True,
                 field_parameterization="raw", absorb_sigma_in_singular_velocity=False):
    if field_parameterization not in {"raw", "endpoint_regularized"}:
        raise ValueError(f"unsupported field parameterization: {field_parameterization}")
    if field_parameterization == "endpoint_regularized" and sigma <= 0:
        raise ValueError("endpoint_regularized targets require positive sigma")
    if validate and torch.any(state["var"] <= 0):
        raise ValueError("matching targets require open local times")
    sqrt_var = state["var"].sqrt()
    centered = spectral_apply(state, sqrt_var, noise)
    x = state["xi"] + centered
    exp_halfZ = (0.5 * state["Z"]).exp()
    target_s_scaled = -spectral_apply(state, 1 / exp_halfZ, noise)
    sqrt_wf = state["tau"].sqrt()
    target_v_regular = -state["b"]
    target_v_singular_scaled = spectral_apply(
        state, (sigma ** 2 / 2 - state["K"]) / (sigma * exp_halfZ), noise)
    target_v_scaled = sqrt_wf * target_v_regular + target_v_singular_scaled
    target_g = state["gamma"] - ((state["b"] * centered).sum(-1, keepdim=True)
                + 0.5 * ((noise * spectral_apply(state, state["K"], noise)).sum(-1, keepdim=True)
                         - state["K"].sum(-1, keepdim=True))) / alpha
    if state.get("cost_objective") in {"tight_logmoment_upper_bound", "gaussian_reference_upper_bound", "diffusion_full_square_upper_bound"}:
        target_v_scaled = sqrt_wf*state["drift_mean"]+spectral_apply(
            state,(state["drift_times_var"]+sigma**2/2)/(sigma*exp_halfZ),noise)
        target_g = state["gamma"]+(state["growth_linear"]*centered).sum(-1,keepdim=True)
        target_g = target_g+.5*((noise*spectral_apply(state,state["growth_quadratic"],noise)).sum(-1,keepdim=True)
                               -state["growth_quadratic"].sum(-1,keepdim=True))
    if field_parameterization == "endpoint_regularized":
        # The finite mean/drift contribution and the singular fluctuation are
        # supervised independently, each in its natural finite scale.
        sqrt_wf = torch.ones_like(sqrt_wf)
        sqrt_ws = torch.ones_like(sqrt_wf)
    else:
        sqrt_ws = sigma * sqrt_wf
    targets = dict(x=x, target_v_scaled=target_v_scaled, target_g=target_g,
                target_s_scaled=target_s_scaled, mass=state["mass"],
                sqrt_wf=sqrt_wf, sqrt_ws=sqrt_ws)
    if field_parameterization == "endpoint_regularized":
        if state.get("cost_objective") in {"tight_logmoment_upper_bound", "gaussian_reference_upper_bound", "diffusion_full_square_upper_bound"}:
            target_v_regular = state["drift_mean"]
            target_v_singular_scaled = spectral_apply(
                state, (state["drift_times_var"] + sigma ** 2 / 2) / (sigma * exp_halfZ), noise)
        targets["target_v_regular"] = target_v_regular
        targets["target_v_singular_scaled"] = (target_v_singular_scaled / sigma
                                                if absorb_sigma_in_singular_velocity
                                                else target_v_singular_scaled)
    return targets


def matching_loss_weights(config):
    """Return fixed field-loss weights selected by the parameterization.

    The legacy endpoint encoding leaves sigma in the singular-flow label and
    cancels it with a 1/sigma**2 loss weight. The normalized encoding absorbs
    sigma into the decoder and uses unit singular-flow weight. Growth uses its
    quadratic RUOT action coefficient alpha; the score label is already
    sigma-scaled by its parameterization.
    """
    matching = config.get("matching", {})
    mode = matching.get("loss_weighting", "none")
    sigma = float(config.get("model_sigma", 0.25))
    alpha = float(config.get("growth_penalty", 1.0))
    growth_multiplier = float(matching.get("growth_loss_multiplier", 1.0))
    if not math.isfinite(growth_multiplier) or growth_multiplier <= 0:
        raise ValueError("growth_loss_multiplier must be finite and positive")
    if mode == "none":
        return {"flow_regular": 1.0, "flow_singular": 1.0,
                "flow": 1.0, "growth": growth_multiplier, "score": 1.0}
    if mode != "endpoint_sigma_balanced":
        if mode != "endpoint_explicit_sigma":
            raise ValueError(f"unknown Matching loss weighting: {mode}")
    if matching.get("field_parameterization") != "endpoint_regularized":
        raise ValueError("endpoint field weights require endpoint_regularized fields")
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("sigma and growth_penalty must be finite and positive for loss weighting")
    absorb_sigma = matching.get("absorb_sigma_in_singular_velocity", False)
    if mode == "endpoint_explicit_sigma" and not absorb_sigma:
        raise ValueError("endpoint_explicit_sigma requires sigma-normalized singular-flow parameterization")
    return {"flow_regular": 1.0,
            "flow_singular": (1.0 if absorb_sigma else 1.0 / (sigma * sigma)),
            "flow": 1.0, "growth": alpha * growth_multiplier, "score": 1.0}


def matching_loss(prediction, targets, interval_factor, loss_weights=None):
    weights = loss_weights or {}
    if "target_v_regular" in targets:
        v_regular, v_singular_scaled, g, score = prediction
        lf_regular = (v_regular - targets["target_v_regular"]).square().sum(-1)
        lf_singular = (v_singular_scaled - targets["target_v_singular_scaled"]).square().sum(-1)
        lf = (weights.get("flow_regular", 1.0) * lf_regular
              + weights.get("flow_singular", 1.0) * lf_singular)
    else:
        v, g, score = prediction
        lf_regular = (targets["sqrt_wf"] * v - targets["target_v_scaled"]).square().sum(-1)
        lf_singular = torch.zeros_like(lf_regular)
        lf = weights.get("flow", 1.0) * lf_regular
    lg = (g - targets["target_g"]).square().squeeze(-1)
    ls = (targets["sqrt_ws"] * score - targets["target_s_scaled"]).square().sum(-1)
    mass_factor = interval_factor * targets["mass"].squeeze(-1)
    flow_regular = weights.get("flow_regular", weights.get("flow", 1.0)) * (mass_factor * lf_regular).mean()
    flow_singular = weights.get("flow_singular", 1.0) * (mass_factor * lf_singular).mean()
    flow = (mass_factor * lf).mean()
    growth = weights.get("growth", 1.0) * (mass_factor * lg).mean()
    score = weights.get("score", 1.0) * (mass_factor * ls).mean()
    return flow + growth + score, dict(flow=flow, flow_regular=flow_regular,
           flow_singular=flow_singular, growth=growth, score=score)


class CouplingSampler:
    def __init__(self, run, config, device):
        self.config = config
        run = Path(run)
        status_file = run / "status.json"
        if not status_file.exists() or json.loads(status_file.read_text(encoding="utf-8")).get("couplings", {}).get("status") != "completed":
            raise RuntimeError("matching requires successfully completed couplings")
        manifest = json.loads((run / "couplings" / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "converged" or manifest["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt"):
            raise RuntimeError("missing/stale/unconverged couplings")
        if config['coupling'].get('cost_evaluation') == 'shared_distance_hermite':
            if (manifest.get('cost_evaluation') != 'shared_distance_hermite' or
                    manifest.get('table_sha256') != file_hash(run/'distance_cost_table.pt')):
                raise RuntimeError('couplings do not match the shared path cost table')
        if config.get("distillation", {}).get("enabled"):
            if (manifest.get("cost_oracle") != "distilled_mlp"
                    or manifest.get("cost_network_sha256") != file_hash(run / "cost_network.pt")):
                raise RuntimeError("couplings do not match the distilled cost network")
        snapshots = snapshots_for(config)
        self.masses = torch.tensor([s.mass for s in snapshots[:-1]], dtype=torch.float64, device=device)
        # Store endpoint coordinates once and pair indices, rather than copying
        # x0 and x1-x0 once for every supported pair.  For real datasets this
        # reduces GPU storage from O(sum_ij P_ij * dimension) to
        # O(sum_i x_i * dimension + sum_ij P_ij).
        source_positions, target_positions = [], []
        pair_source, pair_target, log_ratios, probabilities = [], [], [], []
        interval_ends = []
        pair_count = 0
        for k in range(len(snapshots) - 1):
            blocks = [block for block in manifest["intervals"] if block["interval"] == k]
            if not blocks:
                raise RuntimeError(f"no coupling for interval {k}")
            probability_start = len(probabilities)
            interval_mass = 0.0
            for block in blocks:
                if block["status"] != "converged":
                    raise RuntimeError("unconverged coupling cannot be used for matching")
                with np.load(run / "couplings" / block["file"]) as saved:
                    P, Q, r = saved["P"], saved["Q"], saved["r"]
                    # Recheck saved matrices and support before any training.
                    if np.abs(P.sum(1) - saved["source_weights"]).sum() / saved["source_weights"].sum() >= config["coupling"]["marginal_tolerance"]:
                        raise RuntimeError("cached P row constraint failure")
                    if np.abs(Q.sum(0) - saved["target_weights"]).sum() / saved["target_weights"].sum() >= config["coupling"]["marginal_tolerance"]:
                        raise RuntimeError("cached Q column constraint failure")
                    src = np.asarray(saved["source_positions"], dtype=np.float64)
                    tgt = np.asarray(saved["target_positions"], dtype=np.float64)
                    src_offset = sum(len(a) for a in source_positions)
                    tgt_offset = sum(len(a) for a in target_positions)
                    ns, nt = P.shape
                    if r.shape != P.shape or src.shape[0] != ns or tgt.shape[0] != nt:
                        raise RuntimeError("cached coupling dimensions are inconsistent")
                    if src_offset + ns > np.iinfo(np.int32).max or tgt_offset + nt > np.iinfo(np.int32).max:
                        raise RuntimeError("too many endpoint positions for compact pair indices")
                    pair_source.append(np.repeat(np.arange(ns, dtype=np.int32) + src_offset, nt))
                    pair_target.append(np.tile(np.arange(nt, dtype=np.int32) + tgt_offset, ns))
                    flat_r = np.asarray(r, dtype=np.float64).reshape(-1)
                    if not np.isfinite(flat_r).all() or np.any(flat_r <= 0):
                        raise RuntimeError("cached coupling contains invalid mass ratios")
                    log_ratios.append(np.log(flat_r))
                    flat_p = np.asarray(P, dtype=np.float64).reshape(-1)
                    probabilities.append(flat_p)
                    interval_mass += float(flat_p.sum())
                    source_positions.append(src)
                    target_positions.append(tgt)
            if interval_mass <= 0:
                raise RuntimeError(f"nonpositive coupling mass in interval {k}")
            # Normalize each P block by the total interval mass, preserving the
            # existing law: choose an interval uniformly, then a pair according
            # to P / mass(interval).
            for idx in range(probability_start, len(probabilities)):
                probabilities[idx] /= interval_mass * (len(snapshots) - 1)
                pair_count += len(probabilities[idx])
            interval_ends.append(pair_count)

        def to_device(chunks, dtype):
            return torch.as_tensor(np.concatenate(chunks), dtype=dtype, device=device)

        self.source_positions = to_device(source_positions, torch.float64)
        self.target_positions = to_device(target_positions, torch.float64)
        self.pair_source = to_device(pair_source, torch.int32)
        self.pair_target = to_device(pair_target, torch.int32)
        self.log_r = to_device(log_ratios, torch.float64).reshape(-1, 1)
        probability = to_device(probabilities, torch.float64)
        self.cdf = probability.cumsum(0)
        self.cdf /= self.cdf[-1].clone()
        self.cdf[-1] = 1
        self.interval_ends = torch.as_tensor(interval_ends[:-1], dtype=torch.long, device=device)
        self.masses = self.masses.to(device=device, dtype=torch.float64)
        if len(self.cdf) != len(self.log_r) or len(self.log_r) != len(self.pair_source):
            raise RuntimeError("compact coupling sampler arrays have different lengths")

    def sample(self, n, device, generator=None):
        uniforms = torch.rand(n, dtype=self.cdf.dtype, device=device, generator=generator)
        ids = torch.searchsorted(self.cdf, uniforms, right=True).clamp_max(len(self.cdf) - 1)
        src = self.source_positions[self.pair_source[ids].long()]
        tgt = self.target_positions[self.pair_target[ids].long()]
        k = torch.bucketize(ids, self.interval_ends, right=True)
        result = {"x0": src, "delta": tgt - src, "log_r": self.log_r[ids]}
        result["interval"] = k
        # p_k=1/K. This is multiplied once by the current conditional mass.
        result["interval_factor"] = len(self.masses) * self.masses[k]
        return result


@torch.no_grad()
def sample_training_batch(sampler, path, n, device, generator=None):
    batch = sampler.sample(n, device, generator=generator)
    eps = torch.finfo(torch.float64).eps
    local_t = eps / 2 + (1 - eps) * torch.rand((n, 1), dtype=torch.float64,
                                             device=device, generator=generator)
    # PairPath computes exact time JVPs explicitly; labels need no autograd graph.
    state = path(local_t, batch["delta"], batch["log_r"], create_graph=False, x0=batch["x0"])
    noise = torch.randn(batch["delta"].shape, dtype=torch.float64, device=device, generator=generator)
    sampler_config = getattr(sampler, "config", {})
    parameterization = sampler_config.get("matching", {}).get("field_parameterization", "raw")
    targets = make_targets(state, noise, path.sigma, path.alpha, validate=False,
                           field_parameterization=parameterization,
                           absorb_sigma_in_singular_velocity=sampler_config.get("matching", {}).get(
                               "absorb_sigma_in_singular_velocity", False))
    targets = {key: value.float() for key, value in targets.items()}
    global_t = (batch["interval"].reshape(-1, 1) + local_t).float()
    return global_t, targets, batch["interval_factor"].float()


def matching_learning_rate(cfg, step):
    schedule = cfg.get("lr_schedule", "constant")
    if schedule == "constant":
        return cfg["learning_rate"]
    if schedule not in {"cosine", "cosine_2cycle", "cosine_2cycle_half"}:
        raise ValueError(f"unknown matching lr_schedule: {schedule}")
    floor = cfg.get("min_learning_rate", 0.0)
    if not 0 <= floor <= cfg["learning_rate"]:
        raise ValueError("min_learning_rate must be between zero and learning_rate")
    cycle_peak, cycle_floor = cfg["learning_rate"], floor
    if schedule in {"cosine_2cycle", "cosine_2cycle_half"}:
        total_steps = int(cfg["steps"])
        if total_steps < 2:
            raise ValueError(f"{schedule} requires at least two Matching steps")
        cycle = min((2 * int(step)) // total_steps, 1)
        cycle_start = (cycle * total_steps) // 2
        cycle_end = ((cycle + 1) * total_steps) // 2
        progress = min(max((step - cycle_start) / max(cycle_end - cycle_start - 1, 1), 0.0), 1.0)
        if schedule == "cosine_2cycle_half" and cycle == 1:
            cycle_peak *= 0.5
            cycle_floor *= 0.5
    else:
        progress = min(step / max(cfg["steps"] - 1, 1), 1)
    return cycle_floor + (cycle_peak - cycle_floor) * (1 + math.cos(math.pi * progress)) / 2


def make_feedback(config, device):
    c = config["matching"]
    times = config.get("model_times", [0.0, 1.0])
    time_scale = float(times[-1] - times[0])
    return Feedback(config["dimension"], c["hidden_width"], c["hidden_layers"],
                    c.get("feedback_activation", "silu"), c.get("last_hidden_activation"),
                    c.get("network_variant", "plain"), time_scale,
                    c.get("field_parameterization", "raw"), config.get("model_sigma", 0.25),
                    config.get("generation", {}).get("endpoint_guard", 1e-5),
                    c.get("independent_velocity_networks", False),
                    c.get("absorb_sigma_in_singular_velocity", False),
                    c.get("growth_network_variant"),
                    c.get("score_network_variant"))\
        .to(device=device, dtype=torch.float32)


def feedback_architecture(config):
    matching = config["matching"]
    hidden = matching.get("feedback_activation", "silu")
    architecture = {
        "hidden_activation": hidden,
        "last_hidden_activation": matching.get("last_hidden_activation", hidden),
        "output_activation": "linear",
        "network_variant": matching.get("network_variant", "plain"),
        "growth_network_variant": matching.get("growth_network_variant",
                                                 matching.get("network_variant", "plain")),
        "score_network_variant": matching.get("score_network_variant",
                                                matching.get("network_variant", "plain")),
        "field_parameterization": matching.get("field_parameterization", "raw"),
        "independent_velocity_networks": bool(matching.get("independent_velocity_networks", False)),
        "absorb_sigma_in_singular_velocity": bool(matching.get("absorb_sigma_in_singular_velocity", False)),
    }
    if architecture["network_variant"] == "fourier":
        times = config.get("model_times", [0.0, 1.0])
        architecture["time_scale"] = float(times[-1] - times[0])
        architecture["time_frequencies"] = [1.0, 2.0, 4.0, 8.0]
    if architecture["field_parameterization"] == "endpoint_regularized":
        architecture["sigma"] = float(config["model_sigma"])
        architecture["endpoint_guard"] = float(config.get("generation", {}).get("endpoint_guard", 1e-5))
    if "leaky_relu" in (architecture["hidden_activation"], architecture["last_hidden_activation"]):
        architecture["leaky_relu_negative_slope"] = LEAKY_RELU_NEGATIVE_SLOPE
    return architecture


def validate_feedback_architecture(checkpoint, config):
    architecture = checkpoint.get("feedback_architecture")
    if architecture is None:
        saved_config = checkpoint.get("config", {})
        saved_matching = saved_config.get("matching", {})
        hidden = saved_matching.get("feedback_activation", "silu")
        architecture = feedback_architecture(saved_config) if saved_matching else {
            "hidden_activation": hidden, "last_hidden_activation": hidden,
            "output_activation": "linear", "network_variant": "plain",
            "growth_network_variant": "plain",
            "score_network_variant": "plain",
            "field_parameterization": "raw", "independent_velocity_networks": False,
            "absorb_sigma_in_singular_velocity": False}
    elif architecture == {"stability_radius": None, "stability_width": None}:
        architecture = {"hidden_activation": "silu", "last_hidden_activation": "silu",
                        "output_activation": "linear", "network_variant": "plain",
                        "growth_network_variant": "plain",
                        "score_network_variant": "plain",
                        "field_parameterization": "raw", "independent_velocity_networks": False,
                        "absorb_sigma_in_singular_velocity": False}
    else:
        architecture = dict(architecture)
        architecture.setdefault("last_hidden_activation", architecture.get("hidden_activation", "silu"))
        architecture.setdefault("output_activation", "linear")
        architecture.setdefault("network_variant", "plain")
        architecture.setdefault("growth_network_variant", architecture.get("network_variant", "plain"))
        architecture.setdefault("score_network_variant", architecture.get("network_variant", "plain"))
        architecture.setdefault("field_parameterization", "raw")
        architecture.setdefault("independent_velocity_networks", False)
        architecture.setdefault("absorb_sigma_in_singular_velocity", False)
        if architecture["network_variant"] == "fourier":
            times = config.get("model_times", [0.0, 1.0])
            architecture.setdefault("time_scale", float(times[-1] - times[0]))
            architecture.setdefault("time_frequencies", [1.0, 2.0, 4.0, 8.0])
    if architecture != feedback_architecture(config):
        raise RuntimeError("feedback checkpoint uses a different network architecture")


def train_matching(config, run):
    device = configure(config)
    run = init_run(run, config)
    sampler = CouplingSampler(run, config, device)
    path = load_path(run, config, device)
    model = make_feedback(config, device)
    cfg = config["matching"]
    loss_weights = matching_loss_weights(config)
    objective_version = matching_version(config)
    matching_learning_rate(cfg, 0)
    diagnostic_rollout_steps = set(int(value) for value in cfg.get("diagnostic_rollout_steps", []))
    if any(value < 1 or value > cfg["steps"] for value in diagnostic_rollout_steps):
        raise ValueError("diagnostic_rollout_steps must be within the Matching training range")
    diagnostic_stop_after_step = cfg.get("diagnostic_stop_after_step")
    if diagnostic_stop_after_step is not None:
        diagnostic_stop_after_step = int(diagnostic_stop_after_step)
        if diagnostic_stop_after_step not in diagnostic_rollout_steps:
            raise ValueError("diagnostic_stop_after_step must be one of diagnostic_rollout_steps")
    diagnostic_rollout_seeds = tuple(int(value) for value in cfg.get("diagnostic_rollout_seeds", (342, 343)))
    diagnostic_snapshots = snapshots_for(config, full=True) if diagnostic_rollout_steps else None
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    gradient_clip_norm = float(cfg.get("gradient_clip_norm", 10.0))
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("matching.gradient_clip_norm must be finite and positive")
    started = time.perf_counter()
    first_step, prior_seconds = 0, 0.0
    checkpoint_file = run / "feedback.pt"
    if checkpoint_file.exists():
        cp = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if cp.get("matching_version", MATCHING_VERSION) != objective_version:
            raise RuntimeError("feedback checkpoint uses a different Matching objective")
        validate_feedback_architecture(cp, config)
        if cp["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt") or cp["coupling_manifest_sha256"] != file_hash(run / "couplings" / "manifest.json"):
            raise RuntimeError("feedback checkpoint belongs to different paths/couplings")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        first_step, prior_seconds = cp["completed_steps"], cp["seconds"]
        torch.set_rng_state(cp["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in cp["cuda_rng"]])
    validation = None
    validation_every = cfg.get("validation_every", 0)
    if validation_every:
        generator = torch.Generator(device=device).manual_seed(cfg.get("validation_seed", config["seed"] + 7919))
        validation = sample_training_batch(sampler, path, cfg.get("validation_batch_size", cfg["batch_size"]),
                                           device, generator=generator)
    path_hash = file_hash(run / "pair_path.pt")
    coupling_hash = file_hash(run / "couplings" / "manifest.json")
    update_stage(run, "matching", "running", completed_steps=first_step)
    (run / "matching_loss_weights.json").write_text(
        json.dumps(loss_weights, indent=2, ensure_ascii=False), encoding="utf-8")
    last_print = time.perf_counter()
    for step in range(first_step, cfg["steps"]):
        global_t, targets, factor = sample_training_batch(sampler, path, cfg["batch_size"], device)
        for group in optimizer.param_groups:
            group["lr"] = matching_learning_rate(cfg, step)
        prediction = model.matching_fields(global_t, targets["x"])
        loss, parts = matching_loss(prediction, targets, factor, loss_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if model.independent_velocity_networks:
            # Clip each independently optimized field separately so
            # action-scaled Growth gradients do not shrink the other fields'
            # updates through a shared global clipping norm.
            submodels = (model.flow_regular, model.flow_singular, model.growth, model.score)
            grad_parts = [torch.nn.utils.clip_grad_norm_(submodel.parameters(), gradient_clip_norm,
                                                         error_if_nonfinite=False)
                          for submodel in submodels]
            grad = torch.linalg.vector_norm(torch.stack(grad_parts))
        else:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm,
                                                   error_if_nonfinite=False)
        # Avoid a device-to-host barrier every step. Finiteness is checked at
        # existing logging/checkpoint boundaries before accepting a checkpoint.
        log_step = (step + 1) % cfg["log_every"] == 0 or step == first_step or step + 1 == cfg["steps"]
        if log_step or (step + 1) % 1000 == 0:
            if not torch.isfinite(torch.stack((loss, grad))).all().item():
                raise FloatingPointError("nonfinite matching loss/gradient")
        optimizer.step()
        if step + 1 in diagnostic_rollout_steps:
            # This opt-in evaluation is outside the loss graph and uses a
            # private RNG in rollout(), so it does not alter Matching samples,
            # optimizer state, or the objective. It pinpoints when instability
            # first appears along training without retaining large checkpoints.
            from .sample import rollout
            for diagnostic_seed in diagnostic_rollout_seeds:
                diagnostic = dict(step=step + 1, seed=diagnostic_seed,
                                  matching_version=objective_version)
                try:
                    generated = rollout(
                        model=model, first_snapshot=diagnostic_snapshots[0].x,
                        times=config["model_times"], sigma=config["model_sigma"],
                        n_particles=len(diagnostic_snapshots[0].x),
                        steps_per_unit=config["generation"]["steps_per_unit_interval"],
                        seed=diagnostic_seed,
                        endpoint_guard=config["generation"]["endpoint_guard"],
                        initial_x=diagnostic_snapshots[0].x, stochastic=True)
                    diagnostics = []
                    for time_index, (positions, log_weights, snapshot) in enumerate(zip(
                            generated["positions"], generated["log_weights"], diagnostic_snapshots)):
                        log_weights = np.asarray(log_weights, dtype=np.float64)
                        log_total_mass = float(np.logaddexp.reduce(log_weights))
                        probabilities = np.exp(log_weights - log_total_mass)
                        log_mass_ratio = float(log_total_mass - diagnostics[0]["log_total_mass"]) if diagnostics else 0.0
                        mass_ratio = (float(math.exp(log_mass_ratio))
                                      if log_mass_ratio <= math.log(np.finfo(np.float64).max)
                                      else None)
                        observed_mass_ratio = float(snapshot.mass / diagnostic_snapshots[0].mass)
                        diagnostics.append(dict(
                            time=float(snapshot.time), log_total_mass=log_total_mass,
                            log_mass_ratio=log_mass_ratio, predicted_mass_ratio=mass_ratio,
                            observed_mass_ratio=observed_mass_ratio,
                            mass_relative_error=(abs(mass_ratio - observed_mass_ratio)
                                                 / max(abs(observed_mass_ratio), 1e-12)
                                                 if mass_ratio is not None else None),
                            ESS=float(1.0 / np.square(probabilities).sum()),
                            max_abs_position=float(np.abs(positions).max()),
                            max_abs_log_weight=float(np.abs(log_weights).max()),
                        ))
                    diagnostic.update(status=("mass_range_overflow"
                                              if any(row["predicted_mass_ratio"] is None for row in diagnostics)
                                              else "completed"), metrics=diagnostics)
                except FloatingPointError as error:
                    diagnostic.update(status="rollout_failure", failure_reason=str(error))
                log_json(run / "matching_rollout_diagnostics.jsonl", diagnostic)
        if log_step:
            record = dict(step=step + 1, loss=loss.item(), **{k: v.item() for k, v in parts.items()},
                          grad_norm=grad.item(), learning_rate=optimizer.param_groups[0]["lr"],
                          matching_version=objective_version, seconds=prior_seconds + time.perf_counter() - started)
            log_json(run / "matching_history.jsonl", record)
        if validation is not None and ((step + 1) % validation_every == 0 or step == first_step or step + 1 == cfg["steps"]):
            with torch.no_grad():
                vt, vy, vf = validation
                vl, vp = matching_loss(model.matching_fields(vt, vy["x"]), vy, vf, loss_weights)
            log_json(run / "matching_validation.jsonl", dict(step=step + 1, loss=vl.item(),
                     **{key: value.item() for key, value in vp.items()},
                     samples=len(vt), matching_version=objective_version))
        if time.perf_counter() - last_print > 20 or step + 1 == cfg["steps"]:
            print(f"Matching {step + 1}/{cfg['steps']}: loss={loss.item():.5g}", flush=True)
            last_print = time.perf_counter()
        if (step + 1) % 1000 == 0 or step + 1 == cfg["steps"]:
            temp = checkpoint_file.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), completed_steps=step + 1,
                            matching_version=objective_version,
                            feedback_architecture=feedback_architecture(config),
                            seconds=prior_seconds + time.perf_counter() - started, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                            path_checkpoint_sha256=path_hash,
                            coupling_manifest_sha256=coupling_hash, config=config), temp)
            temp.replace(checkpoint_file)
            update_stage(run, "matching", "running", completed_steps=step + 1)
            if diagnostic_stop_after_step == step + 1:
                update_stage(run, "matching", "diagnostic_stopped", completed_steps=step + 1,
                             seconds=prior_seconds + time.perf_counter() - started)
                return model
    update_stage(run, "matching", "completed", completed_steps=cfg["steps"], seconds=prior_seconds + time.perf_counter() - started)
    return model

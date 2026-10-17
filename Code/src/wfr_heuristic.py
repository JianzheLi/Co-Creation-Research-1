"""WFR-FM semi-couplings with an analytic WFR mean/mass bridge heuristic."""
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import ot

from .matching import (feedback_architecture, make_feedback, matching_learning_rate, matching_version,
                      matching_loss, validate_feedback_architecture)
from .data import partition_indices
from .runtime import (configure, file_hash, init_run, json_write, log_json,
                      snapshots_for, update_stage)


class WFRBridgePath:
    """Traveling WFR mean/mass path with a Brownian-bridge variance."""
    def __init__(self, alpha=10.0, sigma=0.25):
        if alpha <= 0 or sigma <= 0:
            raise ValueError("alpha and sigma must be positive")
        self.alpha, self.delta, self.sigma = float(alpha), math.sqrt(float(alpha)), float(sigma)

    def __call__(self, t, x0, delta, log_r):
        r = log_r.exp()
        sqrt_r = torch.sqrt(r)
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        theta = distance / (2 * self.delta)
        if bool(torch.any(theta >= math.pi / 2)):
            raise ValueError("traveling WFR path is undefined for pairs beyond the pi*sqrt(alpha) cutoff")
        cosine, sine = theta.cos(), theta.sin()
        A = 1 + r - 2 * sqrt_r * cosine
        B = 1 - sqrt_r * cosine
        mass = A * t.square() - 2 * B * t + 1
        mass_dot = 2 * A * t - 2 * B
        gamma = mass_dot / mass

        angle = torch.atan2(t * sqrt_r * sine,
                            (1 - t) + t * sqrt_r * cosine)
        # angle/theta -> t*sqrt(r) / (1-t+t*sqrt(r)) as theta -> 0.
        safe_theta = theta.clamp_min(torch.finfo(theta.dtype).tiny)
        limit = t * sqrt_r / ((1 - t) + t * sqrt_r).clamp_min(torch.finfo(t.dtype).tiny)
        fraction = torch.where(theta > 1e-7, angle / safe_theta, limit)
        xi = x0 + fraction * delta
        unit = delta / distance.clamp_min(torch.finfo(distance.dtype).tiny)
        omega = 2 * self.delta * sqrt_r * sine
        dot_xi = omega / mass * unit

        tau = t * (1 - t)
        var = self.sigma ** 2 * tau
        dotvar = self.sigma ** 2 * (1 - 2 * t)
        return dict(xi=xi, dotxi=dot_xi, mass=mass, gamma=gamma, tau=tau,
                    var=var, dotvar=dotvar)

    def conditional_samples(self, t, x0, delta, log_r, noise):
        state = self(t, x0, delta, log_r)
        sqrt_var = state["var"].sqrt()
        x = state["xi"] + sqrt_var * noise
        # Probability-flow velocity for the Gaussian bridge; the SDE rollout
        # later adds sigma^2/2 times this score to its drift.
        v = state["dotxi"] + state["dotvar"] / (2 * state["var"]) * (x - state["xi"])
        score = -noise / sqrt_var
        sqrt_tau = state["tau"].sqrt()
        target_v_singular_scaled = sqrt_tau * v - sqrt_tau * state["dotxi"]
        return dict(x=x, target_v_scaled=sqrt_tau * v,
                    target_v_regular=state["dotxi"],
                    target_v_singular_scaled=target_v_singular_scaled,
                    target_g=state["gamma"], target_s_scaled=-noise,
                    mass=state["mass"], sqrt_wf=sqrt_tau,
                    sqrt_ws=self.sigma * sqrt_tau)


class WFRBrownianBridgePath:
    """Linear-mean Brownian bridge with exponential interpolation of pair mass."""
    def __init__(self, alpha=10.0, sigma=0.25):
        if alpha <= 0 or sigma <= 0:
            raise ValueError("alpha and sigma must be positive")
        self.alpha, self.sigma = float(alpha), float(sigma)

    def __call__(self, t, x0, delta, log_r):
        xi = x0 + t * delta
        mass = torch.exp(t * log_r)
        tau = t * (1 - t)
        var = self.sigma ** 2 * tau
        dotvar = self.sigma ** 2 * (1 - 2 * t)
        return dict(xi=xi, dotxi=delta, mass=mass, gamma=log_r, tau=tau,
                    var=var, dotvar=dotvar)

    def conditional_samples(self, t, x0, delta, log_r, noise):
        state = self(t, x0, delta, log_r)
        sqrt_var = state["var"].sqrt()
        x = state["xi"] + sqrt_var * noise
        # Probability-flow velocity of N(x0+t*delta, sigma^2*t*(1-t) I).
        v = state["dotxi"] + state["dotvar"] / (2 * state["var"]) * (x - state["xi"])
        sqrt_tau = state["tau"].sqrt()
        target_v_singular_scaled = sqrt_tau * v - sqrt_tau * state["dotxi"]
        return dict(x=x, target_v_scaled=sqrt_tau * v,
                    target_v_regular=state["dotxi"],
                    target_v_singular_scaled=target_v_singular_scaled,
                    target_g=state["gamma"], target_s_scaled=-noise,
                    mass=state["mass"], sqrt_wf=sqrt_tau,
                    sqrt_ws=self.sigma * sqrt_tau)


def _path_description(config):
    if config["pair"].get("training_mode") == "analytic_brownian_bridge_exp_mass":
        return dict(path_family="wfr_coupling_brownian_bridge_exp_mass",
                    method="wfr_fm_brownian_bridge_exp_mass",
                    mean="linear interpolation between paired endpoints",
                    mass="m_t=P*exp(t*log(Q/P)); log mass is linear",
                    variance="sigma^2*t*(1-t) Brownian bridge",
                    description="WFR coupling + Brownian-bridge conditional path with log-linear mass")
    return dict(path_family="wfr_fm_analytic_bridge", method="wfr_fm_bridge",
                mean="traveling WFR geodesic of two Dirac masses",
                mass="quadratic WFR geodesic mass with m0=1 and m1=Q/P",
                variance="sigma^2*t*(1-t) Brownian bridge heuristic",
                description="analytic WFR mean/mass and Brownian-bridge variance")


def _path_for_config(config):
    if config["pair"].get("training_mode") == "analytic_brownian_bridge_exp_mass":
        return WFRBrownianBridgePath(config["growth_penalty"], config["model_sigma"])
    return WFRBridgePath(config["growth_penalty"], config["model_sigma"])


def wfr_static_cost(P, Q, kappa, alpha):
    """Dirac WFR semi-coupling cost, including the growth penalty coefficient."""
    return 2 * alpha * (P + Q - 2 * kappa * torch.sqrt(P * Q))


def _kl_divergence(x, reference):
    return (x * (x.clamp_min(torch.finfo(x.dtype).tiny).log() - reference.log()) - x + reference).sum()


def _solve_oet_gpu(a, b, cost, kernel, max_iterations, tolerance):
    """POT's unregularized MM solver for the WFR OET objective, in torch/CUDA."""
    # For reg_m=[1,1], reg=0 and KL divergence, mm_unbalanced's MM update is
    # G <- sqrt(a) sqrt(b) exp(-C/2) G / (sqrt(G1) sqrt(G^T1)).
    G = a[:, None] * b[None, :]
    K = a.sqrt()[:, None] * b.sqrt()[None, :] * kernel
    history = []
    converged = False
    error = float("inf")
    for iteration in range(1, max_iterations + 1):
        row, col = G.sum(1, keepdim=True), G.sum(0, keepdim=True)
        next_G = K * G / (row.sqrt() * col.sqrt() + 1e-16)
        error_tensor = torch.linalg.vector_norm(next_G - G)
        G = next_G
        if iteration == 1 or iteration % 50 == 0:
            error = float(error_tensor.item())
            history.append(dict(iteration=iteration, absolute_plan_change=error))
        if float(error_tensor.item()) < tolerance:
            error = float(error_tensor.item())
            converged = True
            if not history or history[-1]["iteration"] != iteration:
                history.append(dict(iteration=iteration, absolute_plan_change=error))
            break
    return G, history, converged


def _ensure_path_checkpoint(run, config):
    checkpoint = run / "pair_path.pt"
    description = _path_description(config)
    if checkpoint.exists():
        existing = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if existing.get("path_family") != description["path_family"] or existing.get("alpha") != config["growth_penalty"] or existing.get("sigma") != config["model_sigma"]:
            raise RuntimeError("existing analytic path checkpoint does not match this WFR configuration")
        return checkpoint
    record = dict(path_family=description["path_family"], alpha=float(config["growth_penalty"]),
                  delta=float(math.sqrt(config["growth_penalty"])), sigma=float(config["model_sigma"]),
                  method=description["description"], mean=description["mean"],
                  mass=description["mass"], variance=description["variance"],
                  static_cost="2*alpha*(P+Q-2*kappa*sqrt(P*Q))",
                  kappa="cos(min(||x-y||/(2*sqrt(alpha)), pi/2))")
    temp = checkpoint.with_suffix(".tmp")
    torch.save(record, temp)
    temp.replace(checkpoint)
    return checkpoint


def solve_wfr_couplings(config, run):
    device = configure(config)
    run = init_run(run, config)
    snapshots = snapshots_for(config)
    cfg = config["coupling"]
    run_path = _ensure_path_checkpoint(run, config)
    path_hash = file_hash(run_path)
    output = run / "couplings"
    output.mkdir(exist_ok=True)
    old_manifest = output / "manifest.json"
    current_status_path = run / "status.json"
    if old_manifest.exists() and current_status_path.exists():
        old_status = json.loads(current_status_path.read_text(encoding="utf-8"))
        old = json.loads(old_manifest.read_text(encoding="utf-8"))
        if (old_status.get("couplings", {}).get("status") == "completed"
                and old.get("status") in {"converged", "audited"}
                and old.get("path_checkpoint_sha256") == path_hash
                and old.get("alpha") == config["growth_penalty"]
                and old.get("mode") == cfg["mode"]
                and old.get("partition_order") == "dataset_default"
                and old.get("minibatch_blocks") == cfg.get("minibatch_blocks")
                and all((output / item["file"]).exists() for item in old.get("intervals", []))
                and len(old.get("intervals", [])) == (len(snapshots) - 1) *
                    (cfg.get("minibatch_blocks", 1) if cfg["mode"] == "minibatch" else 1)):
            update_stage(run, "pair", "completed", method=_path_description(config)["description"], training_steps=0)
            print("Reusing completed WFR OET couplings", flush=True)
            return old
    update_stage(run, "pair", "completed", method=_path_description(config)["description"], training_steps=0)
    update_stage(run, "couplings", "running", method="WFR-FM optimal entropy transport",
                 device=str(device), completed_intervals=0)

    intervals, total_seconds = [], 0.0
    alpha = float(config["growth_penalty"])
    delta_wfr = math.sqrt(alpha)
    partition_rng = np.random.default_rng(cfg["partition_seed"])
    for k, (source, target) in enumerate(zip(snapshots[:-1], snapshots[1:])):
        source_blocks, target_blocks = partition_indices(len(source.x), len(target.x), cfg, partition_rng)
        for block, (source_ids, target_ids) in enumerate(zip(source_blocks, target_blocks)):
            started = time.perf_counter()
            source_x, target_x = source.x[source_ids], target.x[target_ids]
            source_weights, target_weights = source.weights[source_ids], target.weights[target_ids]
            x0 = torch.as_tensor(source_x, dtype=torch.float64, device=device)
            x1 = torch.as_tensor(target_x, dtype=torch.float64, device=device)
            a = torch.as_tensor(source_weights, dtype=torch.float64, device=device)
            b = torch.as_tensor(target_weights, dtype=torch.float64, device=device)
            distances = torch.cdist(x0, x1)
            angles = (distances / (2 * delta_wfr)).clamp(max=math.pi / 2)
            kappa = angles.cos()
            cos_sq = kappa.square()
            # Same numerical cutoff used in the WFR-FM reference implementation.
            cost = -torch.log(cos_sq.clamp_min(1e-10))
            kernel = torch.exp(-0.5 * cost)
            gamma, history, oet_converged = _solve_oet_gpu(a, b, cost, kernel,
                                                           cfg["max_mm_iterations"], cfg["mm_stop_tolerance"])
            row, col = gamma.sum(1, keepdim=True), gamma.sum(0, keepdim=True)
            if bool(torch.any(row <= 0) | torch.any(col <= 0)):
                raise RuntimeError(f"WFR OET coupling has an empty endpoint marginal at interval {k}, block {block}")
            # WFR-FM Theorem 3.1: turn its single OET coupling into semi-couplings.
            P = a[:, None] * gamma / row
            Q = gamma * (b[None, :] / col)
            # On the positive support, Q/P = b_j * row_i / (a_i * col_j).
            # Compute this algebraically instead of dividing the two plans:
            # underflow can make P_ij=Q_ij=0 off support, where Q/P is 0/0
            # although that entry has zero sampling probability.
            ratio = (b[None, :] * row) / (a[:, None] * col)
            if not bool(torch.isfinite(P).all() & torch.isfinite(Q).all() & torch.isfinite(ratio).all()):
                raise FloatingPointError(f"nonfinite WFR semi-coupling at interval {k}, block {block}")

            row_residual = (P.sum(1) - a).abs().sum() / a.sum()
            col_residual = (Q.sum(0) - b).abs().sum() / b.sum()
            semi_cost = wfr_static_cost(P, Q, kappa, alpha).sum()
            oet_inner = (gamma * cost).sum() + _kl_divergence(row[:, 0], a) + _kl_divergence(col[0, :], b)
            oet_cost = 2 * alpha * oet_inner
            oet_semicoupling_gap = (semi_cost - oet_cost).abs() / semi_cost.abs().clamp_min(1e-12)
            gap_value = float(oet_semicoupling_gap.item())
            if gap_value > cfg.get("max_oet_semicoupling_relative_gap", 1e-3):
                outcome = "converged" if oet_converged else "iteration cap reached"
                raise RuntimeError(f"WFR OET {outcome} at interval {k}, block {block}: semi-coupling/OET relative cost gap "
                                   f"{gap_value:.3g} exceeds the configured audit tolerance")
            if max(float(row_residual.item()), float(col_residual.item())) > cfg["marginal_tolerance"]:
                raise RuntimeError(f"WFR semi-coupling marginal audit failed at interval {k}, block {block}")

            name = f"interval_{k}" if cfg["mode"] == "full" else f"interval_{k}_block_{block}"
            arrays = [P, Q, ratio, gamma, kappa, cost]
            P_np, Q_np, ratio_np, gamma_np, kappa_np, cost_np = [v.cpu().numpy() for v in arrays]
            np.savez_compressed(output / f"{name}.npz", P=P_np, Q=Q_np, r=ratio_np, oet_gamma=gamma_np,
                                kappa=kappa_np, oet_cost=cost_np,
                                source_indices=source_ids, target_indices=target_ids,
                                source_weights=source_weights, target_weights=target_weights,
                                source_positions=source_x, target_positions=target_x, interval=k, partition=block)
            status_text = "converged" if oet_converged else "accepted_max_iterations"
            record = dict(file=f"{name}.npz", interval=k, block=block, status=status_text,
                          oet_fixed_point_converged=oet_converged, solver="POT mm_unbalanced update (GPU torch)",
                          mm_iterations=history[-1]["iteration"], final_plan_change=history[-1]["absolute_plan_change"],
                          row_relative_l1=float(row_residual.item()), column_relative_l1=float(col_residual.item()),
                          row_relative_tv=float(row_residual.item()) / 2, column_relative_tv=float(col_residual.item()) / 2,
                          wfr_semicoupling_cost=float(semi_cost.item()), wfr_oet_cost=float(oet_cost.item()),
                          oet_semicoupling_relative_gap=gap_value,
                          ratio_min=float(ratio.min().item()), ratio_max=float(ratio.max().item()),
                          seconds=time.perf_counter() - started, history=history)
            json_write(output / f"{name}.json", record)
            intervals.append({key: value for key, value in record.items() if key != "history"})
            total_seconds += record["seconds"]
            print(f"WFR coupling interval {k}, block {block + 1}/{len(source_blocks)}: "
                  f"iterations={record['mm_iterations']}, static cost={record['wfr_semicoupling_cost']:.6g}, "
                  f"semi gap={record['oet_semicoupling_relative_gap']:.3g}, "
                  f"marginals=({record['row_relative_l1']:.2g},{record['column_relative_l1']:.2g})", flush=True)

    manifest = dict(status="audited", method="wfr_fm_oet", mode=cfg["mode"], device=str(device),
                    partition_order="dataset_default", minibatch_blocks=cfg.get("minibatch_blocks"),
                    partition_seed=cfg["partition_seed"],
                    alpha=alpha, wfr_delta=delta_wfr, psi="g^2/2",
                    oet_marginal_kl_coefficients=[1.0, 1.0], joint_entropy_regularization=0.0,
                    numerical_cost_floor=1e-10, path_checkpoint_sha256=path_hash, intervals=intervals,
                    seconds=total_seconds,
                    static_cost_formula="2*alpha*sum(P+Q-2*kappa*sqrt(P*Q))",
                    oet_formula="2*alpha*(< -2*log(kappa), gamma > + KL(gamma_0|a) + KL(gamma_1|b))")
    json_write(output / "manifest.json", manifest)
    update_stage(run, "couplings", "completed", method="WFR-FM optimal entropy transport",
                 mode=cfg["mode"], backend="torch", device=str(device), seconds=total_seconds,
                 max_row_relative_tv=max(x["row_relative_tv"] for x in intervals),
                 max_column_relative_tv=max(x["column_relative_tv"] for x in intervals))
    return manifest


class WFRCouplingSampler:
    def __init__(self, run, config, device):
        self.run, self.config, self.device = Path(run), config, device
        status = json.loads((self.run / "status.json").read_text(encoding="utf-8"))
        manifest_path = self.run / "couplings" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if status.get("couplings", {}).get("status") != "completed" or manifest.get("status") not in {"converged", "audited"}:
            raise RuntimeError("WFR Matching requires converged WFR-FM couplings")
        if manifest.get("path_checkpoint_sha256") != file_hash(self.run / "pair_path.pt"):
            raise RuntimeError("WFR coupling was built for a different analytic path")
        snapshots = snapshots_for(config)
        self.masses = torch.tensor([s.mass for s in snapshots[:-1]], dtype=torch.float64, device=device)
        interval_data = []
        for k in range(len(snapshots) - 1):
            parts = []
            for item in manifest["intervals"]:
                if item["interval"] != k:
                    continue
                if item.get("status") not in {"converged", "accepted_max_iterations"}:
                    raise RuntimeError(f"unusable WFR coupling status in interval {k}")
                with np.load(self.run / "couplings" / item["file"]) as saved:
                    P, Q, ratio = saved["P"], saved["Q"], saved["r"]
                    row_err = np.abs(P.sum(1) - saved["source_weights"]).sum() / saved["source_weights"].sum()
                    col_err = np.abs(Q.sum(0) - saved["target_weights"]).sum() / saved["target_weights"].sum()
                    if row_err >= config["coupling"]["marginal_tolerance"] or col_err >= config["coupling"]["marginal_tolerance"]:
                        raise RuntimeError(f"cached WFR marginal audit failed for interval {k}")
                    keep = (P > 0) & (Q > 0) & np.isfinite(ratio) & (ratio > 0)
                    rows, cols = np.nonzero(keep)
                    x0 = saved["source_positions"][rows]
                    delta = saved["target_positions"][cols] - x0
                    parts.append(dict(x0=torch.as_tensor(x0, dtype=torch.float64, device=device),
                                      delta=torch.as_tensor(delta, dtype=torch.float64, device=device),
                                      log_r=torch.as_tensor(np.log(ratio[rows, cols]), dtype=torch.float64,
                                                            device=device).reshape(-1, 1),
                                      P=torch.as_tensor(P[rows, cols], dtype=torch.float64, device=device)))
            if not parts:
                raise RuntimeError(f"no WFR coupling blocks for interval {k}")
            interval_data.append({key: torch.cat([part[key] for part in parts])
                                  for key in ("x0", "delta", "log_r", "P")})
        # Select intervals uniformly, then pairs according to P conditioned on
        # that interval. This matches RUOT's sampling law when snapshot masses vary.
        n_intervals = len(interval_data)
        self.flat = {key: torch.cat([v[key] for v in interval_data]) for key in ("x0", "delta", "log_r")}
        probability = torch.cat([v["P"] / self.masses[k] / n_intervals
                                 for k, v in enumerate(interval_data)])
        self.probability = probability
        self.cdf = self.probability.cumsum(0)
        self.cdf[-1] = 1
        self.interval_ids = torch.cat([torch.full((len(v["P"]),), k, dtype=torch.long, device=device)
                                       for k, v in enumerate(interval_data)])

    def sample(self, n, generator=None):
        uniforms = torch.rand(n, dtype=torch.float64, device=self.device, generator=generator)
        ids = torch.searchsorted(self.cdf, uniforms, right=True).clamp_max(len(self.cdf) - 1)
        k = self.interval_ids[ids]
        return dict(x0=self.flat["x0"][ids], delta=self.flat["delta"][ids],
                    log_r=self.flat["log_r"][ids], interval=k,
                    interval_factor=len(self.masses) * self.masses[k])


@torch.no_grad()
def sample_wfr_training_batch(sampler, path, n, device, generator=None,
                              field_parameterization="raw",
                              absorb_sigma_in_singular_velocity=False):
    batch = sampler.sample(n, generator)
    eps = torch.finfo(torch.float64).eps
    t = eps / 2 + (1 - eps) * torch.rand((n, 1), dtype=torch.float64, device=device, generator=generator)
    noise = torch.randn(batch["delta"].shape, dtype=torch.float64, device=device, generator=generator)
    targets = path.conditional_samples(t, batch["x0"], batch["delta"], batch["log_r"], noise)
    if field_parameterization == "endpoint_regularized":
        if absorb_sigma_in_singular_velocity:
            targets["target_v_singular_scaled"] = targets["target_v_singular_scaled"] / path.sigma
        targets["sqrt_wf"] = torch.ones_like(targets["sqrt_wf"])
        targets["sqrt_ws"] = torch.ones_like(targets["sqrt_ws"])
    else:
        targets.pop("target_v_regular")
        targets.pop("target_v_singular_scaled")
    targets = {key: value.float() for key, value in targets.items()}
    global_t = (batch["interval"].reshape(-1, 1) + t).float()
    return global_t, targets, batch["interval_factor"].float()


def train_wfr_matching(config, run):
    device = configure(config)
    run = init_run(run, config)
    sampler = WFRCouplingSampler(run, config, device)
    path = _path_for_config(config)
    path_description = _path_description(config)
    model = make_feedback(config, device)
    cfg = config["matching"]
    objective_version = matching_version(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    checkpoint_file = run / "feedback.pt"
    first_step, prior_seconds = 0, 0.0
    if checkpoint_file.exists():
        cp = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if cp.get("matching_version", objective_version) != objective_version:
            raise RuntimeError("feedback checkpoint uses a different Matching objective")
        if cp.get("method") != path_description["method"] or cp.get("path_checkpoint_sha256") != file_hash(run / "pair_path.pt"):
            raise RuntimeError("existing feedback checkpoint does not belong to this WFR bridge run")
        validate_feedback_architecture(cp, config)
        if cp["coupling_manifest_sha256"] != file_hash(run / "couplings" / "manifest.json"):
            raise RuntimeError("existing feedback checkpoint belongs to another WFR coupling")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        first_step, prior_seconds = cp["completed_steps"], cp["seconds"]
        torch.set_rng_state(cp["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in cp["cuda_rng"]])
    path_hash = file_hash(run / "pair_path.pt")
    coupling_hash = file_hash(run / "couplings" / "manifest.json")
    update_stage(run, "matching", "running", method=path_description["description"], completed_steps=first_step)
    started, last_print = time.perf_counter(), time.perf_counter()
    for step in range(first_step, cfg["steps"]):
        global_t, targets, factor = sample_wfr_training_batch(
            sampler, path, cfg["batch_size"], device,
            field_parameterization=cfg.get("field_parameterization", "raw"),
            absorb_sigma_in_singular_velocity=cfg.get("absorb_sigma_in_singular_velocity", False))
        for group in optimizer.param_groups:
            group["lr"] = matching_learning_rate(cfg, step)
        loss, parts = matching_loss(model.matching_fields(global_t, targets["x"]), targets, factor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=False)
        log_step = (step + 1) % cfg["log_every"] == 0 or step == first_step or step + 1 == cfg["steps"]
        if log_step or (step + 1) % 1000 == 0:
            if not torch.isfinite(torch.stack((loss, grad))).all().item():
                raise FloatingPointError("nonfinite WFR Matching loss/gradient")
        optimizer.step()
        if log_step:
            log_json(run / "matching_history.jsonl", dict(
                step=step + 1, loss=loss.item(), **{k: v.item() for k, v in parts.items()},
                grad_norm=grad.item(), learning_rate=optimizer.param_groups[0]["lr"],
                matching_version=objective_version, method=path_description["method"],
                seconds=prior_seconds + time.perf_counter() - started))
        if time.perf_counter() - last_print > 20 or step + 1 == cfg["steps"]:
            print(f"WFR Matching {step + 1}/{cfg['steps']}: loss={loss.item():.5g}", flush=True)
            last_print = time.perf_counter()
        if (step + 1) % 1000 == 0 or step + 1 == cfg["steps"]:
            temp = checkpoint_file.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), completed_steps=step + 1,
                            matching_version=objective_version, method=path_description["method"],
                            feedback_architecture=feedback_architecture(config),
                            seconds=prior_seconds + time.perf_counter() - started,
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                            path_checkpoint_sha256=path_hash,
                            coupling_manifest_sha256=coupling_hash, config=config), temp)
            temp.replace(checkpoint_file)
            update_stage(run, "matching", "running", method=path_description["description"], completed_steps=step + 1)
    update_stage(run, "matching", "completed", method=path_description["description"],
                 completed_steps=cfg["steps"], seconds=prior_seconds + time.perf_counter() - started)
    return model


def compare_to_ruot(config, run):
    run = Path(run)
    baseline = Path(config.get("comparison_run", "../Results/gene_steps_gpu_4000_seed42"))
    if not baseline.is_absolute():
        baseline = Path(__file__).resolve().parents[1] / baseline
    if not (baseline / "metrics.csv").exists():
        return None
    current, reference = pd.read_csv(run / "metrics.csv"), pd.read_csv(baseline / "metrics.csv")
    columns = ["W1_normalized", "W2_normalized", "mass_pred", "mass_true", "mass_relative_error"]
    merged = current[["time", *columns]].merge(reference[["time", *columns]], on="time", suffixes=("_wfr", "_ruot"))
    merged.to_csv(run / "comparison_to_ruot.csv", index=False)
    return merged


def run_wfr_heuristic(config, run):
    run = Path(run)
    solve_wfr_couplings(config, run)
    train_wfr_matching(config, run)
    from .experiment import generate_and_evaluate
    metrics = generate_and_evaluate(config, run)
    comparison = compare_to_ruot(config, run)
    if comparison is not None:
        print("WFR vs RUOT metrics saved to comparison_to_ruot.csv", flush=True)
    from .action import compute_action
    compute_action(config, run)
    from .experiment import write_report
    write_report(run, config)
    return metrics

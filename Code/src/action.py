"""Post-training running-action estimate along the saved weighted rollout."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .matching import make_feedback, validate_feedback_architecture
from .runtime import configure, file_hash, init_run, json_write, snapshots_for
from .sample import rollout


@torch.no_grad()
def compute_action(config, run):
    device = configure(config)
    run = init_run(run, config)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("matching", {}).get("status") != "completed":
        raise RuntimeError("completed Matching is required to compute its action")
    checkpoint_file = run / "feedback.pt"
    coupling_file = run / "couplings" / "manifest.json"
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    validate_feedback_architecture(checkpoint, config)
    if (checkpoint["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt")
            or checkpoint["coupling_manifest_sha256"] != file_hash(coupling_file)):
        raise RuntimeError("feedback checkpoint is stale relative to the path or Couplings")

    model = make_feedback(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    truth = snapshots_for(config, full=True)
    generation = config["generation"]
    trajectory = rollout(
        model, truth[0].x, config["model_times"], config["model_sigma"],
        generation["n_particles"], generation["steps_per_unit_interval"],
        config["seed"] + 300, generation["endpoint_guard"],
        action_alpha=config["growth_penalty"],
    )
    with np.load(run / "generated.npz") as saved:
        initial_x = saved["positions"][0].copy()
        position_error = float(np.max(np.abs(trajectory["positions"] - saved["positions"])))
        log_weight_error = float(np.max(np.abs(trajectory["log_weights"] - saved["log_weights"])))

    action = trajectory["action"]
    # Couple 200- and 400-step action estimates with the same GPU Brownian path
    # to isolate most of the time-discretization sensitivity.
    steps = generation["steps_per_unit_interval"]
    fine_steps = 2 * steps
    n_particles = generation["n_particles"]
    dimension = config["dimension"]
    dtype = next(model.parameters()).dtype
    rollout_generator = torch.Generator(device=device).manual_seed(config["seed"] + 300)
    # Match the initial-index draw made by the saved rollout so the generated
    # coarse Brownian increments reproduce its exact random stream.
    torch.randint(len(truth[0].x), (n_particles,), device=device, generator=rollout_generator)
    coarse_noise = torch.stack([
        torch.randn((n_particles, dimension), device=device, dtype=dtype,
                    generator=rollout_generator) / math.sqrt(steps)
        for _ in range((len(config["model_times"]) - 1) * steps)
    ])
    bridge_generator = torch.Generator(device=device).manual_seed(config["seed"] + 202)
    bridge_noise = torch.randn(coarse_noise.shape, device=device, dtype=dtype,
                               generator=bridge_generator) * math.sqrt(1 / (4 * steps))
    fine_noise = torch.stack((coarse_noise / 2 + bridge_noise,
                              coarse_noise / 2 - bridge_noise), dim=1).reshape(
                                  -1, n_particles, dimension)
    coarse_check = rollout(
        model, truth[0].x, config["model_times"], config["model_sigma"], n_particles,
        steps, config["seed"] + 202, generation["endpoint_guard"], initial_x=initial_x,
        normal_increments=coarse_noise, action_alpha=config["growth_penalty"],
    )["action"]
    fine_check = rollout(
        model, truth[0].x, config["model_times"], config["model_sigma"], n_particles,
        fine_steps, config["seed"] + 202, generation["endpoint_guard"], initial_x=initial_x,
        normal_increments=fine_noise, action_alpha=config["growth_penalty"],
    )["action"]
    coarse_reproduction_error = abs(coarse_check["total"] - action["total"])
    if coarse_reproduction_error > 1e-8:
        raise RuntimeError("replayed coarse action does not match the saved rollout")
    action_discretization = dict(
        common_brownian_increments=True,
        coarse_steps_per_unit=steps,
        fine_steps_per_unit=fine_steps,
        coarse_action=coarse_check["total"],
        fine_action=fine_check["total"],
        coarse_reproduction_abs_error=coarse_reproduction_error,
        relative_difference=abs(fine_check["total"] - coarse_check["total"])
        / max(abs(fine_check["total"]), 1e-12),
    )
    result = dict(
        definition="learned_feedback_running_action",
        formula="integral sum_i w_i(t) * (0.5*|U(t,X_i)|^2 + alpha/2*G(t,X_i)^2) dt",
        drift="U = V + sigma^2/2 * Score",
        alpha=float(config["growth_penalty"]),
        psi="g^2/2",
        sigma=float(config["model_sigma"]),
        quadrature="left-endpoint Euler rectangle on the saved Euler-Maruyama grid",
        steps_per_unit=int(generation["steps_per_unit_interval"]),
        particles=int(generation["n_particles"]),
        seed=int(config["seed"] + 300),
        reproduced_saved_rollout=dict(max_position_abs_error=position_error,
                                      max_log_weight_abs_error=log_weight_error),
        kinetic_action=action["kinetic"],
        growth_action=action["growth"],
        total_action=action["total"],
        particle_mc_standard_error=action["mc_standard_error"],
        kinetic_mc_standard_error=action["kinetic_mc_standard_error"],
        growth_mc_standard_error=action["growth_mc_standard_error"],
        intervals=action["intervals"],
        discretization_check=action_discretization,
    )
    json_write(run / "action.json", result)
    pd.DataFrame(action["intervals"]).to_csv(run / "action_intervals.csv", index=False)
    from .experiment import write_report
    write_report(run, config)
    return result

from pathlib import Path
import json
import math

import numpy as np
import torch
from .data import PairIndex
from .pair_cost import CostOracle, finite_pair_integrand
from .runtime import configure, json_write, load_path, snapshots_for


def audit_run(config, run):
    """Independent saved-artifact checks, plus displacement/ratio-stratified queries."""
    run = Path(run)
    device = configure(config)
    snapshots = snapshots_for(config)
    path = load_path(run, config, device)
    index = PairIndex(snapshots, device)
    rng = np.random.default_rng(config["seed"] + 401)
    n = 512
    ids = rng.integers(index.size, size=n)
    delta_tensor = index.delta(ids)
    delta = delta_tensor.cpu().numpy()
    ratio = np.exp(rng.uniform(math.log(config["pair"]["ratio_min"]), math.log(config["pair"]["ratio_max"]), n))
    cp = config["pair"]
    nodes = cp["quadrature_nodes"]
    quadrature_file = run / "quadrature_validation.json"
    if quadrature_file.exists():
        nodes = json.loads(quadrature_file.read_text(encoding="utf-8")).get("selected_nodes", nodes)
    coarse, fine = CostOracle(path, nodes), CostOracle(path, 2 * nodes)
    a, b = coarse(delta, ratio), fine(delta, ratio)
    length = np.linalg.norm(delta, axis=1)
    boundaries = np.quantile(length, [0, 0.25, 0.5, 0.75, 1])
    stratified = []
    for i in range(4):
        selected = (length >= boundaries[i]) & (length <= boundaries[i + 1] if i == 3 else length < boundaries[i + 1])
        stratified.append(dict(axis="displacement_length", lower=float(boundaries[i]), upper=float(boundaries[i+1]),
                               count=int(selected.sum()), cost_quadrature_max_error=float(np.abs(a[0][selected] - b[0][selected]).max()),
                               min_direct_curvature=float(b[2][selected].min()), mean_cost=float(b[0][selected].mean())))
    for lower, upper in zip([0.1, 0.3, 1, 3], [0.3, 1, 3, 10]):
        selected = (ratio >= lower) & (ratio < upper)
        if selected.any():
            stratified.append(dict(axis="mass_ratio", lower=lower, upper=upper, count=int(selected.sum()),
                                   cost_quadrature_max_error=float(np.abs(a[0][selected] - b[0][selected]).max()),
                                   min_direct_curvature=float(b[2][selected].min()), mean_cost=float(b[0][selected].mean())))
    near_times = torch.tensor([0, 1e-12, 1 - 1e-12, 1], dtype=torch.float64, device=device)
    t = near_times.repeat(n).reshape(-1, 1)
    ell = torch.as_tensor(np.log(ratio), dtype=torch.float64, device=device).reshape(-1, 1).repeat_interleave(4, 0)
    s = path(t, delta_tensor.repeat_interleave(4, 0), ell)
    endpoint = finite_pair_integrand(s, path.sigma, path.alpha, path.control_mode)
    if not torch.isfinite(endpoint).all():
        raise AssertionError("nonfinite endpoint audit")
    # Independent centered differences in r on interior validation conditions.
    selected = np.flatnonzero((ratio > 0.11) & (ratio < 9.9))[:32]
    h = 1e-4 * ratio[selected]
    minus, plus = fine(delta[selected], ratio[selected] - h, 0), fine(delta[selected], ratio[selected] + h, 0)
    derivative_error = np.abs((plus - minus) / (2 * h) - b[1][selected])
    report = dict(stratified=stratified, sampled_conditions=n, quadrature_nodes=nodes,
                  endpoint_checks=n * 4, endpoint_finite=True, min_direct_curvature=float(b[2].min()),
                  derivative_finite_difference_max_error=float(derivative_error.max()))
    manifest_file = run / "couplings" / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest["status"] != "converged":
            raise AssertionError("coupling manifest not converged")
        rows, cols, pair_count, block_ratio_min, block_ratio_max = [], [], 0, 10.0, 0.1
        for item in manifest["intervals"]:
            with np.load(run / "couplings" / item["file"]) as data:
                P, Q, r = data["P"], data["Q"], data["r"]
                row_error = np.abs(P.sum(1) - data["source_weights"]).sum() / data["source_weights"].sum()
                col_error = np.abs(Q.sum(0) - data["target_weights"]).sum() / data["target_weights"].sum()
                if max(row_error, col_error) >= config["coupling"]["marginal_tolerance"]:
                    raise AssertionError("saved coupling residual too large")
                if not np.all((r >= cp["ratio_min"]) & (r <= cp["ratio_max"])):
                    raise AssertionError("ratio outside training domain")
                np.testing.assert_allclose(Q, P * r, rtol=1e-10, atol=1e-15)
                rows.append(float(row_error))
                cols.append(float(col_error))
                pair_count += P.size
                block_ratio_min, block_ratio_max = min(block_ratio_min, r.min()), max(block_ratio_max, r.max())
        # Each observed point must occur in exactly one block, preserving raw mass.
        for k in range(len(snapshots) - 1):
            sources, targets = [], []
            for item in manifest["intervals"]:
                if item["interval"] == k:
                    with np.load(run / "couplings" / item["file"]) as data:
                        sources.extend(data["source_indices"].tolist())
                        targets.extend(data["target_indices"].tolist())
            assert sorted(sources) == list(range(len(snapshots[k].x)))
            assert sorted(targets) == list(range(len(snapshots[k+1].x)))
        report["couplings"] = dict(blocks=len(manifest["intervals"]), support_pairs=pair_count,
                                    max_row_residual=max(rows), max_column_residual=max(cols),
                                    ratio_min=float(block_ratio_min), ratio_max=float(block_ratio_max),
                                    every_observation_used_once_per_interval=True)
    generated_file = run / "generated.npz"
    metrics_file = run / "metrics.csv"
    if generated_file.exists() and metrics_file.exists():
        import pandas as pd
        from scipy.special import logsumexp
        with np.load(generated_file) as data:
            assert data["positions"].shape == (len(config["model_times"]), config["generation"]["n_particles"], 2)
            logw = data["log_weights"]
            assert logw.shape == data["positions"].shape[:2]
            assert np.isfinite(logw).all() and np.isfinite(data["positions"]).all()
            mass = np.exp(logsumexp(logw, axis=1))
            metrics = pd.read_csv(metrics_file)
            np.testing.assert_allclose(mass, metrics.mass_pred, rtol=1e-12)
            np.testing.assert_allclose(metrics.mass_true, [s.mass for s in snapshots], rtol=1e-12)
        report["generated"] = dict(fixed_particles=True, saved_raw_mass_matches_metrics=True, full_observed_masses=True)
    json_write(run / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report

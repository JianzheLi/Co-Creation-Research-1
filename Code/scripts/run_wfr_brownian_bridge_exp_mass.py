"""Add a WFR-coupling / Brownian-bridge path baseline to Experiment 1."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment1_sigma_sweep import (  # noqa: E402
    DATASETS, SIGMAS, OUT, check_completed, output_root, prepare, sigma_tag,
)
from evaluate_matching_stability import checked_checkpoint, simulate_batch  # noqa: E402
from src.matching import make_feedback, validate_feedback_architecture  # noqa: E402
from src.runtime import configure, file_hash, json_write, load_config, snapshots_for, update_stage  # noqa: E402


METHOD = "WFR coupling + Brownian bridge (log-linear mass)"


def prepare_run(dataset: str, sigma: float):
    _, wfr_config_path, _, wfr_run = prepare(dataset, sigma)
    wfr_config = load_config(wfr_config_path)
    config = dict(wfr_config)
    config["experiment_method"] = "wfr_fm_bridge_exp_mass"
    config["pair"] = dict(wfr_config["pair"],
                           training_mode="analytic_brownian_bridge_exp_mass",
                           training_steps=0)
    tag = sigma_tag(sigma)
    stem = f"experiment1_{dataset}_alpha2_sigma{tag}_wfr_brownian_bridge_exp_mass"
    config_path = ROOT / "configs" / f"{stem}.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    # Re-load through the same validation and normalization used by all runs.
    config = load_config(config_path)
    run = output_root(dataset) / dataset / f"sigma_{tag}" / "wfr_brownian_bridge_exp_mass_seed42"
    return config_path, config, run, wfr_run


def reuse_exact_wfr_coupling(config, run: Path, source_run: Path):
    """Reuse the already-solved WFR coupling so only the conditional path changes."""
    if not check_completed(source_run, source_run / "config.yaml"):
        raise RuntimeError(f"the reference WFR coupling run is incomplete: {source_run}")
    run.mkdir(parents=True, exist_ok=True)
    from src.runtime import init_run
    from src.wfr_heuristic import _ensure_path_checkpoint

    init_run(run, config)
    new_path_checkpoint = _ensure_path_checkpoint(run, config)
    source_couplings = source_run / "couplings"
    destination_couplings = run / "couplings"
    if not destination_couplings.exists():
        shutil.copytree(source_couplings, destination_couplings)
    manifest_path = destination_couplings / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads((source_couplings / "manifest.json").read_text(encoding="utf-8"))
    for key in ("method", "mode", "partition_order", "partition_seed", "alpha", "psi",
                "oet_marginal_kl_coefficients", "joint_entropy_regularization", "intervals"):
        if manifest.get(key) != source_manifest.get(key):
            raise ValueError(f"copied WFR coupling differs from source at {key}")
    manifest["path_checkpoint_sha256"] = file_hash(new_path_checkpoint)
    manifest["reused_exactly_from"] = str(source_run.resolve())
    json_write(manifest_path, manifest)
    update_stage(run, "pair", "completed", method="WFR coupling + Brownian bridge conditional path",
                 training_steps=0)
    update_stage(run, "couplings", "completed", method="reused exact WFR-FM optimal entropy transport",
                 mode=manifest["mode"], backend="torch", device="cuda", seconds=0.0,
                 max_row_relative_tv=max(x["row_relative_tv"] for x in manifest["intervals"]),
                 max_column_relative_tv=max(x["column_relative_tv"] for x in manifest["intervals"]))
    # Audit that endpoint support, semi-coupling masses, and conditional mass ratios are byte-identical.
    for item in source_manifest["intervals"]:
        with np.load(source_couplings / item["file"]) as expected, np.load(destination_couplings / item["file"]) as actual:
            for key in ("P", "Q", "r", "source_indices", "target_indices", "source_positions",
                        "target_positions", "source_weights", "target_weights"):
                np.testing.assert_array_equal(actual[key], expected[key])


def _stat(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                median=float(np.median(x)), finite_seeds=int(np.isfinite(x).sum()))


def _paired_difference(left, right):
    left_by_seed = {int(row["seed"]): float(row["total_action"]) for row in left}
    right_by_seed = {int(row["seed"]): float(row["total_action"]) for row in right}
    seeds = sorted(set(left_by_seed) & set(right_by_seed))
    diff = np.asarray([left_by_seed[seed] - right_by_seed[seed] for seed in seeds], dtype=np.float64)
    return dict(paired_finite_seeds=int(len(diff)), mean=float(diff.mean()) if len(diff) else None,
                standard_error=float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else None,
                left_lower=int((diff < 0).sum()), tied=int((diff == 0).sum()),
                right_lower=int((diff > 0).sum()))


def evaluate_and_merge(dataset: str, sigma: float, baseline_run: Path, run: Path):
    comparison_path = baseline_run.parent / "comparison_20seed.json"
    if not comparison_path.exists():
        raise FileNotFoundError(f"the existing RUOT/WFR comparison is missing: {comparison_path}")
    existing = json.loads(comparison_path.read_text(encoding="utf-8"))
    baseline_cfg = load_config(baseline_run / "config.yaml")
    cfg = load_config(run / "config.yaml")
    for key in ("dataset", "data_path", "dimension", "model_times", "model_sigma", "growth_penalty",
                "generation", "matching"):
        if baseline_cfg[key] != cfg[key]:
            raise ValueError(f"the Brownian-bridge baseline must share {key}")

    source_wfr_run = baseline_run.parent / "wfr_brownian_bridge_seed42"
    source_manifest = json.loads((source_wfr_run / "couplings" / "manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((run / "couplings" / "manifest.json").read_text(encoding="utf-8"))
    for old, new in zip(source_manifest["intervals"], new_manifest["intervals"]):
        with np.load(source_wfr_run / "couplings" / old["file"]) as a, np.load(run / "couplings" / new["file"]) as b:
            for key in ("P", "Q", "r", "source_indices", "target_indices", "source_positions",
                        "target_positions", "source_weights", "target_weights"):
                np.testing.assert_array_equal(a[key], b[key])

    device = configure(cfg)
    if device.type != "cuda":
        raise RuntimeError("this experiment requires CUDA")
    truth = snapshots_for(cfg, full=True)
    first = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)
    seeds = list(range(1000, 1020))
    n = int(cfg["generation"]["n_particles"])
    n_steps = (len(cfg["model_times"]) - 1) * int(cfg["generation"]["steps_per_unit_interval"])
    initial, noise = [], []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed + 300)
        ids = torch.randint(len(first), (n,), device=device, generator=generator)
        initial.append(first[ids])
        noise.append(torch.randn((n_steps, n, cfg["dimension"]), device=device,
                                 dtype=torch.float32, generator=generator))
    initial = torch.stack(initial)
    noise = torch.stack(noise)

    checkpoint = checked_checkpoint(run)
    model = make_feedback(cfg, device)
    validate_feedback_architecture(checkpoint, cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = simulate_batch(model, initial, noise, cfg, seeds)
    values = np.asarray([row["total_action"] if row["finite"] else np.nan for row in rows], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise FloatingPointError("the new baseline has non-finite or non-positive rollout Actions")
    with (run / "metrics.csv").open(encoding="utf-8-sig") as f:
        metric_rows = list(csv.DictReader(f))
    method_result = dict(run=str(run.resolve()), feedback_sha256=file_hash(run / "feedback.pt"),
                         finite_seeds=int(np.isfinite(values).sum()), action_mean=float(values.mean()),
                         action_sd=float(values.std(ddof=1)), action_median=float(np.median(values)),
                         W1_mean=float(np.mean([float(row["W1_normalized"]) for row in metric_rows])),
                         W2_mean=float(np.mean([float(row["W2_normalized"]) for row in metric_rows])),
                         mass_relative_error_mean=float(np.mean([float(row["mass_relative_error"]) for row in metric_rows])),
                         replicates=rows)
    del model
    torch.cuda.empty_cache()

    methods = existing["methods"]
    methods[METHOD] = method_result
    ruot_rows = methods["RUOT"]["replicates"]
    wfr_rows = methods["WFR+Brownian bridge"]["replicates"]
    differences = {
        "WFR+Brownian bridge minus RUOT": _paired_difference(wfr_rows, ruot_rows),
        f"{METHOD} minus RUOT": _paired_difference(rows, ruot_rows),
        f"{METHOD} minus WFR+Brownian bridge": _paired_difference(rows, wfr_rows),
    }
    existing["methods"] = methods
    existing["paired_action_differences"] = differences
    existing["settings"]["third_baseline"] = (
        "Exact same WFR semi-couplings as WFR+Brownian bridge; linear conditional mean, "
        "sigma^2*t*(1-t) Brownian-bridge variance, and m_t=P*exp(t*log(Q/P)) log-linear mass"
    )
    existing["settings"]["common_rollout_seeds"] = seeds
    json_write(comparison_path, existing)
    json_write(run.parent / "wfr_brownian_bridge_exp_mass_20seed.json",
               dict(dataset=dataset, alpha=2.0, sigma=float(sigma), method=METHOD,
                    action_definition=existing["action_definition"], results=method_result,
                    paired_action_differences=differences))

    combined_rows = []
    for method_name, value in methods.items():
        combined_rows.extend(dict(method=method_name, **row) for row in value["replicates"])
    with (run.parent / "action_replicates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)
    summary = {"dataset": dataset, "sigma": sigma, "method": METHOD,
               "Action_mean": method_result["action_mean"], "Action_sd": method_result["action_sd"],
               "finite_seeds": method_result["finite_seeds"], "paired": differences[f"{METHOD} minus RUOT"]}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run_one(dataset: str, sigma: float):
    config_path, config, run, source_wfr_run = prepare_run(dataset, sigma)
    if not check_completed(run, config_path):
        reuse_exact_wfr_coupling(config, run, source_wfr_run)
        log_path = run.parent / "wfr_brownian_bridge_exp_mass.log"
        command = [sys.executable, str(ROOT / "scripts" / "run_wfr_heuristic.py"),
                   "--config", str(config_path.resolve()), "--run-dir", str(run.resolve())]
        print(f"Starting {dataset} sigma={sigma:g}: {METHOD}", flush=True)
        started = time.perf_counter()
        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode and not check_completed(run, config_path):
            raise RuntimeError(f"baseline failed; inspect {log_path}")
        print(f"Finished training {dataset} sigma={sigma:g} in {time.perf_counter()-started:.1f}s", flush=True)
    return evaluate_and_merge(dataset, sigma, source_wfr_run, run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))
    args = parser.parse_args()
    if not args.all_datasets and args.dataset is None:
        parser.error("specify --dataset or --all-datasets")
    if any(sigma not in SIGMAS for sigma in args.sigmas):
        parser.error(f"sigmas must be selected from {SIGMAS}")
    datasets = DATASETS if args.all_datasets else (args.dataset,)
    results = []
    for dataset in datasets:
        for sigma in args.sigmas:
            results.append(run_one(dataset, sigma))
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

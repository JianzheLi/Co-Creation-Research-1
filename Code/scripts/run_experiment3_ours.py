"""Train and evaluate our method on every Experiment 3 HOO split.

Only non-held-out snapshots are written to each training CSV. The observed
sequence is compressed to consecutive unit intervals, so a withheld internal
timepoint is evaluated at the midpoint of the interval formed by its two
nearest retained snapshots. Each Matching model starts from random
initialization; pair paths, Couplings, and Matching are trained for that split.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import ot
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / "data"
DATA_FILES = {
    "cite": "cite_pca50.csv",
    "eb_noscale": "eb_noscale.csv",
    "weinreb": "Weinreb_alltime.csv",
}
CONFIGS = {
    "cite": ROOT / "configs" / "experiment3_cite_base.yaml",
    "eb_noscale": ROOT / "configs" / "experiment3_eb_noscale_base.yaml",
    "weinreb": ROOT / "configs" / "experiment3_weinreb_base.yaml",
}


def _full_w1(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.maximum(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if weights.sum() <= 0:
        weights = np.full(len(pred), 1 / len(pred), dtype=np.float64)
    else:
        weights /= weights.sum()
    target_weights = np.full(len(truth), 1 / len(truth), dtype=np.float64)
    return float(ot.emd2(weights, target_weights, cdist(pred, truth), numItermax=100_000_000))


def _write_config(dataset: str, holdout: int, train_csv: Path, train_times: int,
                  run_dir: Path, config_dir: Path, *, feedback_activation: str | None = None,
                  network_variant: str | None = None,
                  gradient_clip_norm: float | None = None,
                  lr_schedule: str | None = None,
                  base_config: Path | None = None) -> Path:
    config = yaml.safe_load((base_config or CONFIGS[dataset]).read_text(encoding="utf-8"))
    if config["dataset"] != dataset:
        raise ValueError("The base configuration must match the requested dataset")
    config["data_path"] = str(train_csv.resolve())
    config["model_times"] = list(range(train_times))
    config["seed"] = 42
    config["matching"]["validation_every"] = 0
    config["matching"].pop("warm_start", None)
    if feedback_activation is not None:
        config["matching"]["feedback_activation"] = feedback_activation
        config["matching"]["last_hidden_activation"] = feedback_activation
    if network_variant is not None:
        config["matching"]["network_variant"] = network_variant
        config["matching"]["growth_network_variant"] = network_variant
        config["matching"]["score_network_variant"] = network_variant
    if gradient_clip_norm is not None:
        config["matching"]["gradient_clip_norm"] = gradient_clip_norm
    if lr_schedule is not None:
        config["matching"]["lr_schedule"] = lr_schedule
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    config_file.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_file


def _train_stages(config_file: Path, run_dir: Path):
    for stage in ("pairs", "couplings", "matching"):
        print(json.dumps({"stage": stage, "run_dir": str(run_dir), "status": "starting"}), flush=True)
        log_file = run_dir / "experiment3_logs" / f"{stage}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log_file.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.cli", "--config", str(config_file),
                 "--run-dir", str(run_dir), "--stage", stage],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                time.sleep(30)
                if process.poll() is None:
                    print(json.dumps({
                        "stage": stage, "run_dir": str(run_dir), "status": "running",
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                        "log": str(log_file),
                    }), flush=True)
        if process.returncode:
            tail = log_file.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(
                f"Experiment 3 {stage} stage failed with exit code "
                f"{process.returncode}; see {log_file}\n{tail}"
            )


def _predict(config_file: Path, run_dir: Path, holdout: int, query_time: float,
             output_dir: Path, truth: np.ndarray):
    import torch
    from scipy.special import logsumexp
    from src.matching import make_feedback, validate_feedback_architecture
    from src.runtime import configure, load_config, snapshots_for
    from src.sample import rollout

    config = load_config(config_file)
    device = configure(config)
    model = make_feedback(config, device)
    checkpoint = torch.load(run_dir / "feedback.pt", map_location=device, weights_only=False)
    validate_feedback_architecture(checkpoint, config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    training_snapshots = snapshots_for(config, full=True)
    first = training_snapshots[0].x
    started = time.perf_counter()
    generated = rollout(
        model, first, times=tuple(config["model_times"]), sigma=config["model_sigma"],
        n_particles=len(first), steps_per_unit=config["generation"]["steps_per_unit_interval"],
        seed=342, endpoint_guard=config["generation"]["endpoint_guard"],
        initial_x=first, stochastic=True, capture_times=(query_time,),
    )
    prediction = generated["capture_positions"][0]
    logw = generated["capture_log_weights"][0]
    normalized_weights = np.exp(logw - logsumexp(logw))
    predicted_mass_ratio = float(np.exp(logsumexp(logw)))
    observed_mass_ratio = len(truth) / len(first)
    w1 = _full_w1(prediction, truth, normalized_weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "prediction.npz", positions=prediction.astype(np.float32),
        weights=normalized_weights.astype(np.float32),
        truth_time=np.asarray([holdout], dtype=np.float64),
        compressed_prediction_time=np.asarray([query_time], dtype=np.float64),
    )
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    metadata = {
        "dataset": config["dataset"], "method": "RUOT Matching",
        "heldout_index": holdout, "heldout_time": float(holdout),
        "compressed_prediction_time": query_time, "W1": w1,
        "n_prediction": int(len(prediction)), "n_truth": int(len(truth)),
        "seed": config["seed"], "prediction_seed": 342,
        "architecture": config["matching"]["network_variant"],
        "layernorm_affine": config["matching"].get("network_variant") == "plain_output_affine_norm",
        "feedback_activation": config["matching"].get("feedback_activation"),
        "gradient_clip_norm": config["matching"].get("gradient_clip_norm", 10.0),
        "lr_schedule": config["matching"].get("lr_schedule", "constant"),
        "matching_steps": config["matching"]["steps"],
        "learning_rate": config["matching"]["learning_rate"],
        "coupling_blocks": config["coupling"]["minibatch_blocks"],
        "growth_loss_multiplier": config["matching"].get("growth_loss_multiplier", 1.0),
        "model_sigma": config["model_sigma"], "growth_penalty": config["growth_penalty"],
        "pair_path_status": status.get("pair", {}).get("status"),
        "coupling_status": status.get("couplings", {}).get("status"),
        "matching_status": status.get("matching", {}).get("status"),
        "sde_max_abs_position": float(np.max(np.abs(prediction))),
        "sde_max_abs_log_weight": float(np.max(np.abs(logw))),
        "sde_ess": float(1.0 / np.square(normalized_weights).sum()),
        "predicted_mass_ratio": predicted_mass_ratio,
        "observed_mass_ratio": observed_mass_ratio,
        "mass_relative_error": abs(predicted_mass_ratio - observed_mass_ratio) / observed_mass_ratio,
        "prediction_seconds": time.perf_counter() - started,
        "run_dir": str(run_dir),
    }
    (output_dir / "result.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return metadata


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "../Results" / "experiment3")
    parser.add_argument("--dataset", choices=DATA_FILES)
    parser.add_argument("--holdout", type=int)
    parser.add_argument("--feedback-activation", choices=("silu", "leaky_relu", "tanh"))
    parser.add_argument("--network-variant", choices=("plain_output_norm", "plain_output_affine_norm"))
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine", "cosine_2cycle", "cosine_2cycle_half"))
    parser.add_argument("--base-config", type=Path,
                        help="Use a selected recipe; each holdout still retrains all three stages")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    datasets = [args.dataset] if args.dataset else list(DATA_FILES)
    for dataset in datasets:
        frame = pd.read_csv(DATA_ROOT / DATA_FILES[dataset])
        original_times = np.asarray(sorted(frame["samples"].unique()), dtype=np.float64)
        feature_cols = [c for c in frame.columns if c not in ("samples", "label", "cell_type")]
        holdouts = [args.holdout] if args.holdout is not None else list(range(1, len(original_times) - 1))
        for holdout in holdouts:
            result_dir = root / "ours" / dataset / f"hold_{holdout}"
            result_file = result_dir / "result.json"
            if result_file.exists():
                row = json.loads(result_file.read_text(encoding="utf-8"))
                print(json.dumps({"status": "already_completed", **row}, ensure_ascii=False), flush=True)
                all_rows.append(row)
                continue
            if not 0 < holdout < len(original_times) - 1:
                raise ValueError(f"holdout {holdout} is not an interior timepoint for {dataset}")
            retained_times = [float(t) for i, t in enumerate(original_times) if i != holdout]
            relabel = {t: float(i) for i, t in enumerate(retained_times)}
            train = frame.loc[frame["samples"] != original_times[holdout]].copy()
            train["samples"] = train["samples"].map(relabel)
            split_dir = root / "splits" / dataset / f"hold_{holdout}"
            split_dir.mkdir(parents=True, exist_ok=True)
            train_csv = split_dir / "train_compressed_time.csv"
            train.to_csv(train_csv, index=False)
            query_time = float(holdout) - 0.5
            truth = frame.loc[frame["samples"] == original_times[holdout], feature_cols].to_numpy(dtype=np.float64)
            run_dir = result_dir / "run"
            config_dir = result_dir / "config"
            config_file = _write_config(
                dataset, holdout, train_csv, len(retained_times), run_dir, config_dir,
                feedback_activation=args.feedback_activation,
                network_variant=args.network_variant,
                gradient_clip_norm=args.gradient_clip_norm,
                lr_schedule=args.lr_schedule,
                base_config=args.base_config,
            )
            print(json.dumps({
                "status": "starting", "dataset": dataset, "heldout_index": holdout,
                "heldout_time": float(original_times[holdout]), "compressed_prediction_time": query_time,
                "network_variant": config_file and yaml.safe_load(config_file.read_text(encoding="utf-8"))["matching"]["network_variant"],
                "feedback_activation": args.feedback_activation,
                "from_scratch": True,
            }, ensure_ascii=False), flush=True)
            if not (run_dir / "status.json").exists() or json.loads((run_dir / "status.json").read_text(encoding="utf-8")).get("matching", {}).get("status") != "completed":
                _train_stages(config_file, run_dir)
            row = _predict(config_file, run_dir, int(original_times[holdout]), query_time, result_dir, truth)
            all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(root / "ours_holdout_w1.csv", index=False)


if __name__ == "__main__":
    main()

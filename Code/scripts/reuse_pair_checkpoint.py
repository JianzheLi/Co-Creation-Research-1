"""Create a new experiment from a mathematically compatible frozen path stage."""
from pathlib import Path
import argparse
import json
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure, init_run, json_write, load_config, file_hash


def reuse(source, destination, config_path, include_distillation=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    config, old = load_config(config_path), load_config(source / "config.yaml")
    for key in ("seed", "dimension", "model_times", "model_sigma", "growth_penalty", "control_mode", "max_observed_points", "pair"):
        if config[key] != old[key]:
            raise ValueError(f"incompatible path-stage configuration: {key}")
    if include_distillation and config["distillation"] != old["distillation"]:
        raise ValueError("incompatible distillation configuration")
    if destination.exists():
        raise FileExistsError("select a new directory to preserve previous experiments")
    configure(config)
    destination = init_run(destination, config)
    for name in ("pair_path.pt", "data_metadata.json"):
        shutil.copy2(source / name, destination / name)
    for name in ("pair_history.jsonl", "pair_batches.jsonl", "pair_steps.jsonl", "pair_validation.jsonl"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    status = json.loads((source / "status.json").read_text(encoding="utf-8"))["pair"]
    if status["status"] != "completed":
        raise ValueError("source pair pretraining is not complete")
    status.update(reused_from=str(source), checkpoint_sha256=file_hash(destination / "pair_path.pt"))
    json_write(destination / "status.json", dict(pair=status))
    if include_distillation:
        for name in ("cost_network.pt", "distillation_validation.json", "distillation_grid.pt",
                     "distillation_history.jsonl", "deployment_curvature.json"):
            if (source/name).exists():
                shutil.copy2(source/name, destination/name)
        distillation = json.loads((source / "status.json").read_text(encoding="utf-8"))["distillation"]
        if distillation["status"] != "completed":
            raise ValueError("source distillation is incomplete")
        json_write(destination/"status.json", dict(pair=status, distillation=distillation))
    json_write(destination / "provenance.json", dict(path_stage_reused_from=str(source),
               path_checkpoint_sha256=status["checkpoint_sha256"], original_pretraining_seconds=status["seconds"],
               note="Full end-to-end cost includes the original path pretraining; reuse avoids repeating it."))
    print(f"Reused frozen pair checkpoint: {source} -> {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--include-distillation", action="store_true")
    args = parser.parse_args()
    reuse(args.source, args.run_dir, args.config, args.include_distillation)

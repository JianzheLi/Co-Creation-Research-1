"""Recover an existing completed first pass under the fixed-step stop policy.

Only metadata changes; weights and independent accuracy results are preserved.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import file_hash, json_write, load_config, update_stage


def recover(run):
    run = Path(run)
    config = load_config(run / "config.yaml")
    source = run / "cost_network_unvalidated.pt"
    destination = run / "cost_network.pt"
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    steps = config["distillation"]["training_steps"]
    assert checkpoint["refinement_round"] == 0, "only the original optimizer pass may be recovered"
    assert checkpoint["path_checkpoint_sha256"] == file_hash(run / "pair_path.pt")
    assert checkpoint["distillation_config"] == config["distillation"]
    history = [json.loads(line) for line in (run / "distillation_history.jsonl").read_text().splitlines()]
    first_pass = [row for row in history if row.get("refinement_round", 0) == 0]
    assert first_pass[-1]["step"] == steps
    validation = json.loads((run / "distillation_validation.json").read_text())
    assert validation["training_seconds"] == checkpoint["training_seconds"]
    metadata = dict(training_complete=True, completed_steps=steps, stopping_rule="fixed_steps")
    checkpoint.update(**metadata, recovered_from_sha256=file_hash(source))
    torch.save(checkpoint, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=False)
    assert all(torch.equal(value, restored["model"][key]) for key, value in checkpoint["model"].items())
    validation.update(metadata)
    json_write(run / "distillation_validation.json", validation)
    recovery = dict(**metadata, source_checkpoint=source.name, source_sha256=file_hash(source),
                    checkpoint=destination.name, checkpoint_sha256=file_hash(destination),
                    weights_unchanged=True, discarded_refinement_log_records=len(history)-len(first_pass),
                    last_used_training_record=first_pass[-1])
    json_write(run / "distillation_recovery.json", recovery)
    update_stage(run, "distillation", "completed", phase="completed", **metadata,
                 seconds=checkpoint["total_seconds"], validation=validation)
    print(json.dumps(recovery, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    recover(parser.parse_args().run_dir)

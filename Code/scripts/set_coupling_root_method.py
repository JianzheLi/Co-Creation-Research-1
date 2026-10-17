"""Apply a requested root-method change while preserving trained stages."""
import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config, json_write, update_stage

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--run-dir", required=True)
args = parser.parse_args()
run = Path(args.run_dir)
previous = load_config(run / "config.yaml")
config = load_config(args.config)
old_method = previous["coupling"].get("root_method", "newton")
previous["coupling"]["root_method"] = config["coupling"]["root_method"]
if previous != config:
    raise ValueError("only the coupling root method may change in this run")
if (run / "couplings" / "manifest.json").exists():
    raise ValueError("preserve completed couplings in a separate experiment")
json_write(run / "coupling_root_method_change.json", dict(previous=old_method,
           current=config["coupling"]["root_method"], trained_checkpoints_unchanged=True))
(run / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
update_stage(run, "couplings", "interrupted", reason="User requested bisection before continuation")
print(f"Coupling root method: {old_method} -> {config['coupling']['root_method']}")

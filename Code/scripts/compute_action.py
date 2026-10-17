from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.action import compute_action
from src.runtime import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate the trained feedback model's RUOT running action")
    parser.add_argument("--config", default="configs/gene_steps_gpu.yaml")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = compute_action(load_config(args.config), args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))

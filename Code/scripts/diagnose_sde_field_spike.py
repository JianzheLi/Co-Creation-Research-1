"""Probe local time/state sensitivity around an unstable SDE rollout point."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code: src/

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.experiment import make_feedback
from src.runtime import configure, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True,
                        help="rollout step whose score-extreme particle is probed")
    parser.add_argument("--field", choices=("score", "singular"), default="score",
                        help="which field's recorded extreme particle to probe")
    parser.add_argument("--state-eps", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--time-eps", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    args = parser.parse_args()

    run = args.run_dir.resolve()
    config = load_config(run / "config.yaml")
    device = configure(config)
    checkpoint = torch.load(run / "feedback.pt", map_location=device, weights_only=False)
    model = make_feedback(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    trace_rows = [json.loads(line) for line in args.trace_file.read_text(encoding="utf-8").splitlines()]
    row = next(item for item in trace_rows if int(item["step"]) == args.step)
    previous = next(item for item in trace_rows if int(item["step"]) == args.step - 1)
    particle_key = ("score_extreme_particle" if args.field == "score"
                    else "singular_velocity_extreme_particle")
    particle = row[particle_key]
    x = torch.as_tensor(particle["position_before"], dtype=torch.float32, device=device).reshape(1, -1)
    t = torch.full((1, 1), float(previous["time"]), dtype=torch.float32, device=device)
    fields = ("v_regular", "v_singular_scaled", "growth", "score_scaled")

    def evaluate(t_batch: torch.Tensor, x_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return dict(zip(fields, model.matching_fields(t_batch, x_batch)))

    result = {
        "run": str(run),
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "rollout_step": int(args.step),
        "field_extreme": args.field,
        "particle_index": int(particle["index"]),
        "time_input": float(t.item()),
        "position_linf": float(x.abs().max().item()),
        "field_values": {},
        "state_finite_difference": {},
        "time_finite_difference": {},
    }
    with torch.no_grad():
        base = evaluate(t, x)
        result["field_values"] = {
            name: {"max_abs": float(value.abs().max().item()),
                   "l2": float(value.norm().item()),
                   "max_abs_coordinate": int(value.abs().argmax().item())}
            for name, value in base.items()
        }

        dimension = x.shape[1]
        eye = torch.eye(dimension, dtype=x.dtype, device=device)
        for eps in args.state_eps:
            probes = torch.cat((x + eps * eye, x - eps * eye), dim=0)
            times = t.expand(2 * dimension, 1)
            values = evaluate(times, probes)
            row_result = {}
            for name, value in values.items():
                jacobian_columns = ((value[:dimension] - value[dimension:]) / (2 * eps)).T
                row_result[name] = {
                    "finite_difference_frobenius": float(jacobian_columns.norm().item()),
                    "max_column_norm": float(jacobian_columns.norm(dim=0).max().item()),
                }
            result["state_finite_difference"][str(eps)] = row_result

        for eps in args.time_eps:
            values = evaluate(torch.cat((t - eps, t + eps), dim=0), x.expand(2, -1))
            row_result = {}
            for name, value in values.items():
                time_derivative = (value[1] - value[0]) / (2 * eps)
                row_result[name] = float(time_derivative.norm().item())
            result["time_finite_difference"][str(eps)] = row_result

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

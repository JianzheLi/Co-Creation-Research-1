"""Compute full-data SDE W1 for selected seeds of one completed run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_architecture_w1_seeds import evaluate_sde_seed, failed_seed_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[342, 343])
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    frames = []
    for index, seed in enumerate(args.seeds, start=1):
        cached = run_dir / f"sde_seed_{seed}_metrics.csv"
        if cached.exists():
            frame = pd.read_csv(cached)
            if "evaluation_version" in frame and (frame.evaluation_version == 2).all():
                frames.append(frame)
                print(f"[{index}/{len(args.seeds)}] reusing seed {seed}", flush=True)
                continue
        try:
            frame = evaluate_sde_seed(run_dir, seed)
        except (FloatingPointError, RuntimeError) as error:
            frame = failed_seed_frame(run_dir, seed, error)
            frame.to_csv(cached, index=False)
            print(f"[{index}/{len(args.seeds)}] seed {seed} failed: {error}", flush=True)
        else:
            print(f"[{index}/{len(args.seeds)}] seed {seed} complete; mean noninitial W1="
                  f"{frame.loc[frame.time > 0, 'W1'].mean():.6g}", flush=True)
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(run_dir / "selected_seed_w1_metrics.csv", index=False)
    noninitial = metrics.loc[metrics.time > 0]
    summary = {
        "run_dir": str(run_dir),
        "dataset": str(metrics.dataset.iloc[0]),
        "seeds": [int(seed) for seed in args.seeds],
        "seed_count": len(args.seeds),
        "mean_noninitial_W1": float(noninitial.W1.mean()),
        "sd_noninitial_W1_across_seed_time_rows": float(noninitial.W1.std(ddof=1)),
        "mean_mass_relative_error": float(noninitial.mass_relative_error.mean()),
        "failed_rows": int((noninitial.status != "completed").sum()),
        "rows": noninitial.to_dict(orient="records"),
    }
    (run_dir / "selected_seed_w1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

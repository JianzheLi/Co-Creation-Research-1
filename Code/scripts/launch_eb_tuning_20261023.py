"""Launch the six declared EB candidates as their shared couplings finish."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "Results" / "eb_tuning_20261023"
RECIPES = [("affine_silu", 0.003), ("nonaffine_silu", 0.001),
           ("nonaffine_leaky", 0.001)]


def main():
    jobs = []
    for blocks, first_gpu in [(8, 2), (2, 5)]:
        for offset, (recipe, learning_rate) in enumerate(RECIPES):
            name = f"b{blocks}_{recipe}_lr{learning_rate:g}_seed42"
            jobs.append(dict(name=name, blocks=blocks, recipe=recipe,
                             learning_rate=learning_rate, gpu=first_gpu + offset,
                             status="waiting_for_couplings"))
    processes = {}
    handles = {}
    while any(job["status"] in {"waiting_for_couplings", "running"} for job in jobs):
        for job in jobs:
            name = job["name"]
            destination = OUT / name
            if job["status"] == "waiting_for_couplings":
                if (destination / "tuning_summary.json").exists():
                    job["status"] = "completed"
                    continue
                source = OUT / f"shared_blocks{job['blocks']}"
                state_file = source / "status.json"
                if not state_file.exists():
                    continue
                state = json.loads(state_file.read_text()).get("couplings", {}).get("status")
                if state in {"failed", "nonfinite", "not_converged"}:
                    job["status"] = "coupling_failed"
                    continue
                if state != "completed":
                    continue
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=str(job["gpu"]),
                                   MKL_THREADING_LAYER="SEQUENTIAL", OMP_NUM_THREADS="4")
                command = [sys.executable, "-u", str(ROOT / "scripts/run_eb_tuning.py"),
                           "matching", "--source-run", str(source), "--run-dir", str(destination),
                           "--recipe", job["recipe"], "--learning-rate", str(job["learning_rate"]),
                           "--steps", "30000", "--evaluation-seeds", "342", "343"]
                handles[name] = (OUT / "logs" / f"{name}.log").open("a")
                processes[name] = subprocess.Popen(command, cwd=ROOT, env=environment,
                                                   stdout=handles[name], stderr=subprocess.STDOUT)
                job.update(status="running", pid=processes[name].pid, command=command)
                print(f"Started {name} on GPU {job['gpu']}", flush=True)
            elif job["status"] == "running":
                code = processes[name].poll()
                if code is not None:
                    handles[name].close()
                    job.update(status="completed" if code == 0 else "failed", exit_code=code)
                    print(f"Finished {name}: {job['status']}", flush=True)
        (OUT / "campaign_status.json").write_text(json.dumps(jobs, indent=2))
        if any(job["status"] in {"waiting_for_couplings", "running"} for job in jobs):
            time.sleep(20)
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == "__main__":
    main()

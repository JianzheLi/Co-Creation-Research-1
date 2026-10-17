"""Compare batching and synchronization using identical frozen cost queries."""
import importlib.util
import argparse
import json
from pathlib import Path
import sys
import time
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.runtime import configure, load_config, load_path, snapshots_for, json_write
from src.cost_distillation import load_distilled_oracle
from src.neural_coupling import NeuralPairCost
from src.primal_coupling_gpu import solve_primal_gpu


def load_reference(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--reference-dir', type=Path,
                        help='Optional directory containing neural_before.py and primal_before.py')
    args = parser.parse_args()
    run = args.run_dir
    output = run/'acceleration_check'
    output.mkdir(exist_ok=True)
    config = load_config(run/'config.yaml')
    device = configure(config)
    path = load_path(run, config, device)
    oracle = load_distilled_oracle(config, run, path, device)
    data = snapshots_for(config)
    delta = data[-1].x[None, :, :]-data[-2].x[:, None, :]
    table = NeuralPairCost(oracle, delta, root_method='bisection')
    ratio = torch.exp(torch.linspace(np.log(.1), np.log(10.), np.prod(table.shape),
                                     dtype=torch.float64, device=device)).reshape(table.shape).clamp(.1, 10.)
    reference = table(ratio)
    measurements = []
    for chunk in (4096, 16384, 32768):
        table.chunk = chunk
        table(ratio)
        table.value(ratio)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            values = table(ratio)
            table.value(ratio)
            torch.cuda.synchronize()
            samples.append(time.perf_counter()-started)
        errors = [float((a-b).abs().max()) for a,b in zip(reference, values)]
        assert max(errors) < 1e-10, errors
        row = dict(chunk=chunk, pairs=ratio.numel(), median_seconds=statistics.median(samples),
                   max_absolute_errors=errors, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        measurements.append(row)
        print(json.dumps(row), flush=True)
    selected = min(measurements, key=lambda row: row['median_seconds'])['chunk']
    if args.reference_dir is None:
        json_write(output/'benchmark.json', dict(query_measurements=measurements, selected_chunk=selected))
        return
    old_neural = load_reference('src._neural_before', args.reference_dir/'neural_before.py')
    old_primal = load_reference('src._primal_before', args.reference_dir/'primal_before.py')
    delta = data[1].x[None, :, :]-data[0].x[:, None, :]
    old_table = old_neural.NeuralPairCost(oracle, delta, root_method='bisection')
    old_table.chunk = 4096
    new_table = NeuralPairCost(oracle, delta, root_method='bisection')
    new_table.chunk = selected
    results = {}
    for label, solver, cost in [('before', old_primal.solve_primal_gpu, old_table),
                                ('after', solve_primal_gpu, new_table)]:
        result = solver(data[0].weights, data[1].weights, cost, max_iterations=100,
                        stationarity_tolerance=1e-3, device=device)
        results[label] = result
        print(json.dumps(dict(variant=label, seconds=result.seconds,
                              final_residual=result.history[-1]['projected_gradient_relative_l1'])), flush=True)
    errors = dict(P=float(np.abs(results['before'].P-results['after'].P).max()),
                  Q=float(np.abs(results['before'].Q-results['after'].Q).max()))
    assert max(errors.values()) < 1e-10, errors
    json_write(output/'benchmark.json', dict(query_measurements=measurements, selected_chunk=selected,
        solver_steps=100, before_seconds=results['before'].seconds, after_seconds=results['after'].seconds,
        coupling_max_absolute_errors=errors, speedup=results['before'].seconds/results['after'].seconds))


if __name__ == '__main__':
    main()

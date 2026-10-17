"""Time teacher quadrature and its distilled network on identical GPU inputs."""
import argparse
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure, load_config, load_path, json_write
from src.cost_distillation import load_distilled_oracle, canonical_delta
from src.pair_cost import CostOracle


def benchmark(run):
    run = Path(run)
    config = load_config(run/'config.yaml')
    device = configure(config)
    path = load_path(run, config, device)
    student = load_distilled_oracle(config, run, path, device)
    teacher = CostOracle(path, student.nodes, 4096)
    generator = torch.Generator(device=device).manual_seed(819)
    points = torch.rand((2048, 2), dtype=torch.float64, device=device, generator=generator)
    delta = canonical_delta(points[:, :1]*student.distance_max, path.dimension)
    ratio = student.ratio_min+(student.ratio_max-student.ratio_min)*points[:, 1:]
    report = dict(conditions=len(ratio), device=str(device), teacher_quadrature_nodes=student.nodes,
                  timing='median of 7 synchronized resident GPU queries', results=[])
    for derivatives in (1, 2):
        row = dict(derivatives=derivatives)
        for name, oracle in [('teacher', teacher), ('student', student)]:
            oracle.tensor_query(delta, ratio, derivatives)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                oracle.tensor_query(delta, ratio, derivatives)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                samples.append(time.perf_counter()-start)
            row[name+'_seconds'] = statistics.median(samples)
        row['speedup'] = row['teacher_seconds']/row['student_seconds']
        report['results'].append(row)
        print(row, flush=True)
    json_write(run/'teacher_student_query_benchmark.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    benchmark(parser.parse_args().run_dir)

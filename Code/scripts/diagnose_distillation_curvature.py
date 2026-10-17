"""Read-only teacher/student curvature diagnosis; never trains or edits weights."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.cost_distillation import canonical_delta, load_distilled_oracle
from src.pair_cost import CostOracle
from src.runtime import configure, file_hash, json_write, load_config


def diagnose(run):
    run = Path(run)
    config = load_config(run / "config.yaml")
    device = configure(config)
    started = time.perf_counter()
    checkpoint_hash = file_hash(run / "cost_network.pt")
    student = load_distilled_oracle(config, run, device=device, diagnostic_only=True)
    teacher = CostOracle(student.path, student.nodes, student.condition_chunk)
    distances = torch.linspace(0, student.distance_max, 513, dtype=torch.float64, device=device)
    ratios = torch.cat((torch.linspace(math.log(student.ratio_min), math.log(student.ratio_max),
                       1025, dtype=torch.float64, device=device).exp(),
                       torch.linspace(8, 10, 257, dtype=torch.float64, device=device))).unique()
    dd, rr = torch.meshgrid(distances, ratios, indexing="ij")
    d, r = dd.reshape(-1, 1), rr.reshape(-1, 1)
    delta = canonical_delta(d, student.path.dimension)
    print(f"Diagnosing {len(d)} points; teacher and student use identical (d,r)", flush=True)
    reference = teacher.tensor_query(delta, r, 2)
    prediction = student.tensor_query(delta, r, 2)
    negative = prediction[2] < 0
    def summarize(values):
        index = int(values[2].argmin())
        return dict(min_curvature=float(values[2][index]), negative_count=int((values[2] < 0).sum()),
                    worst_distance=float(d[index]), worst_ratio=float(r[index]))
    report = dict(samples=len(d), distance_range=[0, student.distance_max], ratio_range=[.1, 10.],
                  teacher=summarize(reference), student=summarize(prediction),
                  checkpoint_sha256=checkpoint_hash, path_checkpoint_sha256=file_hash(run/"pair_path.pt"),
                  normalization_cost_scale=float(student.model.cost_scale),
                  training_grid_ratio_nodes=config["distillation"]["ratio_grid_nodes"],
                  training_grid_endpoint_fraction=1/config["distillation"]["ratio_grid_nodes"],
                  boundary_bins=[], probes=[])
    report["overall_errors"] = {
        name: dict(max_absolute=float((pred-ref).abs().max()), rmse=float((pred-ref).square().mean().sqrt()))
        for name, ref, pred in zip(("cost", "slope", "curvature"), reference, prediction)
    }
    if bool(negative.any()):
        report["negative_region"] = dict(distance_min=float(d[negative].min()), distance_max=float(d[negative].max()),
                                        ratio_min=float(r[negative].min()), ratio_max=float(r[negative].max()),
                                        teacher_min_curvature=float(reference[2][negative].min()))
    for lower, upper in ((.1, 1.), (1., 5.), (5., 8.), (8., 9.), (9., 9.5), (9.5, 10.00000001)):
        mask = (r.flatten() >= lower) & (r.flatten() < upper)
        errors = {}
        for name, ref, pred in zip(("cost", "slope", "curvature"), reference, prediction):
            diff = pred[mask]-ref[mask]
            errors[name] = dict(rmse=float(diff.square().mean().sqrt()), max_absolute=float(diff.abs().max()))
        report["boundary_bins"].append(dict(ratio_range=[lower, min(upper, 10.)], samples=int(mask.sum()),
                                           teacher_min_curvature=float(reference[2][mask].min()),
                                           student_min_curvature=float(prediction[2][mask].min()), errors=errors))
    # Re-evaluate both a worst point and the actual pair flagged by Coupling.
    worst = int(prediction[2].argmin())
    probe_d = torch.tensor([float(d[worst]), 2.2350315761123247], dtype=torch.float64, device=device)[:, None]
    local_r = torch.tensor([9., 9.5, 9.8, 9.9, 9.99, 10.], dtype=torch.float64, device=device)[:, None]
    pd, pr = probe_d.repeat_interleave(len(local_r), 0), local_r.repeat(len(probe_d), 1)
    pdelta = canonical_delta(pd, student.path.dimension)
    teacher64 = CostOracle(student.path, 2*teacher.nodes, teacher.condition_chunk)
    exact = teacher.tensor_query(pdelta, pr, 2)
    fine = teacher64.tensor_query(pdelta, pr, 2)
    pred = student.tensor_query(pdelta, pr, 2)
    # Backward finite differences stay inside the trained ratio domain at r=10.
    h = 1e-3
    tminus = teacher.tensor_query(pdelta, pr-h, 1)[1]
    tminus2 = teacher.tensor_query(pdelta, pr-2*h, 1)[1]
    sminus = student.tensor_query(pdelta, pr-h, 1)[1]
    sminus2 = student.tensor_query(pdelta, pr-2*h, 1)[1]
    teacher_fd = (3*exact[1]-4*tminus+tminus2)/(2*h)
    student_fd = (3*pred[1]-4*sminus+sminus2)/(2*h)
    for k in range(len(pd)):
        report["probes"].append(dict(distance=float(pd[k]), ratio=float(pr[k]),
            teacher_cost=float(exact[0][k]), student_cost=float(pred[0][k]),
            teacher_slope=float(exact[1][k]), student_slope=float(pred[1][k]),
            teacher_curvature=float(exact[2][k]), student_curvature=float(pred[2][k]),
            teacher_curvature_finite_difference=float(teacher_fd[k]),
            student_curvature_finite_difference=float(student_fd[k])))
    report["teacher_quadrature_check"] = dict(nodes=teacher.nodes, reference_nodes=teacher64.nodes,
        max_absolute_differences=[float((a-b).abs().max()) for a, b in zip(exact, fine)])
    report["finite_difference_check"] = dict(step=h, scheme="second_order_backward_difference_of_slope",
        teacher_max_error=float((teacher_fd-exact[2]).abs().max()), student_max_error=float((student_fd-pred[2]).abs().max()))
    # Quantify what the original training objective sees on its own grid.
    grid = torch.load(run/"distillation_grid.pt", map_location=device, weights_only=False)
    grid_prediction = student.tensor_query(canonical_delta(grid["distance"], student.path.dimension), grid["ratio"], 2)
    value_error = (grid_prediction[0]-grid["cost"])/student.model.cost_scale
    slope_error = (grid_prediction[1]-grid["derivative"])/student.model.cost_scale
    report["original_grid_objective"] = dict(value_mse=float(value_error.square().mean()),
        slope_mse=float(slope_error.square().mean()), total=float(value_error.square().mean()+slope_error.square().mean()),
        student_negative_curvature_count=int((grid_prediction[2] < 0).sum()), grid_points=len(grid["distance"]))
    assert checkpoint_hash == file_hash(run/"cost_network.pt")
    report["seconds"] = time.perf_counter()-started
    report["weights_unchanged"] = True
    json_write(run/"curvature_diagnosis.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    diagnose(parser.parse_args().run_dir)

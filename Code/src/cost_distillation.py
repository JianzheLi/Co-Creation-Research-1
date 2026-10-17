"""Ordinary smooth MLP distillation of frozen distance-conditioned path costs.

Only teacher-label generation uses time quadrature. Every student query and
ratio derivative uses the same MLP; no convexity architecture or loss penalty.
"""
import json
import math
import time
import warnings
from pathlib import Path

import torch
from torch import nn

from .pair_cost import CostOracle
from .runtime import (configure, file_hash, init_run, json_write, load_path,
                      log_json, snapshots_for, update_stage)


class CostNetwork(nn.Module):
    def __init__(self, distance_max, ratio_min, ratio_max, width=128, layers=3,
                 cost_center=0., cost_scale=1.):
        super().__init__()
        if distance_max <= 0 or not 0 < ratio_min < ratio_max or cost_scale <= 0:
            raise ValueError("invalid cost network domain/scaling")
        # Buffers are constructed directly in float64; module.double() must not
        # silently preserve float32-rounded normalization constants.
        self.register_buffer("input_center", torch.tensor(
            [distance_max/2, (math.log(ratio_min)+math.log(ratio_max))/2], dtype=torch.float64))
        self.register_buffer("input_scale", torch.tensor(
            [distance_max/2, (math.log(ratio_max)-math.log(ratio_min))/2], dtype=torch.float64))
        self.register_buffer("cost_center", torch.tensor(cost_center, dtype=torch.float64))
        self.register_buffer("cost_scale", torch.tensor(cost_scale, dtype=torch.float64))
        modules = []
        for k in range(layers):
            modules.extend((nn.Linear(2 if k == 0 else width, width), nn.Tanh()))
        modules.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*modules)

    def normalized_inputs(self, distance, log_r):
        return (torch.cat((distance, log_r), -1)-self.input_center)/self.input_scale

    def forward(self, distance, log_r):
        return self.cost_center + self.cost_scale * self.net(self.normalized_inputs(distance, log_r))


class DistilledCostOracle(CostOracle):
    """Drop-in GPU oracle; inherited autodiff differentiates only cost_tensor."""
    def __init__(self, path, model, ratio_min, ratio_max, distance_max,
                 condition_chunk=4096, teacher_nodes=32):
        # Reuse only the autodiff/chunking interface, without even constructing
        # quadrature nodes in the deployed student oracle.
        self.path = path.freeze()
        self.nodes, self.condition_chunk = teacher_nodes, condition_chunk
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.ratio_min, self.ratio_max, self.distance_max = ratio_min, ratio_max, distance_max

    def cost_tensor(self, delta, log_r):
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        return self.model(distance, log_r).squeeze(-1)

    def tensor_query(self, delta, r, derivatives=2):
        p = next(self.model.parameters())
        delta = torch.as_tensor(delta, device=p.device, dtype=p.dtype).reshape(-1, self.path.dimension)
        r = torch.as_tensor(r, device=p.device, dtype=p.dtype)
        # Never silently extrapolate the distilled cost outside its teacher grid.
        valid = ((delta.square().sum(-1) <= (self.distance_max*(1+1e-12))**2).all()
                 & (r >= self.ratio_min*(1-1e-12)).all() & (r <= self.ratio_max*(1+1e-12)).all())
        if not bool(valid):
            raise ValueError("cost query outside distilled distance/ratio domain")
        return super().tensor_query(delta, r, derivatives)


def canonical_delta(distance, dimension):
    return torch.cat((distance.reshape(-1, 1), distance.new_zeros((distance.numel(), dimension-1))), -1)


def reduce_squared_error(error, reduction="mean", max_error_weight=1.):
    squared = error.square()
    if reduction == "mean":
        return squared.mean()
    if reduction == "mean_plus_max":
        if max_error_weight < 0:
            raise ValueError("max_error_weight must be nonnegative")
        return squared.mean() + max_error_weight*squared.max()
    raise ValueError(f"unknown distillation loss reduction: {reduction}")


def loss_on_grid(model, inputs, values, slopes, reduction="mean", max_error_weight=1.,
                 curvatures=None, curvature_weight=1.):
    z = inputs.detach().requires_grad_(True)
    prediction = model.net(z)
    derivative_z = torch.autograd.grad(prediction.sum(), z, create_graph=True)[0][:, 1:2]
    ratio = (z[:, 1:2]*model.input_scale[1]+model.input_center[1]).exp()
    # Supervise dC/dr directly (in the same output units as normalized C).
    derivative = derivative_z / (ratio*model.input_scale[1])
    value_loss = reduce_squared_error(prediction-values, reduction, max_error_weight)
    slope_loss = reduce_squared_error(derivative-slopes, reduction, max_error_weight)
    if curvatures is not None:
        # z_r is affine in log(r): d/dr = (r*input_scale[1])^-1 d/dz_r.
        # Differentiating derivative includes the derivative of its 1/r factor.
        curvature_z = torch.autograd.grad(derivative.sum(), z, create_graph=True)[0][:, 1:2]
        curvature = curvature_z / (ratio*model.input_scale[1])
        curvature_loss = reduce_squared_error(curvature-curvatures, reduction, max_error_weight)
        return value_loss+slope_loss+curvature_weight*curvature_loss, value_loss, slope_loss, curvature_loss
    return value_loss+slope_loss, value_loss, slope_loss


def sample_continuous_conditions(count, distance_max, ratio_min, ratio_max, reference):
    """Independent uniform draws in physical d and r, not in log(r)."""
    uniform = torch.rand((count, 2), dtype=reference.dtype, device=reference.device)
    return uniform[:, :1]*distance_max, ratio_min+uniform[:, 1:]*(ratio_max-ratio_min)


def sample_boundary_indices(boundary, corners, count):
    if count == 0:
        return boundary[:0]
    if count == len(boundary):
        return boundary
    if not len(corners) <= count <= len(boundary):
        raise ValueError("boundary sample count must include corners and not exceed boundary size")
    draw = torch.randint(len(boundary), (count-len(corners),), device=boundary.device)
    return torch.cat((corners, boundary[draw]))


def check_deployment_curvature(oracle, cfg):
    """Check a refined grid (including every edge), plus independent points.

    This is a deployment diagnostic, not a proof of continuous convexity and
    not a teacher target, training penalty, or automatic retraining trigger.
    """
    parameter = next(oracle.model.parameters())
    options = dict(dtype=parameter.dtype, device=parameter.device)
    nd, nr = 4*(cfg["distance_grid_nodes"]-1)+1, 4*(cfg["ratio_grid_nodes"]-1)+1
    distances = torch.linspace(0, oracle.distance_max, nd, **options)
    ell = torch.linspace(math.log(oracle.ratio_min), math.log(oracle.ratio_max), nr, **options)
    dd, ee = torch.meshgrid(distances, ell, indexing="ij")
    generator = torch.Generator(device=parameter.device).manual_seed(1949)
    random = torch.rand((cfg["validation_samples"], 2), generator=generator, **options)
    d = torch.cat((dd.flatten(), random[:, 0]*oracle.distance_max))[:, None]
    r = torch.cat((ee.flatten(), ell[0]+random[:, 1]*(ell[-1]-ell[0])))[:, None].exp()
    curvature = oracle.tensor_query(canonical_delta(d, oracle.path.dimension), r, 2)[2]
    index = int(curvature.argmin())
    finite = bool(torch.isfinite(curvature).all())
    return dict(passed=finite and bool((curvature > 0).all()), samples=len(d),
                grid_shape=[nd, nr], negative_or_zero_count=int((curvature <= 0).sum()),
                min_curvature=float(curvature[index]) if finite else None,
                worst_distance=float(d[index]), worst_ratio=float(r[index]),
                global_convexity_proven=False, evaluation="direct_mlp_autodiff")


def load_distilled_oracle(config, run, path=None, device=None, *, diagnostic_only=False):
    run = Path(run)
    if device is None:
        device = configure(config)
    checkpoint_file = run / "cost_network.pt"
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    if checkpoint["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt"):
        raise RuntimeError("distilled cost checkpoint belongs to another path")
    if checkpoint["distillation_config"] != config["distillation"]:
        raise RuntimeError("distilled cost configuration differs from checkpoint")
    if not (checkpoint.get("training_complete") or checkpoint.get("validated")):
        raise RuntimeError("distilled cost training is incomplete")
    model = CostNetwork(**checkpoint["network_config"]).to(device=device, dtype=torch.float64)
    model.load_state_dict(checkpoint["model"])
    path = path if path is not None else load_path(run, config, device)
    domain = checkpoint["domain"]
    result = DistilledCostOracle(path, model, **domain,
                                condition_chunk=config["coupling"]["condition_chunk"],
                                teacher_nodes=checkpoint["teacher_quadrature_nodes"])
    result.checkpoint_sha256 = file_hash(checkpoint_file)
    if not diagnostic_only and config["distillation"].get("loss_reduction") == "mean_plus_max":
        report_file = run / "deployment_curvature.json"
        report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
        if report.get("cost_network_sha256") != result.checkpoint_sha256:
            report = check_deployment_curvature(result, config["distillation"])
            report["cost_network_sha256"] = result.checkpoint_sha256
            json_write(report_file, report)
        if not report["passed"]:
            if report["min_curvature"] is None:
                from .coupling import OracleError
                raise OracleError("cost_oracle_nonfinite", report)
            warnings.warn(f"Distilled cost has nonpositive sampled curvature: {report}; continuing with diagnostics",
                          RuntimeWarning, stacklevel=2)
        result.deployment_curvature = report
    return result


def train_distillation(config, run):
    device = configure(config)
    run = init_run(run, config)
    cfg = config["distillation"]
    path = load_path(run, config, device)
    if path.parameterization != "axial_distance":
        raise ValueError("distance-only distillation requires axial_distance PairPath")
    if (run / "cost_network.pt").exists():
        oracle = load_distilled_oracle(config, run, path, device)
        print("Reusing completed distilled cost network", flush=True)
        return oracle
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("pair", {}).get("status") != "completed":
        raise RuntimeError("complete path pretraining before distillation")
    started = time.perf_counter()
    update_stage(run, "distillation", "running", phase="teacher_grid")
    snapshots = snapshots_for(config)
    from .solve_couplings import select_quadrature
    teacher = select_quadrature(path, snapshots, config, run)
    teacher.condition_chunk = cfg.get("teacher_condition_chunk", teacher.condition_chunk)
    points = [torch.as_tensor(s.x, dtype=torch.float64, device=device) for s in snapshots]
    distance_max = max(float(torch.cdist(a, b).amax()) for a, b in zip(points[:-1], points[1:]))
    distance_max = max(distance_max * (1+1e-10), torch.finfo(torch.float64).eps)
    ratio_min, ratio_max = config["pair"]["ratio_min"], config["pair"]["ratio_max"]
    distances = torch.linspace(0, distance_max, cfg["distance_grid_nodes"], dtype=torch.float64, device=device)
    ell = torch.linspace(math.log(ratio_min), math.log(ratio_max), cfg["ratio_grid_nodes"], dtype=torch.float64, device=device)
    dd, ee = torch.meshgrid(distances, ell, indexing="ij")
    d, r = dd.reshape(-1, 1), ee.reshape(-1, 1).exp()
    teacher_started = time.perf_counter()
    curvature_weight = cfg.get("curvature_weight", 0.)
    if curvature_weight < 0:
        raise ValueError("curvature_weight must be nonnegative")
    supervise_curvature = curvature_weight > 0
    teacher_values = teacher.tensor_query(canonical_delta(d, path.dimension), r, 2 if supervise_curvature else 1)
    cost, derivative = teacher_values[:2]
    curvature = teacher_values[2] if supervise_curvature else None
    grid_seconds = time.perf_counter()-teacher_started
    center, scale = float(cost.mean()), max(float(cost.std()), 1e-12)
    network_config = dict(distance_max=distance_max, ratio_min=ratio_min, ratio_max=ratio_max,
                          width=cfg["hidden_width"], layers=cfg["hidden_layers"], cost_center=center, cost_scale=scale)
    model = CostNetwork(**network_config).to(device=device, dtype=torch.float64)
    inputs = model.normalized_inputs(d, r.log()).detach()
    values = (cost[:, None]-center)/scale
    slopes = derivative[:, None]/scale
    curvatures = curvature[:, None]/scale if supervise_curvature else None
    reduction = cfg.get("loss_reduction", "mean")
    sampling = cfg.get("sampling", "grid")
    if sampling not in {"grid", "continuous"}:
        raise ValueError("distillation sampling must be grid or continuous")
    max_error_weight = cfg.get("max_error_weight", 1.)
    # Boundary coverage is explicit. A dense grid may subsample its boundary;
    # all four corners remain present in every maximum-error training batch.
    boundary = ((dd == distances[0]) | (dd == distances[-1]) |
                (ee == ell[0]) | (ee == ell[-1])).flatten().nonzero().flatten()
    batch_size = min(cfg["batch_size"], len(inputs))
    corners = ((dd == distances[0]) | (dd == distances[-1])) & ((ee == ell[0]) | (ee == ell[-1]))
    corners = corners.flatten().nonzero().flatten()
    boundary_count = cfg.get("boundary_samples_per_batch", len(boundary)) if reduction == "mean_plus_max" else 0
    if boundary_count >= batch_size:
        raise ValueError("distillation batch_size must exceed the boundary grid size")
    if boundary_count and not len(corners) <= boundary_count <= len(boundary):
        raise ValueError("invalid boundary_samples_per_batch")
    # Persist teacher samples once; all training tensors remain resident on GPU.
    torch.save(dict(distance=d, ratio=r, cost=cost, derivative=derivative, curvature=curvature,
                    path_checkpoint_sha256=file_hash(run / "pair_path.pt")), run / "distillation_grid.pt")
    print(f"Distillation: grid={tuple(dd.shape)}, samples={len(d)}, teacher seconds={grid_seconds:.2f}, device={device}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    training_started = time.perf_counter()
    for step in range(cfg["training_steps"]):
        floor = cfg.get("min_learning_rate", cfg["learning_rate"]/1000)
        peak = max(floor, cfg["learning_rate"])
        lr = floor + (peak-floor)*(1+math.cos(math.pi*step/max(cfg["training_steps"]-1, 1)))/2
        for group in optimizer.param_groups:
            group["lr"] = lr
        edge = sample_boundary_indices(boundary, corners, boundary_count)
        if sampling == "continuous":
            batch_d, batch_r = sample_continuous_conditions(batch_size-len(edge), distance_max,
                                                           ratio_min, ratio_max, inputs)
            # Exact fresh teacher labels: no rounding, snapping, or grid interpolation.
            batch_teacher = teacher.tensor_query(canonical_delta(batch_d, path.dimension), batch_r,
                                                 2 if supervise_curvature else 1)
            batch_cost, batch_slope = batch_teacher[:2]
            batch_inputs = torch.cat((inputs[edge], model.normalized_inputs(batch_d, batch_r.log()).detach()))
            batch_values = torch.cat((values[edge], (batch_cost[:, None]-center)/scale))
            batch_slopes = torch.cat((slopes[edge], batch_slope[:, None]/scale))
            batch_curvatures = (torch.cat((curvatures[edge], batch_teacher[2][:, None]/scale))
                                if supervise_curvature else None)
        elif reduction == "mean_plus_max":
            ids = torch.cat((edge, torch.randint(len(inputs), (batch_size-len(edge),), device=device)))
        else:
            ids = torch.randint(len(inputs), (batch_size,), device=device)
        if sampling == "grid":
            batch_inputs, batch_values, batch_slopes = inputs[ids], values[ids], slopes[ids]
            batch_curvatures = curvatures[ids] if supervise_curvature else None
        losses = loss_on_grid(model, batch_inputs, batch_values, batch_slopes, reduction, max_error_weight,
                              curvatures=batch_curvatures, curvature_weight=curvature_weight)
        loss, lv, ls = losses[:3]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step+1) % 200 == 0 or step+1 == cfg["training_steps"]:
            record = dict(phase="adam", step=step+1, learning_rate=lr, loss=float(loss.detach()),
                          loss_reduction=reduction,
                          sampling=sampling, max_error_weight=max_error_weight,
                          value_loss=float(lv.detach()), slope_loss=float(ls.detach()),
                          seconds=time.perf_counter()-training_started)
            if supervise_curvature:
                record.update(curvature_loss=float(losses[3].detach()), curvature_weight=curvature_weight)
            if not math.isfinite(record["loss"]):
                raise FloatingPointError("nonfinite distillation loss")
            log_json(run / "distillation_history.jsonl", record)
            update_stage(run, "distillation", "running", phase="adam", completed_steps=step+1)
            print(f"  distill {step+1}/{cfg['training_steps']}: loss={record['loss']:.3g}", flush=True)
    training_seconds = time.perf_counter()-training_started
    domain = dict(distance_max=distance_max, ratio_min=ratio_min, ratio_max=ratio_max)
    student = DistilledCostOracle(path, model, **domain, condition_chunk=teacher.condition_chunk,
                                  teacher_nodes=teacher.nodes)
    # All cell centers plus independent random points and boundary corners.
    md, me = torch.meshgrid((distances[:-1]+distances[1:])/2, (ell[:-1]+ell[1:])/2, indexing="ij")
    generator = torch.Generator(device=device).manual_seed(config["seed"]+919)
    random = torch.rand((cfg["validation_samples"], 2), dtype=torch.float64, device=device, generator=generator)
    vd = torch.cat((md.reshape(-1), random[:, 0]*distance_max, distances[[0, 0, -1, -1]]))[:, None]
    vr = torch.cat((me.reshape(-1), ell[0]+random[:, 1]*(ell[-1]-ell[0]), ell[[0, -1, 0, -1]]))[:, None].exp()
    validation_delta = canonical_delta(vd, path.dimension)
    update_stage(run, "distillation", "running", phase="independent_validation")
    reference = teacher.tensor_query(validation_delta, vr, 2)
    predicted = student.tensor_query(validation_delta, vr, 2)
    errors = {}
    for label, target, actual in zip(("cost", "first_derivative", "second_derivative"), reference, predicted):
        difference = (actual-target).abs()
        errors[label] = dict(max_absolute=float(difference.max()), rmse=float(difference.square().mean().sqrt()),
                             max_tolerance_ratio=float((difference/(cfg["query_atol"]+cfg["query_rtol"]*target.abs())).max()))
    supervised_labels = ["cost", "first_derivative"] + (["second_derivative"] if supervise_curvature else [])
    passed = all(errors[label]["max_tolerance_ratio"] <= 1 for label in supervised_labels)
    # Accuracy and curvature are report metrics, never automatic retraining triggers.
    validation = dict(passed=passed, samples=len(vd), grid_shape=list(dd.shape), errors=errors,
                      teacher_min_curvature=float(reference[2].min()), student_min_curvature=float(predicted[2].min()),
                      teacher_grid_seconds=grid_seconds, training_seconds=training_seconds,
                      path_checkpoint_sha256=file_hash(run / "pair_path.pt"),
                      architecture="ordinary_tanh_mlp", optimizer="Adam",
                      derivative_supervision="dC/dr and d2C/dr2" if supervise_curvature else "dC/dr",
                      supervised_targets=supervised_labels, curvature_weight=curvature_weight,
                      loss_reduction=reduction, boundary_points_per_batch=boundary_count,
                      sampling=sampling, max_error_weight=max_error_weight,
                      training_complete=True, completed_steps=cfg["training_steps"], stopping_rule="fixed_steps")
    # Same resident conditions, no archival I/O, timed with CUDA synchronization.
    if device.type == "cuda":
        torch.cuda.synchronize()
    timing = {}
    benchmark_d, benchmark_r = validation_delta[:2048], vr[:2048]
    for name, oracle in (("teacher", teacher), ("student", student)):
        oracle.tensor_query(benchmark_d[:16], benchmark_r[:16], 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(3):
            oracle.tensor_query(benchmark_d, benchmark_r, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timing[name+"_seconds"] = (time.perf_counter()-begin)/3
    validation["query_benchmark"] = dict(**timing, conditions=len(benchmark_d),
                                           speedup=timing["teacher_seconds"]/timing["student_seconds"])
    json_write(run / "distillation_validation.json", validation)
    log_json(run / "distillation_validation_history.jsonl", validation)
    checkpoint = dict(model=model.state_dict(), network_config=network_config, domain=domain,
                      distillation_config=cfg, teacher_quadrature_nodes=teacher.nodes,
                      path_checkpoint_sha256=validation["path_checkpoint_sha256"], validated=passed,
                      training_seconds=training_seconds, total_seconds=time.perf_counter()-started,
                      training_complete=True, completed_steps=cfg["training_steps"], stopping_rule="fixed_steps")
    torch.save(checkpoint, run / "cost_network.pt")
    update_stage(run, "distillation", "completed", phase="completed", completed_steps=cfg["training_steps"],
                 stopping_rule="fixed_steps", seconds=checkpoint["total_seconds"], validation=validation)
    print(f"Distillation completed at {cfg['training_steps']} steps; accuracy check={passed}, query speedup={validation['query_benchmark']['speedup']:.2f}x", flush=True)
    return load_distilled_oracle(config, run, path, device)

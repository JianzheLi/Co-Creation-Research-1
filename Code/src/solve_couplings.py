from pathlib import Path
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from .coupling import HermiteCost, OracleError, solve_dual
from .pair_cost import CostOracle
from .data import partition_indices
from .runtime import configure, file_hash, init_run, json_write, load_path, snapshots_for, update_stage


def select_quadrature(path, snapshots, config, run):
    c = config["pair"]
    rng = np.random.default_rng(config["seed"] + 111)
    deltas = []
    for _ in range(c["validation_conditions"]):
        k = rng.integers(len(snapshots) - 1)
        deltas.append(snapshots[k + 1].x[rng.integers(len(snapshots[k + 1].x))] - snapshots[k].x[rng.integers(len(snapshots[k].x))])
    deltas = np.asarray(deltas)
    ratios = np.exp(rng.uniform(math.log(c["ratio_min"]), math.log(c["ratio_max"]), len(deltas)))
    if path.cost_objective == "tight_logmoment_upper_bound":
        checks=[]
        for nodes in (32,64,128):
            path.logmoment_nodes=nodes
            v0=CostOracle(path,64,64)(deltas,ratios)
            path.logmoment_nodes=2*nodes
            v1=CostOracle(path,64,64)(deltas,ratios)
            errors=[float(np.max(np.abs(a-b))) for a,b in zip(v0,v1)]
            passed=all(np.allclose(a,b,atol=c['query_atol'],rtol=c['query_rtol']) for a,b in zip(v0,v1))
            checks.append(dict(nodes=nodes,reference_nodes=2*nodes,max_errors=errors,passed=passed))
            if passed:
                path.logmoment_nodes=nodes
                json_write(run/'radial_quadrature_validation.json',dict(status='passed',selected_nodes=nodes,comparisons=checks))
                break
        else:
            raise OracleError('radial_quadrature_validation_failed',dict(comparisons=checks))
    comparisons = []
    nodes = c["quadrature_nodes"]
    while 2 * nodes <= c["max_quadrature_nodes"]:
        coarse, fine = CostOracle(path, nodes), CostOracle(path, 2 * nodes)
        v0, v1 = coarse(deltas, ratios), fine(deltas, ratios)
        error = [float(np.max(np.abs(a - b))) for a, b in zip(v0, v1)]
        passed = all(np.allclose(a, b, atol=c["query_atol"], rtol=c["query_rtol"]) for a, b in zip(v0, v1))
        comparisons.append(dict(nodes=nodes, reference_nodes=2 * nodes, max_errors=error, passed=passed))
        if passed:
            json_write(run / "quadrature_validation.json", dict(status="passed", selected_nodes=nodes, comparisons=comparisons))
            return CostOracle(path, nodes, config["coupling"]["condition_chunk"])
        nodes *= 2
    raise OracleError("quadrature_validation_failed", dict(comparisons=comparisons))


def build_table(oracle, delta, cfg, ratio_min, ratio_max, cache_path, fingerprint):
    shape = delta.shape[:2]
    flat = delta.reshape(-1, delta.shape[-1])
    rng = np.random.default_rng(991)
    pairs = rng.integers(len(flat), size=min(128, len(flat)))
    ratios = np.exp(rng.uniform(math.log(ratio_min), math.log(ratio_max), len(pairs)))
    # Include one midpoint in every segment for randomly chosen endpoint pairs,
    # plus continuous random checks. Never fit independent derivative tables.
    ngrid = cfg["ratio_grid_nodes"]
    while ngrid <= cfg["max_ratio_grid_nodes"]:
        grid = np.geomspace(ratio_min, ratio_max, ngrid)
        metadata = dict(**fingerprint, ratio_grid_nodes=ngrid, ratio_range=[ratio_min, ratio_max],
                        interpolation="cubic_hermite_in_r", shape=list(shape))
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as saved:
                import json
                if json.loads(str(saved["metadata"])) == metadata:
                    return HermiteCost(saved["grid"], saved["costs"], saved["derivatives"], cfg["curvature_tolerance"]), metadata
        expanded = np.repeat(flat, ngrid, axis=0)
        cost, derivative = oracle(expanded, np.tile(grid, len(flat)), derivatives=1)
        cost, derivative = cost.reshape(*shape, ngrid), derivative.reshape(*shape, ngrid)
        try:
            table = HermiteCost(grid, cost, derivative, cfg["curvature_tolerance"])
        except OracleError as exc:
            # Confirm a failed table against the NETWORK; interpolation alone may
            # have produced spurious curvature. No derivative sorting is applied.
            slope_diff = np.diff(derivative, axis=-1)
            worst = np.unravel_index(np.argmin(slope_diff), slope_diff.shape)
            pair = worst[0] * shape[1] + worst[1]
            probe = np.geomspace(grid[worst[2]], grid[worst[2] + 1], 9)
            direct = oracle(np.repeat(flat[pair:pair + 1], len(probe), axis=0), probe)
            if direct[2].min() < -cfg["curvature_tolerance"]:
                raise OracleError("cost_oracle_nonconvex", dict(**exc.details, delta=flat[pair].tolist(),
                                  ratios=probe.tolist(), cost=direct[0].tolist(),
                                  derivative=direct[1].tolist(), curvature=direct[2].tolist(),
                                  grid_nodes=ngrid))
            ngrid = 2 * ngrid - 1
            continue
        probe_pairs = np.concatenate((pairs, rng.integers(len(flat), size=ngrid - 1)))
        probe_ratios = np.concatenate((ratios, np.sqrt(grid[:-1] * grid[1:])))
        direct = oracle(flat[probe_pairs], probe_ratios)
        # Evaluate a sparse batch by indexing its existing cubic coefficients.
        local = HermiteCost(grid, cost.reshape(-1, ngrid)[probe_pairs], derivative.reshape(-1, ngrid)[probe_pairs], cfg["curvature_tolerance"])
        interpolated = local(probe_ratios)
        errors = [float(np.max(np.abs(a - b))) for a, b in zip(direct, interpolated)]
        if direct[2].min() < -cfg["curvature_tolerance"]:
            worst = int(np.argmin(direct[2]))
            raise OracleError("cost_oracle_nonconvex", dict(delta=flat[probe_pairs[worst]].tolist(),
                              ratio=float(probe_ratios[worst]), curvature=float(direct[2][worst])))
        # Cost and first derivative govern both objective and inverse query. The
        # curvature discrepancy is reported separately; convexity was checked
        # over the entire interpolant and sampled directly from the network.
        passed = all(np.allclose(a, b, atol=cfg["table_atol"], rtol=cfg["table_rtol"]) for a, b in zip(direct[:2], interpolated[:2]))
        if not passed:
            ngrid = 2 * ngrid - 1
            continue
        metadata.update(validation_max_errors=errors, min_curvature=table.min_curvature,
                        min_slope_increment=table.min_slope_increment, validation_samples=len(probe_pairs))
        # Store the base fingerprint separately from numeric validation details.
        import json
        np.savez_compressed(cache_path, grid=grid, costs=cost, derivatives=derivative,
                            metadata=json.dumps({k: v for k, v in metadata.items() if k not in {"validation_max_errors", "min_curvature", "min_slope_increment", "validation_samples"}}))
        return table, metadata
    raise OracleError("cost_table_accuracy_failed", dict(max_grid_nodes=cfg["max_ratio_grid_nodes"]))


def _solve_one_block(config, run, snapshot, path, device, oracle, shared_table,
                     shared_metadata, points, fingerprint, k, block, i, j,
                     source_blocks_count, cfg, primal, shared, distilled):
    """Build and solve one independent coupling block.

    This function is run in a small thread pool.  Each worker gets its own CUDA
    stream; all shared model/table inputs are read-only, while output filenames
    are unique to the block.
    """
    import torch

    source, target = snapshot[k], snapshot[k + 1]
    name = f"interval_{k}" if cfg["mode"] == "full" else f"interval_{k}_block_{block}"
    if shared:
        source_gpu = points[k][torch.as_tensor(i, dtype=torch.long, device=device)]
        target_gpu = points[k + 1][torch.as_tensor(j, dtype=torch.long, device=device)]
        distances = torch.cdist(source_gpu, target_gpu)
        support_hash = __import__("hashlib").sha256(
            np.ascontiguousarray(source.x[i]).tobytes() +
            np.ascontiguousarray(target.x[j]).tobytes()).hexdigest()
        shape = tuple(distances.shape)
        delta = None
    else:
        delta = target.x[j][None, :, :] - source.x[i][:, None, :]
        support_hash = __import__("hashlib").sha256(delta.tobytes()).hexdigest()
        distances = None
        shape = delta.shape[:2]

    block_fingerprint = dict(**fingerprint, support_sha256=support_hash,
                             source_indices=i.tolist(), target_indices=j.tolist())
    label = (f"full matrix {shape}" if cfg["mode"] == "full" else
             f"block={block + 1}/{source_blocks_count}, shape={shape}")
    print(f"{'Direct neural cost' if distilled else 'Cost table'}: interval={k}, {label}", flush=True)
    qstart = time.perf_counter()
    if shared:
        table = shared_table.bind(distances)
        metadata = dict(**block_fingerprint, interpolation=shared_metadata['interpolation'],
                        evaluation='shared_distance_hermite', shape=list(table.shape),
                        table_sha256=shared_metadata['table_sha256'],
                        validation_errors=shared_metadata['validation_errors'],
                        algorithm=cfg['algorithm'])
    elif distilled:
        from .neural_coupling import NeuralPairCost
        table = NeuralPairCost(oracle, delta, root_method="bisection" if primal else cfg.get("root_method", "newton"))
        curvature_check = table.check_curvature(config["distillation"]["ratio_grid_nodes"], cfg["curvature_tolerance"])
        metadata = dict(**block_fingerprint, interpolation="none", shape=list(table.shape),
                        evaluation="direct_mlp_autodiff", curvature_check=curvature_check,
                        root_method=None if primal else cfg.get("root_method", "newton"), algorithm=cfg["algorithm"])
    else:
        table_builder = build_table
        if cfg.get("backend") == "torch":
            from .cost_table_gpu import build_table_gpu
            table_builder = build_table_gpu
        table, metadata = table_builder(oracle, delta, cfg, config["pair"]["ratio_min"],
                                        config["pair"]["ratio_max"],
                                        run / "cost_cache" / f"{name}.npz", block_fingerprint)

    query_seconds = time.perf_counter() - qstart
    output = run / "couplings"
    existing_json, existing_npz = output / f"{name}.json", output / f"{name}.npz"
    if existing_json.exists() and existing_npz.exists():
        try:
            previous = json.loads(existing_json.read_text(encoding="utf-8"))
            previous_meta = previous.get("cost_metadata", {})
            same_fingerprint = (previous.get("status") == "converged" and
                all(previous_meta.get(key) == value for key, value in block_fingerprint.items()) and
                previous_meta.get("algorithm") == cfg["algorithm"])
            if same_fingerprint:
                with np.load(existing_npz) as saved:
                    same_indices = (np.array_equal(saved["source_indices"], i) and
                                    np.array_equal(saved["target_indices"], j))
                    valid_values = (same_indices and np.isfinite(saved["P"]).all() and
                                    np.isfinite(saved["Q"]).all())
                if valid_values:
                    last = previous["history"][-1]
                    record = dict(file=f"{name}.npz", interval=k, block=block,
                        status="converged", iterations=len(previous["history"]),
                        row_residual=last["row_relative_l1"],
                        column_residual=last["column_relative_l1"],
                        row_relative_tv=last["row_relative_tv"],
                        column_relative_tv=last["column_relative_tv"],
                        boundary_P_mass_fraction=last["boundary_P_mass_fraction"],
                        objective=last["objective"])
                    print(f"Reusing converged coupling: interval={k}, block={block + 1}/{source_blocks_count}", flush=True)
                    return record, query_seconds, float(previous.get("seconds", 0.0)), 0.0
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    # Resume a feasible point only when the exact same objective/table was
    # previously stopped by the outer iteration budget.
    initial_P = initial_Q = initial_log_P = initial_log_Q = None
    initial_step = 1.0
    if primal and existing_json.exists() and existing_npz.exists():
        try:
            previous = json.loads(existing_json.read_text(encoding="utf-8"))
            previous_meta = previous.get("cost_metadata", {})
            same_fingerprint = (previous.get("status") == "max_outer_iterations" and
                all(previous_meta.get(key) == value for key, value in block_fingerprint.items()) and
                previous_meta.get("algorithm") == cfg["algorithm"])
            if same_fingerprint:
                with np.load(existing_npz) as saved:
                    same_indices = (np.array_equal(saved["source_indices"], i) and
                                    np.array_equal(saved["target_indices"], j))
                    log_p0 = saved["log_P"].copy()
                    ratio0 = saved["r"].copy()
                valid_values = (same_indices and log_p0.shape == shape and ratio0.shape == shape and
                                np.isfinite(log_p0).all() and np.isfinite(ratio0).all() and
                                (ratio0 > 0).all())
                if valid_values:
                    initial_log_P, initial_log_Q = log_p0, log_p0 + np.log(ratio0)
                    initial_step = 2.0 * float(previous["history"][-1].get("step_size", 1.0))
                    print(f"Resuming feasible coupling iterate: interval={k}, block={block + 1}/{source_blocks_count}, "
                          f"outer step={previous['history'][-1]['iteration']}", flush=True)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    solver = solve_dual
    extra = {}
    if distilled or cfg.get("backend") == "torch":
        from .coupling_gpu import solve_dual_gpu
        solver = solve_dual_gpu
        extra["device"] = device
        extra["root_method"] = cfg.get("root_method", "newton")
    def progress(record):
        detail = (f", objective={record['objective']:.6g}, projected gradient={record['projected_gradient_relative_l1']:.3g}"
                  if "projected_gradient_relative_l1" in record else "")
        print(f"  interval={k} iteration={record['iteration']}, row TV={record['row_relative_l1']/2:.3g}, "
              f"column TV={record['column_relative_l1']/2:.3g}{detail}", flush=True)
        json_write(output / f"{name}_progress.json", record)
    def inner_progress(record):
        print(f"  interval={k} iteration={record['iteration']}, phase={record['phase']}, "
              f"inner step={record['inner_iteration']}, elapsed={record['seconds']:.1f}s", flush=True)
        json_write(output / f"{name}_live.json", record)
    if distilled or cfg.get("backend") == "torch":
        extra["inner_callback"] = inner_progress

    entropy_side = ("receiver_conditional" if path.cost_objective in {
        "tight_logmoment_upper_bound", "gaussian_reference_upper_bound", "diffusion_full_square_upper_bound"}
        else ("receiver" if path.cost_objective == "logmoment_upper_bound" else "source"))
    solve_started = time.perf_counter()
    if primal:
        from .primal_coupling_gpu import solve_primal_gpu
        result = solve_primal_gpu(source.weights[i], target.weights[j], table, path.sigma**2,
            config["pair"]["ratio_min"], config["pair"]["ratio_max"], cfg["marginal_tolerance"],
            max_iterations=cfg["max_outer_iterations"], stationarity_tolerance=cfg["stationarity_tolerance"],
            projection_tolerance=cfg.get("projection_tolerance"),
            allow_nonmonotone_fallback=cfg.get("allow_nonmonotone_fallback", False),
            nonmonotone_min_step_fraction=cfg.get("nonmonotone_min_step_fraction", 2**-12),
            callback=progress, device=device, entropy_side=entropy_side,
            initial_P=initial_P, initial_Q=initial_Q,
            initial_log_P=initial_log_P, initial_log_Q=initial_log_Q,
            initial_step=initial_step)
    else:
        result = solver(source.weights[i], target.weights[j], table, path.sigma ** 2,
            config["pair"]["ratio_min"], config["pair"]["ratio_max"],
            cfg["marginal_tolerance"], cfg["root_logmass_tolerance"],
            cfg["max_outer_iterations"], cfg["max_bisection_iterations"], callback=progress, **extra)
    solve_wall_seconds = time.perf_counter() - solve_started
    if distilled:
        metadata["monotonicity_warnings"] = list(table.warning_records.values())
        metadata["deployment_curvature"] = getattr(oracle, "deployment_curvature", None)
        metadata["global_conjugate_optimality_certified"] = False
        metadata["query_calls_by_derivative_order"] = table.query_calls
        metadata["query_conditions_by_derivative_order"] = table.query_conditions
    json_write(output / f"{name}.json", dict(status=result.status, history=result.history,
               seconds=result.seconds, cost_metadata=metadata))
    np.savez_compressed(output / f"{name}.npz", P=result.P, Q=result.Q, r=result.r, log_P=result.log_P,
        log_s=result.log_s, lam=result.lam, source_indices=i, target_indices=j,
        source_weights=source.weights[i], target_weights=target.weights[j],
        source_positions=source.x[i], target_positions=target.x[j], interval=k, partition=block)
    last = result.history[-1]
    manifest_record = dict(file=f"{name}.npz", interval=k, block=block, status=result.status,
        iterations=len(result.history), row_residual=last["row_relative_l1"],
        column_residual=last["column_relative_l1"], row_relative_tv=last["row_relative_l1"] / 2,
        column_relative_tv=last["column_relative_l1"] / 2,
        boundary_P_mass_fraction=last["boundary_P_mass_fraction"], objective=last["objective"])
    print(f"  {result.status}: interval={k}, block={block + 1}/{source_blocks_count}, "
          f"iterations={len(result.history)}, row={last['row_relative_l1']:.2g}, "
          f"column={last['column_relative_l1']:.2g}", flush=True)
    if result.status != "converged":
        raise OracleError("coupling_not_converged", dict(interval=k, block=block, last=last))
    return manifest_record, query_seconds, result.seconds, solve_wall_seconds


def solve_couplings(config, run):
    device = configure(config)
    run = init_run(run, config)
    snapshot = snapshots_for(config)
    path = load_path(run, config, device)
    cfg = config["coupling"]
    primal = cfg["algorithm"] == "projected_mirror_descent"
    shared = cfg.get("cost_evaluation") == "shared_distance_hermite"
    output = run / "couplings"
    output.mkdir(exist_ok=True)
    started = time.perf_counter()
    update_stage(run, "couplings", "running")
    try:
        distilled = config.get("distillation", {}).get("enabled", False)
        if shared and (distilled or not primal):
            raise ValueError("shared path table currently requires primal coupling with distillation disabled")
        if primal and not (distilled or shared):
            raise ValueError("direct primal coupling currently requires the distilled network")
        if distilled:
            from .cost_distillation import load_distilled_oracle
            oracle = load_distilled_oracle(config, run, path, device)
        else:
            (run / "cost_cache").mkdir(exist_ok=True)
            oracle = select_quadrature(path, snapshot, config, run)
        fingerprint = dict(cost_objective=path.cost_objective,
                           entropy_side="receiver_conditional" if path.cost_objective in {"tight_logmoment_upper_bound", "gaussian_reference_upper_bound", "diffusion_full_square_upper_bound"} else ("receiver" if path.cost_objective == "logmoment_upper_bound" else "source"),
                           path_checkpoint_sha256=file_hash(run / "pair_path.pt"), sigma=path.sigma,
                           alpha=path.alpha, control_mode=path.control_mode, quadrature_nodes=oracle.nodes,
                           partition_seed=cfg["partition_seed"], mode=cfg["mode"],
                           table_atol=cfg["table_atol"], table_rtol=cfg["table_rtol"])
        fingerprint["cost_oracle"] = "distilled_mlp" if distilled else "path_quadrature"
        if path.cost_objective == "tight_logmoment_upper_bound":
            fingerprint['logmoment_nodes']=path.logmoment_nodes
        if path.cost_objective in {'gaussian_reference_upper_bound', 'diffusion_full_square_upper_bound'}:
            fingerprint['reference_second_moment']=path.reference_second_moment
        if distilled:
            fingerprint["cost_network_sha256"] = oracle.checkpoint_sha256
        manifests = []
        query_seconds, solve_seconds = 0.0, 0.0
        if shared:
            import torch
            from .distance_cost_table import build_distance_table
            qstart = time.perf_counter()
            points = [torch.as_tensor(s.x, dtype=torch.float64, device=device) for s in snapshot]
            # Compute the global distance range in bounded target chunks.  The
            # largest official pair support is hundreds of millions of pairs;
            # a single dense cdist output would need several GB even though the
            # final shared cost table is only a few MB.
            distance_max = 0.0
            for a, b in zip(points[:-1], points[1:]):
                for start in range(0, len(b), 1024):
                    distance_max = max(distance_max, float(torch.cdist(a, b[start:start + 1024]).max()))
            distance_max = max(distance_max*(1+1e-10), torch.finfo(torch.float64).eps)
            shared_table, shared_metadata = build_distance_table(oracle, distance_max, config, run)
            query_seconds += time.perf_counter()-qstart
        import torch
        worker_count = int(cfg.get("parallel_workers", 1))
        if worker_count < 1:
            raise ValueError("coupling.parallel_workers must be at least 1")
        rng = np.random.default_rng(cfg["partition_seed"])
        interval_blocks = []
        for k in range(len(snapshot) - 1):
            source, target = snapshot[k], snapshot[k + 1]
            source_blocks, target_blocks = partition_indices(len(source.x), len(target.x), cfg, rng)
            interval_blocks.append([(k, block, i, j, len(source_blocks))
                                    for block, (i, j) in enumerate(zip(source_blocks, target_blocks))])

        def run_block(spec):
            k, block, i, j, block_count = spec
            try:
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        stream = torch.cuda.Stream(device=device)
                        with torch.cuda.stream(stream):
                            answer = _solve_one_block(config, run, snapshot, path, device, oracle,
                                shared_table if shared else None, shared_metadata if shared else None,
                                points if shared else None, fingerprint, k, block, i, j, block_count,
                                cfg, primal, shared, distilled)
                        stream.synchronize()
                        return answer
                return _solve_one_block(config, run, snapshot, path, device, oracle,
                    shared_table if shared else None, shared_metadata if shared else None,
                    points if shared else None, fingerprint, k, block, i, j, block_count,
                    cfg, primal, shared, distilled)
            except OracleError as exc:
                if exc.status == "primal_line_search_failed":
                    exc.details.setdefault("interval", k)
                    exc.details.setdefault("block", block)
                raise

        if device.type == "cuda":
            # The shared distance grid and observation tensors are prepared on
            # the main stream. Complete those writes before workers read them.
            torch.cuda.synchronize(device)
        block_wall_started = time.perf_counter()
        # Blocks are independent across both minibatches and time intervals.
        # Experiment 2 can opt into a single queue to keep workers occupied
        # when block runtimes vary. The default preserves interval-wise
        # scheduling for all existing experiments.
        global_block_queue = bool(cfg.get("global_block_queue", False))
        schedules = ([ [spec for specs in interval_blocks for spec in specs] ]
                     if global_block_queue else interval_blocks)
        failures = []
        for specs in schedules:
            if not specs:
                continue
            with ThreadPoolExecutor(max_workers=min(worker_count, len(specs)),
                                    thread_name_prefix="ruot-coupling") as pool:
                futures = [pool.submit(run_block, spec) for spec in specs]
                for future in as_completed(futures):
                    try:
                        record, q_seconds, s_seconds, _ = future.result()
                        manifests.append(record)
                        query_seconds += q_seconds
                        solve_seconds += s_seconds
                    except Exception as exc:
                        failures.append(exc)
                        for pending in futures:
                            if pending is not future:
                                pending.cancel()
                if failures:
                    break
        if failures:
            raise failures[0]
        block_wall_seconds = time.perf_counter() - block_wall_started
        manifests.sort(key=lambda record: (record["interval"], record["block"]))
        json_write(output / "manifest.json", dict(status="converged", mode=cfg["mode"],
                   partition_order="dataset_default", minibatch_blocks=cfg.get("minibatch_blocks"),
                   partition_seed=cfg["partition_seed"],
                   parallel_workers=worker_count,
                   marginal_tolerance=cfg["marginal_tolerance"],
                   stationarity_tolerance=cfg.get("stationarity_tolerance"),
                   projection_tolerance=cfg.get("projection_tolerance"),
                   allow_nonmonotone_fallback=cfg.get("allow_nonmonotone_fallback", False),
                   nonmonotone_min_step_fraction=cfg.get("nonmonotone_min_step_fraction", 2**-12),
                   global_block_queue=global_block_queue,
                   path_checkpoint_sha256=fingerprint["path_checkpoint_sha256"], intervals=manifests,
                   cost_oracle=fingerprint["cost_oracle"], algorithm=cfg["algorithm"],
                   cost_evaluation="shared_distance_hermite" if shared else ("direct_mlp_autodiff" if distilled else "cubic_hermite_in_r"),
                   table_sha256=shared_metadata['table_sha256'] if shared else None,
                   cost_network_sha256=fingerprint.get("cost_network_sha256"),
                   query_seconds=query_seconds, solve_seconds=solve_seconds,
                   block_wall_seconds=block_wall_seconds))
        update_stage(run, "couplings", "completed", mode=cfg["mode"], blocks=len(manifests),
                     backend=cfg.get("backend", "numpy"), device=str(device) if cfg.get("backend") == "torch" else "cpu",
                     marginal_tolerance=cfg["marginal_tolerance"], tolerance_type="relative_l1", relative_tv_tolerance=cfg["marginal_tolerance"] / 2,
                     stationarity_tolerance=cfg.get("stationarity_tolerance"),
                     projection_tolerance=cfg.get("projection_tolerance"),
                     allow_nonmonotone_fallback=cfg.get("allow_nonmonotone_fallback", False),
                     nonmonotone_min_step_fraction=cfg.get("nonmonotone_min_step_fraction", 2**-12),
                     global_block_queue=global_block_queue,
                     parallel_workers=worker_count,
                     query_seconds=query_seconds, solve_seconds=solve_seconds,
                     block_wall_seconds=block_wall_seconds, seconds=time.perf_counter() - started)
        return True
    except OracleError as exc:
        json_write(run / "coupling_failure.json", dict(status=exc.status, details=exc.details))
        update_stage(run, "couplings", exc.status, seconds=time.perf_counter() - started, details=exc.details)
        print(f"Coupling stopped: {exc.status}; failure details saved.", flush=True)
        return False
    except Exception as exc:
        json_write(run / "coupling_failure.json", dict(status="unexpected_error", error=repr(exc)))
        update_stage(run, "couplings", "failed", seconds=time.perf_counter() - started, error=repr(exc))
        print(f"Coupling failed: {type(exc).__name__}: {exc}", flush=True)
        return False

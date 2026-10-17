from pathlib import Path
import math
import time

import numpy as np
import torch
from .data import PairIndex
from .pair_cost import CostOracle, finite_pair_integrand
from .runtime import configure, init_run, json_write, log_json, make_path, snapshots_for, update_stage


PAIR_TRAINING_VERSION = 2


def sample_pair_conditions(index, cfg, n, device):
    """Uniform draws over every adjacent pair, with an explicit ratio measure."""
    ids = torch.randint(index.size, (n,), device=device)
    draw = torch.rand((n, 1), dtype=torch.float64, device=device)
    mode = cfg.get("ratio_sampling", "uniform")
    if mode == "uniform":
        ell = (cfg["ratio_min"] + (cfg["ratio_max"] - cfg["ratio_min"]) * draw).log()
    elif mode == "log_uniform":
        lower, upper = math.log(cfg["ratio_min"]), math.log(cfg["ratio_max"])
        ell = lower + (upper - lower) * draw
    else:
        raise ValueError(f"unknown ratio_sampling: {mode}")
    return ids, index.delta_tensor(ids), ell


def _train_steps(config, run, index, path, optimizer, checkpoint, validation_delta, validation_ratio):
    cfg = config["pair"]
    device = index.positions.device
    steps = cfg["training_steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("pair.training_steps must be a positive integer")
    if cfg.get("ratio_sampling", "uniform") not in {"uniform", "log_uniform"}:
        raise ValueError("pair.ratio_sampling must be uniform or log_uniform")
    checkpoint = checkpoint or {}
    first_step = checkpoint.get("updates", 0)
    prior_seconds = checkpoint.get("training_seconds", 0.0)
    prior_epochs = checkpoint.get("completed_epochs", 0)
    visits = checkpoint.get("candidate_pair_visits", prior_epochs * index.size)
    time_samples = checkpoint.get("condition_time_samples", visits * cfg["times_per_condition"])
    visited = checkpoint.get("visited_pair_ids")
    if visited is None:
        visited = torch.full((index.size,), prior_epochs > 0, dtype=torch.bool, device=device)
    else:
        visited = visited.to(device=device, dtype=torch.bool)
        if visited.shape != (index.size,):
            raise ValueError("checkpoint candidate support differs from current data")
    checkpoint_every = cfg.get("checkpoint_every_steps", 1000)
    log_every = cfg.get("log_every_steps", 100)
    validation_every = cfg.get("validation_every_steps", 0)
    if checkpoint_every <= 0 or log_every <= 0 or validation_every < 0:
        raise ValueError("invalid step logging/checkpoint/validation interval")
    started, last_print = time.perf_counter(), time.perf_counter()
    sums = torch.zeros(6 if path.cost_objective.endswith("upper_bound") else 5, dtype=torch.float64, device=device)
    mass_min = torch.full((), float("inf"), dtype=torch.float64, device=device)
    mass_max = torch.zeros((), dtype=torch.float64, device=device)
    finite = torch.ones((), dtype=torch.bool, device=device)
    window_steps = 0
    update_stage(run, "pair", "running", training_mode="steps", completed_steps=first_step,
                 candidate_pairs=index.size, candidate_pair_visits=visits)
    print(f"Pair pretraining: device={device}, candidate_pairs={index.size}, steps={steps}, "
          f"ratio_sampling={cfg.get('ratio_sampling', 'uniform')}", flush=True)
    for step in range(first_step, steps):
        ids, delta, ell = sample_pair_conditions(index, cfg, cfg["conditions_per_batch"], device)
        visited[ids] = True
        visits += len(ids)
        nt = cfg["times_per_condition"]
        delta, ell = delta.repeat_interleave(nt, 0), ell.repeat_interleave(nt, 0)
        time_samples += len(delta)
        eps = torch.finfo(torch.float64).eps
        t = eps / 2 + (1 - eps) * torch.rand((len(delta), 1), dtype=torch.float64, device=device)
        state = path(t, delta, ell, create_graph=True)
        value, parts = finite_pair_integrand(state, path.sigma, path.alpha, path.control_mode, True)
        loss = value.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(path.parameters(), 10., error_if_nonfinite=False)
        finite &= torch.isfinite(loss.detach()) & torch.isfinite(grad)
        optimizer.step()
        sums += torch.stack([loss.detach()] + [part.detach().mean() for part in parts.values()])
        mass_min = torch.minimum(mass_min, state["mass"].detach().min())
        mass_max = torch.maximum(mass_max, state["mass"].detach().max())
        window_steps += 1
        now = time.perf_counter()
        save = (step + 1) % checkpoint_every == 0 or step + 1 == steps
        log = (step + 1) % log_every == 0 or step == first_step or now - last_print >= 20 or save
        if log:
            if not finite.item():
                update_stage(run, "pair", "nonfinite", completed_steps=step + 1)
                raise FloatingPointError("nonfinite finite-integrand training loss/gradient")
            averages = (sums / window_steps).cpu().tolist()
            unique = int(visited.sum().item())
            record = dict(step=step + 1, update=step + 1, training_mode="steps",
                          pair_training_version=PAIR_TRAINING_VERSION,
                          ratio_sampling=cfg.get("ratio_sampling", "uniform"),
                          candidate_pair_visits=visits, unique_pairs_seen=unique,
                          condition_time_samples=time_samples,
                          unique_coverage_fraction=unique / index.size,
                          loss=averages[0], loss_terms=dict(zip(parts, averages[1:])),
                          window_steps=window_steps, mass_min=mass_min.item(), mass_max=mass_max.item(),
                          grad_norm=grad.item(), elapsed_seconds=prior_seconds + now - started)
            log_json(run / "pair_steps.jsonl", record)
            print(f"  step {step + 1}/{steps}: loss={averages[0]:.5g}, unique pairs={unique}/{index.size}", flush=True)
            sums.zero_()
            mass_min.fill_(float("inf"))
            mass_max.zero_()
            window_steps, last_print = 0, now
        if validation_every and ((step + 1) % validation_every == 0 or step + 1 == steps):
            log_json(run / "pair_validation.jsonl", dict(step=step + 1,
                     **validate(path, validation_delta, validation_ratio, config)))
        if save:
            # Logging at every save has already checked accumulated finiteness.
            cp_file = run / "pair_path.pt"
            temp = cp_file.with_suffix(".tmp")
            torch.save(dict(model=path.state_dict(), optimizer=optimizer.state_dict(),
                            completed_epochs=prior_epochs, updates=step + 1, training_mode="steps",
                            completed_steps=step + 1, candidate_pair_visits=visits,
                            condition_time_samples=time_samples,
                            visited_pair_ids=visited, pair_training_version=PAIR_TRAINING_VERSION,
                            numpy_rng=checkpoint.get("numpy_rng", np.random.default_rng(config["seed"]).bit_generator.state),
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                            training_seconds=prior_seconds + time.perf_counter() - started, config=config), temp)
            temp.replace(cp_file)
            update_stage(run, "pair", "running", training_mode="steps", completed_steps=step + 1,
                         candidate_pair_visits=visits, unique_pairs_seen=unique,
                         condition_time_samples=time_samples,
                         unique_coverage_fraction=unique / index.size)
    unique = int(visited.sum().item())
    update_stage(run, "pair", "completed", training_mode="steps", completed_steps=max(first_step, steps),
                 candidate_pairs=index.size, candidate_pair_visits=visits, unique_pairs_seen=unique,
                 condition_time_samples=time_samples,
                 unique_coverage_fraction=unique / index.size,
                 seconds=prior_seconds + time.perf_counter() - started)
    return path


def validate(path, delta, ratio, config):
    c = config["pair"]
    path.freeze()
    coarse = CostOracle(path, c["quadrature_nodes"])
    fine = CostOracle(path, 2 * c["quadrature_nodes"])
    a = coarse(delta, ratio)
    if path.cost_objective == "tight_logmoment_upper_bound":
        path.logmoment_nodes=64
    b = fine(delta, ratio)
    if path.cost_objective == "tight_logmoment_upper_bound":
        path.logmoment_nodes=32
    errors = [float(np.max(np.abs(x - y))) for x, y in zip(a, b)]
    t = torch.tensor([[0.0], [1e-10], [0.5], [1 - 1e-10], [1.0]], device=delta.device, dtype=torch.float64)
    state = path(t, delta[:1].expand(5, -1), torch.zeros_like(t))
    if path.cost_objective.endswith("upper_bound"):
        state = path(t[1:-1], delta[:1].expand(3, -1), torch.zeros_like(t[1:-1]))
    value = finite_pair_integrand(state, path.sigma, path.alpha, path.control_mode)
    result = dict(cost_objective=path.cost_objective, endpoint_check="open_interval_log_integrable" if path.cost_objective.endswith("upper_bound") else "closed_interval", cost_mean=float(np.mean(b[0])), min_curvature=float(np.min(b[2])),
                  quadrature_max_errors=dict(cost=errors[0], first_derivative=errors[1], second_derivative=errors[2]),
                  endpoint_integrand_finite=bool(torch.isfinite(value).all()),
                  integrand_endpoints=value.detach().cpu().numpy().reshape(-1).tolist())
    for p in path.parameters():
        p.requires_grad_(True)
    path.train()
    return result


def train_pairs(config, run):
    device = configure(config)
    run = init_run(run, config)
    snapshots = snapshots_for(config)
    index = PairIndex(snapshots, device)
    cfg = config["pair"]
    path = make_path(config, device)
    optimizer = torch.optim.Adam(path.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(config["seed"])
    vrng = np.random.default_rng(config["seed"] + 100)
    validation_ids = vrng.integers(0, index.size, cfg["validation_conditions"])
    validation_delta = index.delta(validation_ids)
    validation_ratio = np.exp(vrng.uniform(math.log(cfg["ratio_min"]), math.log(cfg["ratio_max"]), len(validation_ids)))
    first_epoch, completed_updates = 0, 0
    prior_seconds = 0.0
    checkpoint_file = run / "pair_path.pt"
    cp = None
    if checkpoint_file.exists():
        cp = torch.load(checkpoint_file, map_location=device, weights_only=False)
        path.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        first_epoch, completed_updates = cp["completed_epochs"], cp["updates"]
        rng.bit_generator.state = cp["numpy_rng"]
        torch.set_rng_state(cp["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in cp["cuda_rng"]])
        prior_seconds = cp.get("training_seconds", 0.0)
    json_write(run / "data_metadata.json", dict(counts_used=[len(s.x) for s in snapshots],
               original_counts=[s.original_count for s in snapshots], masses=[s.mass for s in snapshots],
               pair_counts=index.counts, candidate_pairs=index.size, coordinates="original unscaled",
               downsampled=config.get("max_observed_points") is not None))
    mode = cfg.get("training_mode", "coverage")
    if mode == "steps":
        return _train_steps(config, run, index, path, optimizer, cp, validation_delta, validation_ratio)
    if mode != "coverage":
        raise ValueError("pair.training_mode must be coverage or steps")
    if cp is not None and cp.get("training_mode") == "steps":
        raise ValueError("cannot resume a steps checkpoint as full-coverage epochs")
    started = time.perf_counter()
    last_print = 0.0
    update_stage(run, "pair", "running", completed_epochs=first_epoch, candidate_pairs=index.size)
    print(f"Pair pretraining: device={device}, candidate_pairs={index.size}, epochs={cfg['coverage_epochs']}", flush=True)
    for epoch in range(first_epoch, cfg["coverage_epochs"]):
        epoch_started = time.perf_counter()
        # The existing torch/CUDA checkpoint RNG states now also cover the
        # full-coverage permutation and continuous log-ratio draws.
        permutation = torch.randperm(index.size, device=device)
        sums = torch.zeros(6 if path.cost_objective.endswith("upper_bound") else 5, dtype=torch.float64, device=device)
        seen = 0
        mass_min = torch.full((), float("inf"), dtype=torch.float64, device=device)
        mass_max = torch.zeros((), dtype=torch.float64, device=device)
        finite = torch.ones((), dtype=torch.bool, device=device)
        for start in range(0, index.size, cfg["conditions_per_batch"]):
            ids = permutation[start:start + cfg["conditions_per_batch"]]
            delta = index.delta_tensor(ids)
            if epoch < 3:
                ell = torch.full((len(ids), 1), math.log([cfg["ratio_min"], 1.0, cfg["ratio_max"]][epoch]),
                                 dtype=torch.float64, device=device)
            else:
                lower, upper = math.log(cfg["ratio_min"]), math.log(cfg["ratio_max"])
                ell = lower + (upper - lower) * torch.rand((len(ids), 1), dtype=torch.float64, device=device)
            nt = cfg["times_per_condition"]
            delta, ell = delta.repeat_interleave(nt, 0), ell.repeat_interleave(nt, 0)
            # Half-ULP shift gives an open interval without discarding endpoint neighborhoods.
            eps = torch.finfo(torch.float64).eps
            t = eps / 2 + (1 - eps) * torch.rand((len(delta), 1), dtype=torch.float64, device=device)
            state = path(t, delta, ell, create_graph=True)
            value, parts = finite_pair_integrand(state, path.sigma, path.alpha, path.control_mode, True)
            loss = value.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(path.parameters(), 10.0, error_if_nonfinite=False)
            finite &= torch.isfinite(loss.detach()) & torch.isfinite(grad_norm)
            optimizer.step()
            n = len(ids)
            sums += n * torch.stack([loss.detach()] + [part.detach().mean() for part in parts.values()])
            seen += n
            mass_min = torch.minimum(mass_min, state["mass"].detach().min())
            mass_max = torch.maximum(mass_max, state["mass"].detach().max())
            completed_updates += 1
            now = time.perf_counter()
            if now - last_print >= 20:
                if not finite.item():
                    update_stage(run, "pair", "nonfinite", epoch=epoch + 1, update=completed_updates)
                    raise FloatingPointError("nonfinite finite-integrand training loss/gradient")
                record = dict(epoch=epoch + 1, update=completed_updates, pairs_in_epoch=seen,
                              loss=loss.item(), grad_norm=grad_norm.item(), pair_training_version=PAIR_TRAINING_VERSION,
                              elapsed_seconds=prior_seconds + now - started)
                print(f"  epoch {epoch + 1}: {seen}/{index.size} pairs, loss={loss.item():.5g}", flush=True)
                log_json(run / "pair_batches.jsonl", record)
                last_print = now
        if not finite.item():
            update_stage(run, "pair", "nonfinite", epoch=epoch + 1, update=completed_updates)
            raise FloatingPointError("nonfinite finite-integrand training loss/gradient")
        averages = (sums / seen).cpu().tolist()
        report = dict(epoch=epoch + 1, pairs_seen=seen, coverage_fraction=seen / index.size,
                      pair_training_version=PAIR_TRAINING_VERSION,
                      cumulative_pair_visits=(epoch + 1) * index.size,
                      ratio_schedule="anchor" if epoch < 3 else "continuous_log_uniform",
                      loss=averages[0], loss_terms=dict(zip(parts, averages[1:])),
                      mass_min=mass_min.item(), mass_max=mass_max.item(), last_grad_norm=grad_norm.item(),
                      epoch_seconds=time.perf_counter() - epoch_started)
        if (epoch + 1) % cfg.get("validation_every", 1) == 0 or epoch + 1 == cfg["coverage_epochs"]:
            report["validation"] = validate(path, validation_delta, validation_ratio, config)
        log_json(run / "pair_history.jsonl", report)
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds = prior_seconds + time.perf_counter() - started
        temp = checkpoint_file.with_suffix(".tmp")
        torch.save(dict(model=path.state_dict(), optimizer=optimizer.state_dict(), completed_epochs=epoch + 1,
                        pair_training_version=PAIR_TRAINING_VERSION,
                        updates=completed_updates, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        training_seconds=training_seconds, config=config), temp)
        temp.replace(checkpoint_file)
        update_stage(run, "pair", "running", completed_epochs=epoch + 1, coverage_fraction=1.0,
                     cumulative_pair_visits=(epoch + 1) * index.size, seconds=training_seconds)
        print(f"Epoch {epoch + 1} completed: loss={report['loss']:.5g}, seconds={report['epoch_seconds']:.1f}", flush=True)
    update_stage(run, "pair", "completed", completed_epochs=cfg["coverage_epochs"], candidate_pairs=index.size,
                 cumulative_pair_visits=cfg["coverage_epochs"] * index.size,
                 seconds=prior_seconds + time.perf_counter() - started)
    return path

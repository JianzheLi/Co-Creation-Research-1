from pathlib import Path
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np
import torch
import yaml
from .pair_path import PairPath

ROOT = Path(__file__).resolve().parents[1]


def merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent), config)
    method = config.get("experiment_method", "ruot")
    if method not in {"ruot", "wfr_fm_bridge", "wfr_fm_bridge_exp_mass"}:
        raise ValueError(f"unsupported experiment_method: {method}")
    if not math.isfinite(config["growth_penalty"]) or config["growth_penalty"] <= 0:
        raise ValueError("growth_penalty must be finite and positive")
    objective = config["pair"].get("cost_objective", "renormalized_surrogate")
    if objective not in {"renormalized_surrogate", "logmoment_upper_bound", "tight_logmoment_upper_bound", "gaussian_reference_upper_bound", "diffusion_full_square_upper_bound"}:
        raise ValueError("unknown pair cost objective")
    if objective.endswith("upper_bound"):
        if method != "ruot" or config["coupling"]["algorithm"] != "projected_mirror_descent":
            raise ValueError("upper bound requires RUOT projected mirror descent")
    times = config["model_times"]
    if (not isinstance(times, list) or len(times) < 2 or times[0] != 0
            or any(b - a != 1 for a, b in zip(times[:-1], times[1:]))):
        raise ValueError("model_times must start at zero and use consecutive unit intervals")
    if config["generation"]["particle_birth_death"]:
        raise ValueError("fixed weighted particles required")
    if method == "ruot":
        if config["control_mode"] != "quadratic_minimal":
            raise ValueError("main experiments require quadratic_minimal controls")
        if config["coupling"]["algorithm"] not in {"row_normalization_column_root", "projected_mirror_descent"}:
            raise ValueError("unsupported coupling algorithm")
        reference = "source_target_product" if objective.endswith("upper_bound") else "ones"
        if config["coupling"]["entropy_coefficient"] != "sigma_squared" or config["coupling"]["reference_weights"] != reference:
            raise ValueError(f"epsilon=sigma^2 and reference_weights={reference} required")
    elif config["coupling"].get("method") != "wfr_fm_oet":
        raise ValueError("WFR-FM heuristic requires the WFR optimal-entropy-transport coupling")
    if config["pair"]["dtype"] != "float64" or config["matching"]["dtype"] != "float32":
        raise ValueError("pair/coupling must be float64; matching must be float32")
    if config["matching"]["interval_sampling"] != "uniform":
        raise ValueError("only uniform interval sampling is implemented")
    if config["generation"].get("carry_log_weights") is not True:
        raise ValueError("unnormalized log weights must be carried")
    ev = config["evaluation"]
    if ev["library"] != "POT" or ev["transport_solver"] != "emd2" or not ev["normalize_for_W1_W2"] or not ev["report_mass_separately"]:
        raise ValueError("POT probability EMD and separate raw mass errors required")
    if objective in {'gaussian_reference_upper_bound', 'diffusion_full_square_upper_bound'}:
        from .gaussian_reference_bound import observed_second_moment
        derived = observed_second_moment(snapshots_for(config))
        saved = config['pair'].get('reference_second_moment', derived)
        if not math.isclose(saved, derived, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('reference_second_moment is derived from the observations, not adjustable')
        config['pair']['reference_second_moment'] = derived
    return config


def configure(config):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config.get("cpu_threads", 4))
    # Float64 pair computations and reproducible float32 feedback computations.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def log_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def init_run(run, config):
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    saved = run / "config.yaml"
    if saved.exists() and yaml.safe_load(saved.read_text(encoding="utf-8")) != config:
        raise ValueError("run directory already contains a different configuration")
    saved.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if not (run / "environment.json").exists():
        versions = {}
        for name in ("torch", "numpy", "scipy", "POT", "pandas", "PyYAML", "matplotlib", "pytest"):
            versions[name] = importlib.metadata.version(name)
        json_write(run / "environment.json", dict(python=sys.version, executable=sys.executable,
                   platform=platform.platform(), versions=versions,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   seed=config["seed"], tf32=False, MKL_THREADING_LAYER=os.environ.get("MKL_THREADING_LAYER")))
        packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata["Name"])
        (run / "environment-lock.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    return run


def update_stage(run, stage, status, **details):
    path = Path(run) / "status.json"
    value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    value[stage] = dict(status=status, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details)
    json_write(path, value)


def make_path(config, device):
    c = config["pair"]
    return PairPath(config["dimension"], config["model_sigma"], config["growth_penalty"],
                    c["hidden_width"], c["hidden_layers"], c["correction_bounds"],
                    c.get("delta_scale", 1), config["control_mode"],
                    c.get("parameterization", "diagonal"),
                    c.get("cost_objective", "renormalized_surrogate"),
                    c.get('reference_second_moment')).to(device=device, dtype=torch.float64)


def load_path(run, config, device):
    path = make_path(config, device)
    checkpoint = torch.load(Path(run) / "pair_path.pt", map_location=device, weights_only=False)
    saved_objective = checkpoint.get("config", {}).get("pair", {}).get("cost_objective", "renormalized_surrogate")
    if saved_objective != path.cost_objective:
        raise ValueError("pair checkpoint cost objective differs from the requested algorithm")
    path.load_state_dict(checkpoint["model"])
    radial_file=Path(run)/'radial_quadrature_validation.json'
    if path.cost_objective == 'tight_logmoment_upper_bound' and radial_file.exists():
        import json
        path.logmoment_nodes=json.loads(radial_file.read_text(encoding='utf-8'))['selected_nodes']
    return path.freeze()


def snapshots_for(config, full=False):
    from .data import load_csv_snapshots, load_gene
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    max_points = None if full else config.get("max_observed_points")
    if config.get("dataset") == "official_deepruot_simulation_gene":
        return load_gene(data_path, max_points, config["seed"])
    return load_csv_snapshots(data_path, config["model_times"], max_points, config["seed"],
                              config.get("data_url"), config.get("data_sha256"), config["dimension"])

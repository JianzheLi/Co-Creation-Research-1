"""Create reproducible low-dimensional benchmarks and matched RUOT/WFR configs.

The generated data are finite snapshots (rather than paired training paths),
with low-dimensional support and explicit observation counts that the loader
interprets as unnormalised snapshot masses.
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lowdim_action_benchmarks"
CONFIGS = ROOT / "configs" / "lowdim_action_benchmarks"
TIMES = np.arange(4, dtype=np.float64)
COUNTS = {
    "gaussian_translation": [128, 144, 160, 176],
    "gaussian_breathing": [160, 160, 160, 160],
    "gmm_split": [128, 144, 160, 176],
    "gmm_rotate_reweight": [176, 160, 144, 128],
    "double_well_sde": [128, 144, 160, 176],
    "ou_sde_5d": [128, 144, 160, 176],
    "two_moons_morph": [128, 144, 160, 176],
}
ALPHA = 2.0
ALPHA_SWEEPS = {"gmm_split": (2.0, 10.0), "double_well_sde": (2.0, 10.0)}
DATA_SEED = 20261012


def truncated_gaussian(rng, mean, covariance, n, whiten_radius=2.2):
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    chol = np.linalg.cholesky(covariance)
    kept = []
    while sum(len(x) for x in kept) < n:
        z = rng.normal(size=(max(2 * n, 256), len(mean)))
        z = z[np.linalg.norm(z, axis=1) <= whiten_radius]
        kept.append(mean + z @ chol.T)
    return np.concatenate(kept, axis=0)[:n]


def snapshots_gaussian_translation(rng):
    counts = COUNTS["gaussian_translation"]
    means = [(-1.0, -0.15), (-0.35, 0.12), (0.35, 0.28), (1.0, 0.10)]
    covariance = np.diag([0.13**2, 0.10**2])
    return [truncated_gaussian(rng, m, covariance, n) for m, n in zip(means, counts)]


def snapshots_gaussian_breathing(rng):
    counts = COUNTS["gaussian_breathing"]
    scales = [0.10, 0.22, 0.36, 0.19]
    return [truncated_gaussian(rng, (0.0, 0.0), np.eye(2) * s**2, n, 2.1)
            for s, n in zip(scales, counts)]


def snapshots_gmm_split(rng):
    counts = COUNTS["gmm_split"]
    centers = [0.08, 0.30, 0.58, 0.82]
    covariance = np.diag([0.12**2, 0.11**2])
    output = []
    for center, n in zip(centers, counts):
        labels = rng.integers(0, 2, size=n)
        signs = np.where(labels == 0, -1.0, 1.0)
        means = np.column_stack([signs * center, np.zeros(n)])
        z = rng.normal(size=(n, 2))
        z = np.clip(z, -2.0, 2.0)
        output.append(means + z * np.array([0.12, 0.11]))
    return output


def snapshots_gmm_rotate_reweight(rng):
    counts = COUNTS["gmm_rotate_reweight"]
    angles = [0.0, math.pi / 6, math.pi / 3, math.pi / 2]
    p_left = [0.8, 0.65, 0.35, 0.2]
    radius = 0.62
    output = []
    for n, angle, probability in zip(counts, angles, p_left):
        center = radius * np.array([math.cos(angle), math.sin(angle)])
        centers = np.stack([center, -center])
        labels = (rng.random(n) >= probability).astype(np.int64)
        noise = rng.normal(size=(n, 2)) * np.array([0.12, 0.10])
        noise = np.clip(noise, -0.24, 0.24)
        output.append(centers[labels] + noise)
    return output


def snapshots_two_moons_morph(rng):
    counts = COUNTS["two_moons_morph"]
    first_moon_probability = [0.50, 0.58, 0.42, 0.62]
    lower_offsets = [0.50, 0.42, 0.34, 0.26]
    angles = [0.0, 0.10, 0.22, 0.35]
    output = []
    for n, probability, lower_offset, angle in zip(counts, first_moon_probability,
                                                    lower_offsets, angles):
        first = rng.random(n) < probability
        theta = rng.uniform(0, math.pi, size=n)
        points = np.empty((n, 2), dtype=np.float64)
        points[first, 0] = np.cos(theta[first])
        points[first, 1] = np.sin(theta[first])
        points[~first, 0] = 1 - np.cos(theta[~first])
        points[~first, 1] = lower_offset - np.sin(theta[~first])
        points = 0.55 * points + rng.normal(size=(n, 2)) * 0.035
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        output.append(points @ rotation.T)
    return output


def simulate_sde(n, dimension, drift, diffusion, initial, dt, rng, projector):
    state = np.asarray(initial(rng, n), dtype=np.float64)
    observations = [state.copy()]
    steps_per_unit = round(1 / dt)
    for _ in range(3):
        for _ in range(steps_per_unit):
            state = state + drift(state) * dt + diffusion * math.sqrt(dt) * rng.normal(size=(n, dimension))
            state = projector(state)
        observations.append(state.copy())
    return observations


def snapshots_double_well_sde(rng):
    counts = COUNTS["double_well_sde"]
    def drift(x):
        return np.column_stack([-x[:, 0] * (x[:, 0] ** 2 - 1.0), -0.7 * x[:, 1]])
    def initial(local_rng, n):
        return local_rng.normal(size=(n, 2)) * np.array([0.18, 0.12])
    def projector(x):
        x[:, 0] = np.clip(x[:, 0], -1.25, 1.25)
        x[:, 1] = np.clip(x[:, 1], -0.85, 0.85)
        return x
    trajectories = simulate_sde(max(counts), 2, drift, 0.25, initial, 0.005, rng, projector)
    return [x[:n] for x, n in zip(trajectories, counts)]


def snapshots_ou_sde(rng):
    counts = COUNTS["ou_sde_5d"]
    means = [np.array([-0.72 + 0.48 * t, 0.18 * math.sin(t), 0.0, 0.0, 0.0])
             for t in TIMES]
    n = max(counts)
    state = rng.normal(size=(n, 5)) * 0.16 + means[0]
    output = [state.copy()]
    dt = 0.01
    for interval in range(3):
        for step in range(round(1 / dt)):
            t = interval + step * dt
            fraction = (t + dt) - interval
            center = means[interval] * (1 - fraction) + means[interval + 1] * fraction
            state = state + (-0.9 * (state - center)) * dt + 0.20 * math.sqrt(dt) * rng.normal(size=state.shape)
            displacement = state - center
            radius = np.linalg.norm(displacement, axis=1)
            too_far = radius > 0.95
            state[too_far] = center + displacement[too_far] * (0.95 / radius[too_far, None])
        output.append(state.copy())
    return [x[:n] for x, n in zip(output, counts)]


GENERATORS = {
    "gaussian_translation": (2, snapshots_gaussian_translation),
    "gaussian_breathing": (2, snapshots_gaussian_breathing),
    "gmm_split": (2, snapshots_gmm_split),
    "gmm_rotate_reweight": (2, snapshots_gmm_rotate_reweight),
    "double_well_sde": (2, snapshots_double_well_sde),
    "ou_sde_5d": (5, snapshots_ou_sde),
    "two_moons_morph": (2, snapshots_two_moons_morph),
}


def write_dataset(name, dimension, generator):
    rng = np.random.default_rng(DATA_SEED + sum(map(ord, name)))
    snapshots = generator(rng)
    rows = []
    for time, points in zip(TIMES, snapshots):
        if points.shape != (COUNTS[name][int(time)], dimension) or not np.isfinite(points).all():
            raise ValueError(f"bad generated snapshot {name}, t={time}, shape={points.shape}")
        for point in points:
            row = {"samples": int(time)}
            row.update({f"x{i + 1}": float(point[i]) for i in range(dimension)})
            rows.append(row)
    path = OUT / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10g")
    return path


def write_configs(name, dimension, data_path, alpha):
    alpha_tag = str(alpha).replace(".", "p")
    ruot_run = f"../Results/lowdim_action_benchmarks/{name}_alpha{alpha_tag}/ruot"
    common = {
        "dataset": f"lowdim_{name}",
        "data_path": data_path.relative_to(ROOT).as_posix(),
        "dimension": dimension,
        "model_times": [0, 1, 2, 3],
        "model_sigma": 0.25,
        "growth_penalty": alpha,
        "seed": 42,
        "max_observed_points": None,
        "pair": {"training_mode": "steps", "training_steps": 6000,
                 "times_per_condition": 8, "conditions_per_batch": 2048,
                 "checkpoint_every_steps": 500, "log_every_steps": 100,
                 "validation_every_steps": 0},
        "coupling": {"mode": "full", "marginal_tolerance": 0.002,
                     "target_relative_tv": 0.001, "stationarity_tolerance": 0.001},
        "matching": {"steps": 30000, "validation_every": 0,
                      "feedback_activation": "tanh"},
    }
    CONFIGS.mkdir(parents=True, exist_ok=True)
    ruot = {"extends": "../gene_distance_hermite_primal.yaml", **common}
    wfr = {"extends": "../gene_wfr_bridge_tanh_matching.yaml", **common}
    wfr["comparison_run"] = ruot_run
    wfr["pair"] = {"training_mode": "analytic_wfr_bridge", "training_steps": 0,
                   "times_per_condition": 8, "conditions_per_batch": 2048,
                   "checkpoint_every_steps": 500, "log_every_steps": 100,
                   "validation_every_steps": 0}
    for method, config in (("ruot", ruot), ("wfr", wfr)):
        path = CONFIGS / f"{name}_alpha{alpha_tag}_{method}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main():
    for name, (dimension, generator) in GENERATORS.items():
        data_path = write_dataset(name, dimension, generator)
        for alpha in ALPHA_SWEEPS.get(name, (ALPHA,)):
            write_configs(name, dimension, data_path, alpha)
            print(f"{name}: d={dimension}, counts={COUNTS[name]}, alpha={alpha}, data={data_path}")


if __name__ == "__main__":
    main()

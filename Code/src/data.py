from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import urllib.request

import numpy as np
import pandas as pd
import torch

COMMIT = "e9b06bca530b4f442c829692bf43b656d0a1af4e"
URL = f"https://raw.githubusercontent.com/zhenyiizhang/DeepRUOT/{COMMIT}/data/simulation_gene_data.csv"
SHA256 = "d41e3bcd94ff55383264461f2eed048780dbea8faffb77011cff9b8dce231d10"
COUNTS = (400, 442, 530, 690, 969)


@dataclass
class Snapshot:
    time: float
    x: np.ndarray
    weights: np.ndarray
    original_count: int

    @property
    def mass(self):
        return float(self.weights.sum())


def load_gene(path="data/simulation_gene_data.csv", max_points=None, seed=42):
    """Download pinned bytes; subsampling preserves each original total mass."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError("official dataset SHA256 mismatch")
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("local dataset SHA256 mismatch; file was not overwritten")
    frame = pd.read_csv(path)
    if not {"samples", "x1", "x2"}.issubset(frame.columns):
        raise ValueError("unexpected CSV schema")
    if len(frame) != sum(COUNTS) or set(frame.samples) != set(range(5)):
        raise ValueError("unexpected observation times or row count")
    rng = np.random.default_rng(seed)
    snapshots = []
    for k, count in enumerate(COUNTS):
        x = frame.loc[frame.samples == k, ["x1", "x2"]].to_numpy(np.float64)
        if len(x) != count or not np.isfinite(x).all():
            raise ValueError(f"invalid snapshot {k}")
        if max_points is not None and len(x) > max_points:
            x = x[rng.choice(len(x), max_points, replace=False)]
        weights = np.full(len(x), (count / COUNTS[0]) / len(x))
        snapshots.append(Snapshot(float(k), x, weights, count))
    return snapshots


def load_csv_snapshots(path, times, max_points=None, seed=42, url=None, sha256=None, dimension=2):
    """Load weighted snapshots without silently dropping coordinate columns."""
    path = Path(path)
    if not path.exists():
        if not url:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if sha256 and hashlib.sha256(content).hexdigest() != sha256.lower():
            raise ValueError("downloaded CSV SHA256 mismatch")
        path.write_bytes(content)
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 and content_hash != sha256.lower():
        raise ValueError("local CSV SHA256 mismatch")
    frame = pd.read_csv(path)
    coordinates = [f"x{i}" for i in range(1, dimension + 1)]
    actual_coordinates = {c for c in frame.columns if c.startswith("x") and c[1:].isdigit()}
    if "samples" not in frame.columns or actual_coordinates != set(coordinates):
        raise ValueError(f"CSV coordinate columns must match configured dimension={dimension}: {coordinates}")
    observed_times = np.sort(frame["samples"].dropna().unique().astype(np.float64))
    times = np.asarray(times, dtype=np.float64)
    if len(observed_times) != len(times) or not np.allclose(observed_times, times, rtol=0, atol=1e-12):
        raise ValueError(f"CSV sample times {observed_times.tolist()} do not match configured times {times.tolist()}")
    counts = [int((frame["samples"] == t).sum()) for t in times]
    if not counts or min(counts) <= 0:
        raise ValueError("every configured sample time must have observations")
    rng = np.random.default_rng(seed)
    snapshots = []
    for t, count in zip(times, counts):
        x = frame.loc[frame.samples == t, coordinates].to_numpy(np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"invalid coordinates at time {t}")
        if max_points is not None and len(x) > max_points:
            x = x[rng.choice(len(x), max_points, replace=False)]
        mass = count / counts[0]
        weights = np.full(len(x), mass / len(x), dtype=np.float64)
        snapshots.append(Snapshot(float(t), x, weights, count))
    return snapshots


def partition_indices(n_source, n_target, coupling_config, rng):
    """Return corresponding source/target blocks with each point used once."""
    mode = coupling_config["mode"]
    if mode == "full":
        return [np.arange(n_source)], [np.arange(n_target)]
    if mode != "minibatch":
        raise ValueError(f"unsupported coupling mode: {mode}")
    requested = coupling_config.get("minibatch_blocks")
    if requested is None:
        hint = coupling_config["minibatch_size_hint"]
        if not isinstance(hint, int) or isinstance(hint, bool) or hint <= 0:
            raise ValueError("minibatch_size_hint must be a positive integer")
        nblocks = math.ceil(max(n_source, n_target) / hint)
    else:
        if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
            raise ValueError("minibatch_blocks must be a positive integer")
        nblocks = requested
    nblocks = min(nblocks, n_source, n_target)
    # Keep dataset order so paired methods use the exact same deterministic
    # block support. The RNG argument remains for API compatibility.
    source_blocks = np.array_split(np.arange(n_source), nblocks)
    target_blocks = np.array_split(np.arange(n_target), nblocks)
    return source_blocks, target_blocks


class PairIndex:
    """Flattened union of ALL adjacent candidate pairs; no nearest-neighbor filter."""

    def __init__(self, snapshots, device="cpu"):
        self.x = [torch.as_tensor(s.x, dtype=torch.float64, device=device) for s in snapshots]
        self.counts = [len(s.x) * len(t.x) for s, t in zip(snapshots[:-1], snapshots[1:])]
        self.offsets = np.cumsum([0] + self.counts)
        self.size = int(self.offsets[-1])
        self.positions = torch.cat(self.x)
        self.pair_offsets = torch.as_tensor(self.offsets, dtype=torch.long, device=device)
        self.point_offsets = torch.as_tensor(np.cumsum([0] + [len(s.x) for s in snapshots]),
                                             dtype=torch.long, device=device)
        self.target_counts = torch.tensor([len(s.x) for s in snapshots[1:]], dtype=torch.long, device=device)

    def delta_tensor(self, ids):
        """Map flattened pair ids entirely on their resident device."""
        if ids.ndim != 1 or ids.dtype != torch.long or ids.device != self.positions.device:
            raise ValueError("pair ids must be a resident one-dimensional int64 tensor")
        intervals = torch.bucketize(ids, self.pair_offsets[1:], right=True)
        local = ids - self.pair_offsets[intervals]
        n_target = self.target_counts[intervals]
        source = self.point_offsets[intervals] + torch.div(local, n_target, rounding_mode="floor")
        target = self.point_offsets[intervals + 1] + local.remainder(n_target)
        return self.positions[target] - self.positions[source]

    def delta(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.positions.device)
        return self.delta_tensor(ids)

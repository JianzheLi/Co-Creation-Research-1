"""Build and validate a whole interval cost table without chunk-wise transfers."""
import json
import math
import time
import numpy as np
import torch
from .coupling import OracleError
from .coupling_gpu import TorchHermiteCost


def build_table_gpu(oracle, delta, cfg, ratio_min, ratio_max, cache_path, fingerprint):
    parameter = next(oracle.path.parameters())
    shape = delta.shape[:2]
    flat = torch.as_tensor(delta.reshape(-1, delta.shape[-1]), device=parameter.device, dtype=torch.float64)
    rng = np.random.default_rng(991)
    ngrid = cfg["ratio_grid_nodes"]
    while ngrid <= cfg["max_ratio_grid_nodes"]:
        grid = np.geomspace(ratio_min, ratio_max, ngrid)
        metadata = dict(**fingerprint, ratio_grid_nodes=ngrid, ratio_range=[ratio_min, ratio_max],
                        interpolation="cubic_hermite_in_r", shape=list(shape))
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as saved:
                if json.loads(str(saved["metadata"])) == metadata:
                    costs = torch.as_tensor(saved["costs"], dtype=torch.float64, device=parameter.device)
                    derivatives = torch.as_tensor(saved["derivatives"], dtype=torch.float64, device=parameter.device)
                    table = TorchHermiteCost.from_tensors(grid, costs, derivatives, cfg["curvature_tolerance"])
                    return table, metadata
        started = time.perf_counter()
        # All displacement and ratio inputs are created on the GPU. Result
        # tensors are preallocated there by tensor_query; no per-chunk D2H copy.
        ratios = torch.as_tensor(grid, dtype=torch.float64, device=parameter.device).repeat(len(flat))
        expanded = flat.repeat_interleave(ngrid, dim=0)
        costs, derivatives = oracle.tensor_query(expanded, ratios, derivatives=1)
        del expanded, ratios
        costs, derivatives = costs.reshape(*shape, ngrid), derivatives.reshape(*shape, ngrid)
        try:
            table = TorchHermiteCost.from_tensors(grid, costs, derivatives, cfg["curvature_tolerance"])
        except OracleError as exc:
            increments = torch.diff(derivatives, dim=-1)
            index = int(increments.argmin())
            pair, segment = divmod(index, ngrid - 1)
            probe = np.geomspace(grid[segment], grid[segment + 1], 9)
            direct = oracle.tensor_query(flat[pair:pair + 1].expand(len(probe), -1), probe, derivatives=2)
            if float(direct[2].min()) < -cfg["curvature_tolerance"]:
                raise OracleError("cost_oracle_nonconvex", dict(**exc.details, delta=flat[pair].cpu().tolist(),
                                  ratios=probe.tolist(), cost=direct[0].cpu().tolist(),
                                  derivative=direct[1].cpu().tolist(), curvature=direct[2].cpu().tolist()))
            del costs, derivatives, increments
            ngrid = 2 * ngrid - 1
            continue
        pairs = np.concatenate((rng.integers(len(flat), size=min(512, len(flat))), rng.integers(len(flat), size=ngrid - 1)))
        probe_ratios = np.concatenate((np.exp(rng.uniform(math.log(ratio_min), math.log(ratio_max), min(512, len(flat)))),
                                       np.sqrt(grid[:-1] * grid[1:])))
        pair_ids = torch.as_tensor(pairs, device=parameter.device)
        probe = torch.as_tensor(probe_ratios, dtype=torch.float64, device=parameter.device)
        direct = oracle.tensor_query(flat[pair_ids], probe, derivatives=2)
        local = TorchHermiteCost.from_tensors(grid, costs.reshape(-1, ngrid)[pair_ids], derivatives.reshape(-1, ngrid)[pair_ids], cfg["curvature_tolerance"])
        interpolated = local(probe)
        errors = [float((a - b).abs().max()) for a, b in zip(direct, interpolated)]
        if float(direct[2].min()) < -cfg["curvature_tolerance"]:
            worst = int(direct[2].argmin())
            raise OracleError("cost_oracle_nonconvex", dict(delta=flat[pair_ids[worst]].cpu().tolist(),
                              ratio=float(probe[worst]), curvature=float(direct[2][worst])))
        passed = all(torch.allclose(a, b, atol=cfg["table_atol"], rtol=cfg["table_rtol"]) for a, b in zip(direct[:2], interpolated[:2]))
        if not passed:
            del table, costs, derivatives, local
            ngrid = 2 * ngrid - 1
            continue
        # One archival transfer per interval. The live table stays on the GPU
        # and is passed directly into the solver, never round-tripped via NumPy.
        np.savez_compressed(cache_path, grid=grid, costs=costs.cpu().numpy(), derivatives=derivatives.cpu().numpy(),
                            metadata=json.dumps(metadata))
        metadata.update(validation_max_errors=errors, min_curvature=table.min_curvature,
                        min_slope_increment=table.min_slope_increment, validation_samples=len(pairs),
                        table_seconds=time.perf_counter() - started, backend="torch", resident_device=str(parameter.device))
        return table, metadata
    raise OracleError("cost_table_accuracy_failed", dict(max_grid_nodes=cfg["max_ratio_grid_nodes"]))

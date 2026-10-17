"""Shared GPU distance/ratio table sampled directly from the frozen path."""
import math
import time
import warnings
import torch
from .coupling import OracleError
from .runtime import file_hash, json_write


class DistanceHermiteGrid:
    def __init__(self, distances, ratios, costs, slopes):
        self.distances, self.ratios = distances, ratios
        self.device = costs.device
        h = ratios.diff()
        secant = costs.diff(dim=-1)/h
        c0, c1 = costs[:, :-1], slopes[:, :-1]
        c2 = (3*secant-2*c1-slopes[:, 1:])/h
        c3 = (c1+slopes[:, 1:]-2*secant)/h.square()
        self.coefficients = torch.stack((c0, c1, c2, c3), -1)
        self.flat = self.coefficients.reshape(-1, 4)
        self.min_curvature = float(torch.minimum(2*c2, 2*c2+6*c3*h).min())
        if not bool(torch.isfinite(self.coefficients).all()):
            raise OracleError('cost_oracle_nonfinite', {})

    def bind(self, distances):
        return BoundDistanceCost(self, distances)


class BoundDistanceCost:
    """Bind fixed pair distances once; feasible ratios are supplied by the solver."""
    def __init__(self, grid, distances):
        self.grid = grid
        d = torch.as_tensor(distances, device=grid.device, dtype=grid.distances.dtype)
        self.shape = tuple(d.shape)
        if not bool(torch.isfinite(d).all() & (d >= grid.distances[0]).all() &
                    (d <= grid.distances[-1]*(1+1e-12)).all()):
            raise ValueError('distance outside shared table')
        self.index = (torch.searchsorted(grid.distances, d.contiguous(), right=True)-1).clamp(0, len(grid.distances)-2)
        self.weight = ((d-grid.distances[self.index]) /
                       (grid.distances[self.index+1]-grid.distances[self.index])).clamp(0, 1)[..., None]
        self.base = self.index*(len(grid.ratios)-1)

    def _lookup(self, r):
        r = torch.broadcast_to(torch.as_tensor(r, dtype=self.grid.ratios.dtype, device=self.grid.device), self.shape)
        segment = (torch.searchsorted(self.grid.ratios, r.contiguous(), right=True)-1).clamp(0, len(self.grid.ratios)-2)
        index = self.base+segment
        left = self.grid.flat[index]
        right = self.grid.flat[index+len(self.grid.ratios)-1]
        coefficients = left+self.weight*(right-left)
        return r-self.grid.ratios[segment], coefficients

    def __call__(self, r):
        z, c = self._lookup(r)
        value = c[..., 0]+z*(c[..., 1]+z*(c[..., 2]+z*c[..., 3]))
        slope = c[..., 1]+z*(2*c[..., 2]+3*z*c[..., 3])
        return value, slope

    def value(self, r):
        z, c = self._lookup(r)
        return c[..., 0]+z*(c[..., 1]+z*(c[..., 2]+z*c[..., 3]))


def _delta(d, dimension):
    return torch.cat((d.reshape(-1, 1), d.new_zeros((d.numel(), dimension-1))), -1)


def build_distance_table(oracle, distance_max, config, run):
    """Validate C and Cr on all cell centers, both axes' edges and random points."""
    started = time.perf_counter()
    cfg = config['coupling']
    path = oracle.path
    if path.parameterization != 'axial_distance':
        raise ValueError('shared distance table requires axial_distance path')
    parameter = next(path.parameters())
    opts = dict(dtype=parameter.dtype, device=parameter.device)
    nd, nr = cfg['distance_grid_nodes'], cfg['ratio_grid_nodes']
    rmin, rmax = config['pair']['ratio_min'], config['pair']['ratio_max']
    table_file = run/'distance_cost_table.pt'
    validation_file = run/'distance_table_validation.json'
    if table_file.exists() and validation_file.exists():
        cached = torch.load(table_file, map_location=parameter.device, weights_only=False)
        validation = __import__('json').loads(validation_file.read_text(encoding='utf-8'))
        distances, ratios = cached['distances'], cached['ratios']
        same_path = cached.get('path_checkpoint_sha256') == file_hash(run/'pair_path.pt')
        same_domain = (abs(float(distances[-1])-distance_max) <= 1e-12*max(1.0, distance_max)
                       and abs(float(ratios[0])-rmin) <= 1e-12
                       and abs(float(ratios[-1])-rmax) <= 1e-12)
        accurate = validation.get('validation_passed', False)
        if same_path and same_domain and accurate and len(distances) >= nd and len(ratios) >= nr:
            table = DistanceHermiteGrid(distances, ratios, cached['cost'], cached['slope'])
            print(f"Reusing verified shared path table: {len(distances)}x{len(ratios)}", flush=True)
            return table, validation
    attempts = []
    while True:
        dgrid = torch.linspace(0, distance_max, nd, **opts)
        rgrid = torch.linspace(math.log(rmin), math.log(rmax), nr, **opts).exp()
        rgrid[0], rgrid[-1] = rmin, rmax
        dd, rr = torch.meshgrid(dgrid, rgrid, indexing='ij')
        cost, slope = oracle.tensor_query(_delta(dd.flatten(), path.dimension), rr.flatten(), 1)
        table = DistanceHermiteGrid(dgrid, rgrid, cost.reshape(nd, nr), slope.reshape(nd, nr))
        dm, rm = (dgrid[:-1]+dgrid[1:])/2, (rgrid[:-1]*rgrid[1:]).sqrt()
        vd, vr = torch.meshgrid(torch.cat((dgrid[[0, -1]], dm)), torch.cat((rgrid[[0, -1]], rm)), indexing='ij')
        generator = torch.Generator(device=parameter.device).manual_seed(1949)
        random = torch.rand((4096, 2), generator=generator, **opts)
        vd = torch.cat((vd.flatten(), random[:, 0]*distance_max))
        vr = torch.cat((vr.flatten(), (math.log(rmin)+random[:, 1]*math.log(rmax/rmin)).exp()))
        exact = oracle.tensor_query(_delta(vd, path.dimension), vr, 1)
        interpolated = table.bind(vd)(vr)
        errors = {}
        for label, actual, expected in zip(('cost', 'first_derivative'), interpolated, exact):
            difference = (actual-expected).abs()
            errors[label] = dict(max_absolute=float(difference.max()),
                rmse=float(difference.square().mean().sqrt()),
                max_tolerance_ratio=float((difference/(cfg['table_atol']+cfg['table_rtol']*expected.abs())).max()))
        passed = all(item['max_tolerance_ratio'] <= 1 for item in errors.values())
        attempts.append(dict(grid_shape=[nd, nr], passed=passed, errors=errors))
        print(f'Shared path table: grid={nd}x{nr}, accuracy={passed}, errors={errors}', flush=True)
        if passed:
            break
        next_nd, next_nr = min(2*nd-1, cfg['max_distance_grid_nodes']), min(2*nr-1, cfg['max_ratio_grid_nodes'])
        if (next_nd, next_nr) == (nd, nr):
            raise OracleError('cost_table_accuracy_failed', dict(attempts=attempts))
        nd, nr = next_nd, next_nr
    if table.min_curvature < 0:
        warnings.warn(f'Shared table has minimum curvature {table.min_curvature}; projected gradient continues', RuntimeWarning)
    # Archive only the small shared grid, never a per-pair copy of all ratio nodes.
    torch.save(dict(distances=dgrid.cpu(), ratios=rgrid.cpu(), cost=cost.reshape(nd, nr).cpu(),
                    slope=slope.reshape(nd, nr).cpu(), path_checkpoint_sha256=file_hash(run/'pair_path.pt')),
               table_file)
    metadata = dict(cost_oracle='path_quadrature', evaluation='shared_distance_hermite',
                    interpolation='linear_in_distance_cubic_hermite_in_ratio', grid_shape=[nd, nr],
                    teacher_points=nd*nr, validation_points=len(vd), validation_passed=True,
                    validation_errors=errors, min_curvature=table.min_curvature, attempts=attempts,
                    seconds=time.perf_counter()-started, table_sha256=file_hash(table_file),
                    path_checkpoint_sha256=file_hash(run/'pair_path.pt'), quadrature_nodes=oracle.nodes)
    json_write(run/'distance_table_validation.json', metadata)
    return table, metadata

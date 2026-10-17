import numpy as np
import pytest
import torch
from src.distance_cost_table import DistanceHermiteGrid
from src.primal_coupling_gpu import solve_primal_gpu


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_shared_table_exact_cubic_and_consistent_gradient(device):
    d = torch.linspace(0, 3, 9, device=device, dtype=torch.float64)
    r = torch.logspace(-1, 1, 17, device=device, dtype=torch.float64)
    values = d[:, None]+r[None, :]**3
    slopes = (3*r.square()).expand_as(values)
    grid = DistanceHermiteGrid(d, r, values, slopes)
    distances = torch.tensor([[0., .31, 3.], [1., 2.4, .7]], device=device, dtype=torch.float64)
    ratio = torch.tensor([[.1, .42, 10.], [.61, 1.4, 2.1]], device=device, dtype=torch.float64, requires_grad=True)
    bound = grid.bind(distances)
    c, cr = bound(ratio)
    torch.testing.assert_close(c, distances+ratio**3, atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(cr, 3*ratio.square(), atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(bound.value(ratio).sum(), ratio)[0], cr)
    # Nodes agree exactly from either neighboring segment, without a slope jump.
    for knot in r[1:-1]:
        left = bound(torch.full_like(distances, float(knot)-1e-10))[1]
        right = bound(torch.full_like(distances, float(knot)+1e-10))[1]
        assert (left-right).abs().max() < 1e-7


def test_shared_table_primal_matches_direct_cost():
    d = torch.linspace(0, 1, 33, dtype=torch.float64)
    r = torch.logspace(-1, 1, 257, dtype=torch.float64)
    cost = d[:, None]+20*(1+r-2*r.sqrt())
    slopes = (20*(1-r.rsqrt())).expand_as(cost)
    grid = DistanceHermiteGrid(d, r, cost, slopes)
    delta = torch.tensor([[.1, .2, .3], [.4, .2, .1]], dtype=torch.float64)
    class Exact:
        def value(self, r):
            return delta+20*(1+r-2*r.sqrt())
        def __call__(self, r):
            return self.value(r), 20*(1-r.rsqrt())
    args = ([.4, .6], [.3, .5, .4])
    ref = solve_primal_gpu(*args, Exact(), epsilon=.2, device='cpu', stationarity_tolerance=1e-5)
    actual = solve_primal_gpu(*args, grid.bind(delta), epsilon=.2, device='cpu', stationarity_tolerance=1e-5)
    assert actual.status == ref.status == 'converged'
    np.testing.assert_allclose(actual.P, ref.P, atol=3e-6, rtol=1e-4)
    np.testing.assert_allclose(actual.Q, ref.Q, atol=3e-6, rtol=1e-4)

import numpy as np
import pytest
import torch
from src.coupling import HermiteCost, solve_dual
from src.coupling_gpu import TorchHermiteCost, solve_dual_gpu


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_device_hermite_and_dual_match_reference(device):
    grid = np.geomspace(0.1, 10, 65)
    offset = np.array([[0.1, 0.2, 0.3], [0.4, 0.2, 0.1]])
    cost = offset[..., None] + 20 * (1 + grid - 2 * np.sqrt(grid))
    derivative = np.broadcast_to(20 * (1 - 1 / np.sqrt(grid)), cost.shape).copy()
    table = HermiteCost(grid, cost, derivative)
    gpu = TorchHermiteCost(table, device)
    r = np.array([[0.1, 0.63, 2], [1.8, 4, 10]])
    for a, b in zip(table(r), gpu(torch.tensor(r, device=device))):
        np.testing.assert_allclose(a, b.cpu(), rtol=1e-11, atol=1e-11)
    for lam in ([-100, 0, 100], [-2, 1, 2]):
        expected = table.conjugate(np.array(lam))
        actual = gpu.conjugate(torch.tensor(lam, dtype=torch.float64, device=device))
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b.cpu(), rtol=1e-10, atol=1e-10)
        assert actual[3].max().item() < 1e-9
    a, b = np.array([0.4, 0.6]), np.array([0.3, 0.5, 0.4])
    expected = solve_dual(a, b, table, epsilon=0.2, marginal_tolerance=0.002)
    actual = solve_dual_gpu(a, b, table, epsilon=0.2, marginal_tolerance=0.002, device=device)
    assert actual.status == "converged"
    np.testing.assert_allclose(expected.P, actual.P, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(expected.Q, actual.Q, rtol=1e-9, atol=1e-10)
    assert actual.history[-1]["row_relative_tv"] < 1e-3

import copy

import numpy as np
import pytest
import torch

from src.pair_cost import CostOracle
from src.pair_path import PairPath


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("derivatives", [0, 1, 2])
def test_tensor_query_matches_cpu_and_preserves_ratio_derivatives(device, derivatives):
    torch.manual_seed(43)
    path = PairPath(dimension=2, width=8, layers=2).double()
    # Nonzero corrections exercise dependence of the time JVP on log-ratio.
    with torch.no_grad():
        for net in (path.mu_net, path.var_net, path.mass_net):
            net.last.weight.normal_(0, 0.03)
    reference = CostOracle(copy.deepcopy(path), nodes=12, condition_chunk=100)
    oracle = CostOracle(path.to(device), nodes=12, condition_chunk=3)
    delta = torch.randn(11, 2, dtype=torch.float64)
    ratio = torch.linspace(0.3, 2.5, 11, dtype=torch.float64)
    expected = reference(delta, ratio, derivatives)
    actual = oracle.tensor_query(delta.to(device), ratio.to(device), derivatives)
    expected = (expected,) if derivatives == 0 else expected
    actual = (actual,) if derivatives == 0 else actual
    for a, e in zip(actual, expected):
        assert a.device.type == device
        assert not a.requires_grad
        np.testing.assert_allclose(a.cpu(), e, rtol=2e-10, atol=2e-10)
    if derivatives == 2:
        h = 1e-4
        plus = oracle.tensor_query(delta, ratio + h, 0)
        minus = oracle.tensor_query(delta, ratio - h, 0)
        torch.testing.assert_close((plus - minus) / (2 * h), actual[1], rtol=2e-6, atol=2e-6)
        torch.testing.assert_close((plus - 2 * actual[0] + minus) / h**2, actual[2], rtol=2e-5, atol=2e-5)


def test_tensor_query_scalar_ratio_and_empty_input():
    oracle = CostOracle(PairPath(width=8, layers=1).double(), nodes=8, condition_chunk=2)
    output = oracle.tensor_query(torch.zeros(5, 2), 1.0, 1)
    assert all(t.shape == (5,) for t in output)
    assert oracle.tensor_query(torch.empty(0, 2), torch.empty(0), 0).shape == (0,)
    with pytest.raises(ValueError, match="positive ratio"):
        oracle.tensor_query(torch.zeros(2, 2), 0.0)

import numpy as np
import pandas as pd

from src.data import load_csv_snapshots, partition_indices
from src.runtime import load_config


def test_mouse_configs_use_same_eight_ordered_blocks_and_unit_times():
    ruot = load_config("configs/mouse_ruot_tanh_minibatch.yaml")
    wfr = load_config("configs/mouse_wfr_bridge_tanh_minibatch.yaml")
    assert ruot["model_times"] == wfr["model_times"] == [0, 1, 2]
    assert ruot["model_sigma"] == wfr["model_sigma"] == 0.25
    assert ruot["growth_penalty"] == wfr["growth_penalty"] == 10.0
    assert ruot["coupling"]["mode"] == wfr["coupling"]["mode"] == "minibatch"
    assert ruot["coupling"]["minibatch_blocks"] == wfr["coupling"]["minibatch_blocks"] == 8
    for n_source, n_target in ((1429, 3781), (3781, 5788)):
        source_a, target_a = partition_indices(n_source, n_target, ruot["coupling"], np.random.default_rng(1))
        source_b, target_b = partition_indices(n_source, n_target, wfr["coupling"], np.random.default_rng(999))
        assert len(source_a) == len(target_a) == len(source_b) == len(target_b) == 8
        for left, right in zip(source_a, source_b):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(target_a, target_b):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(np.concatenate(source_a), np.arange(n_source))
        np.testing.assert_array_equal(np.concatenate(target_a), np.arange(n_target))


def test_csv_snapshot_loader_preserves_per_timepoint_total_mass(tmp_path):
    frame = pd.DataFrame({
        "samples": [0, 0, 0, 1, 1, 1, 1, 2, 2],
        "x1": np.arange(9, dtype=float),
        "x2": np.arange(9, dtype=float) * -0.5,
    })
    path = tmp_path / "tiny_mouse.csv"
    frame.to_csv(path, index=False)
    snapshots = load_csv_snapshots(path, [0, 1, 2], max_points=2, seed=12)
    assert [snapshot.original_count for snapshot in snapshots] == [3, 4, 2]
    np.testing.assert_allclose([snapshot.mass for snapshot in snapshots], [1, 4 / 3, 2 / 3])
    assert all(len(snapshot.x) == 2 for snapshot in snapshots)

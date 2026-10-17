import numpy as np
import pandas as pd
import pytest
import torch

from src.data import load_csv_snapshots
from src.experiment import sensitivity
from src.matching import Feedback
from src.runtime import load_config, snapshots_for


def test_emt_preserves_all_coordinates_and_shared_settings():
    ruot = load_config("configs/emt_distance_hermite_primal.yaml")
    wfr = load_config("configs/emt_wfr_bridge.yaml")
    for key in ("dimension", "growth_penalty", "model_sigma", "matching", "generation", "model_times"):
        assert ruot[key] == wfr[key]
    assert ruot["dimension"] == 10
    assert ruot["growth_penalty"] == 10
    snapshots = snapshots_for(ruot)
    frame = pd.read_csv(ruot["data_path"])
    assert [len(s.x) for s in snapshots] == [577, 885, 788, 883]
    for s in snapshots:
        np.testing.assert_array_equal(s.x, frame.loc[frame.samples == s.time, [f"x{i}" for i in range(1, 11)]].to_numpy())
        assert s.mass == pytest.approx(s.original_count / 577)
    with pytest.raises(ValueError, match="dimension"):
        load_csv_snapshots(ruot["data_path"], ruot["model_times"])


def test_high_dimensional_step_sensitivity_runs():
    config = load_config("configs/emt_distance_hermite_primal.yaml")
    config["generation"].update(sensitivity_particles=4, n_particles=4, steps_per_unit_interval=2)
    model = Feedback(dimension=10, width=4, layers=1, activation="tanh")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = sensitivity(model, snapshots_for(config), config)
    # With zero drift and growth, coupled Brownian increments agree at snapshots.
    assert result.step_position_rms.max() < 1e-6
    assert result.step_mass_relative_difference.max() == 0

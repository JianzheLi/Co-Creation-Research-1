import json
import math

import numpy as np
import pytest
import torch

from src.data import PairIndex, Snapshot
from src.pair_path import PairPath


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def snapshots():
    return [Snapshot(float(k), np.arange(n * 2).reshape(n, 2) / 10 + k,
                     np.full(n, 1 / n), n) for k, n in enumerate([2, 3, 1, 4, 2])]


@pytest.mark.parametrize("device", DEVICES)
def test_resident_pair_mapping_including_all_interval_boundaries(device):
    data = snapshots()
    index = PairIndex(data, device)
    expected = np.concatenate([(b.x[None, :, :] - a.x[:, None, :]).reshape(-1, 2)
                               for a, b in zip(data[:-1], data[1:])])
    ids = torch.arange(index.size, device=device)
    actual = index.delta_tensor(ids)
    assert actual.device.type == device
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
    torch.testing.assert_close(index.delta(np.arange(index.size)), actual)
    permutation = torch.randperm(index.size, device=device)
    torch.testing.assert_close(index.delta_tensor(permutation), actual[permutation])
    torch.testing.assert_close(permutation.sort().values, ids)
    assert index.delta_tensor(ids[:0]).shape == (0, 2)


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_pretraining_anchor_order_coverage_and_exact_resume(tmp_path, monkeypatch, device):
    import src.train_pairs as training

    def configure(config):
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        return torch.device(device)

    def init(run, config):
        run.mkdir(parents=True, exist_ok=True)
        return run

    monkeypatch.setattr(training, "configure", configure)
    monkeypatch.setattr(training, "init_run", init)
    monkeypatch.setattr(training, "snapshots_for", lambda config: snapshots())
    monkeypatch.setattr(training, "make_path", lambda config, device: PairPath(width=4, layers=1).to(device=device, dtype=torch.float64))
    monkeypatch.setattr(training, "validate", lambda *args: {"test": True})
    original_integrand = training.finite_pair_integrand
    recorded = []

    def capture(state, *args):
        recorded.append((state["log_r"].detach().clone(), state["t"].detach().clone()))
        return original_integrand(state, *args)

    monkeypatch.setattr(training, "finite_pair_integrand", capture)
    config = dict(seed=123, pair=dict(learning_rate=.0001, ratio_min=.2, ratio_max=5.,
                  validation_conditions=4, coverage_epochs=4, conditions_per_batch=8,
                  times_per_condition=3, validation_every=1))
    full_run, resumed_run = tmp_path / "full", tmp_path / "resume"
    training.train_pairs(config, full_run)
    full = torch.load(full_run / "pair_path.pt", weights_only=False)
    batches = math.ceil(PairIndex(snapshots()).size / 8)
    for epoch, anchor in enumerate([math.log(.2), 0, math.log(5)]):
        ell = torch.cat([item[0] for item in recorded[epoch * batches:(epoch + 1) * batches]])
        torch.testing.assert_close(ell, torch.full_like(ell, anchor))
    for ell, t in recorded:
        assert ell.device.type == device and t.device.type == device
        assert ((t > 0) & (t < 1)).all()
        # Every condition receives exactly the configured number of time draws.
        torch.testing.assert_close(ell.reshape(-1, 3)[:, :1].expand(-1, 3), ell.reshape(-1, 3))
    report = [json.loads(line) for line in (full_run / "pair_history.jsonl").read_text().splitlines()]
    assert all(row["coverage_fraction"] == 1 and row["pairs_seen"] == 21 for row in report)
    assert full["pair_training_version"] == training.PAIR_TRAINING_VERSION
    config["pair"]["coverage_epochs"] = 2
    training.train_pairs(config, resumed_run)
    cp_file = resumed_run / "pair_path.pt"
    cp = torch.load(cp_file, weights_only=False)
    del cp["pair_training_version"]  # Also exercise loading the legacy schema.
    torch.save(cp, cp_file)
    config["pair"]["coverage_epochs"] = 4
    training.train_pairs(config, resumed_run)
    resumed = torch.load(cp_file, weights_only=False)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_step_sampling_uniform_pairs_and_explicit_ratio_measure(device):
    from src.train_pairs import sample_pair_conditions
    index = PairIndex(snapshots(), device)
    torch.manual_seed(81)
    config = dict(ratio_min=.2, ratio_max=5.)
    ids, delta, ell = sample_pair_conditions(index, config, 100_000, device)
    frequency = torch.bincount(ids, minlength=index.size).double() / len(ids)
    assert (frequency - 1 / index.size).abs().max().item() < .003
    assert ell.exp().mean().item() == pytest.approx(2.6, abs=.02)
    assert ell.exp().var().item() == pytest.approx(4.8 ** 2 / 12, abs=.025)
    torch.testing.assert_close(delta, index.delta_tensor(ids))
    config["ratio_sampling"] = "log_uniform"
    _, _, log_ratios = sample_pair_conditions(index, config, 100_000, device)
    assert log_ratios.mean().item() == pytest.approx(0, abs=.015)
    assert log_ratios.var().item() == pytest.approx((2 * math.log(5)) ** 2 / 12, abs=.015)


@pytest.mark.parametrize("device", DEVICES)
def test_step_training_counts_coverage_and_resume(tmp_path, monkeypatch, device):
    import src.train_pairs as training

    def configure(config):
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        return torch.device(device)

    def init(run, config):
        run.mkdir(parents=True, exist_ok=True)
        return run

    monkeypatch.setattr(training, "configure", configure)
    monkeypatch.setattr(training, "init_run", init)
    monkeypatch.setattr(training, "snapshots_for", lambda config: snapshots())
    monkeypatch.setattr(training, "make_path", lambda config, device: PairPath(width=4, layers=1).to(device=device, dtype=torch.float64))
    monkeypatch.setattr(training, "validate", lambda *args: {"test": True})
    config = dict(seed=17, pair=dict(training_mode="steps", training_steps=4,
                  learning_rate=.0001, ratio_min=.2, ratio_max=5.,
                  validation_conditions=4, conditions_per_batch=3, times_per_condition=3,
                  checkpoint_every_steps=2, log_every_steps=2, validation_every_steps=2))
    full_run, resumed_run = tmp_path / "full", tmp_path / "resume"
    training.train_pairs(config, full_run)
    full = torch.load(full_run / "pair_path.pt", weights_only=False)
    assert full["updates"] == 4
    assert full["candidate_pair_visits"] == 12
    assert full["condition_time_samples"] == 36
    status = json.loads((full_run / "status.json").read_text())["pair"]
    assert status["unique_pairs_seen"] == full["visited_pair_ids"].sum().item()
    assert status["unique_coverage_fraction"] < 1
    assert "coverage_fraction" not in status
    config["pair"]["training_steps"] = 2
    training.train_pairs(config, resumed_run)
    config["pair"]["training_steps"] = 4
    training.train_pairs(config, resumed_run)
    resumed = torch.load(resumed_run / "pair_path.pt", weights_only=False)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=0, rtol=0)
    torch.testing.assert_close(full["visited_pair_ids"], resumed["visited_pair_ids"])

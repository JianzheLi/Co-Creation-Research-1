"""Statistical checks exercise the actual GPU-resident matching batch builder."""
import pytest
import torch
import json

from src.matching import CouplingSampler, matching_learning_rate, sample_training_batch
from src.pair_path import PairPath


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_sampler(device):
    sampler = CouplingSampler.__new__(CouplingSampler)
    sampler.masses = torch.tensor([1., 1.2, 1.6, 2.], dtype=torch.float64, device=device)
    sampler.config = {"matching": {}}
    source_positions, target_positions, pair_source, pair_target, log_r = [], [], [], [], []
    probabilities = []
    for k in range(4):
        positions = torch.tensor([[k, 0.], [k, 1.], [k, 2.]], dtype=torch.float64, device=device)
        source_positions.append(positions)
        target_positions.append(positions.clone())
        ids = torch.arange(3, dtype=torch.int32, device=device) + 3 * k
        pair_source.append(ids)
        pair_target.append(ids)
        log_r.append(torch.zeros((3, 1), dtype=torch.float64, device=device))
        probabilities.append(torch.tensor([0., .3, .7], dtype=torch.float64, device=device) / 4)
    sampler.source_positions = torch.cat(source_positions)
    sampler.target_positions = torch.cat(target_positions)
    sampler.pair_source = torch.cat(pair_source)
    sampler.pair_target = torch.cat(pair_target)
    sampler.log_r = torch.cat(log_r)
    sampler.cdf = torch.cat(probabilities).cumsum(0)
    sampler.cdf[-1] = 1
    sampler.interval_ends = torch.tensor([3, 6, 9], dtype=torch.long, device=device)
    return sampler


@pytest.mark.parametrize("device", DEVICES)
def test_joint_cdf_preserves_uniform_intervals_and_conditional_pair_law(device):
    sampler = make_sampler(device)
    generator = torch.Generator(device=device).manual_seed(193)
    batch = sampler.sample(100_000, device, generator)
    assert batch["x0"].device.type == device
    for k in range(4):
        selected = batch["interval"] == k
        assert selected.double().mean().item() == pytest.approx(.25, abs=.006)
        assert (batch["x0"][selected, 1] == 1).double().mean().item() == pytest.approx(.3, abs=.012)
    assert (batch["x0"][:, 1] > 0).all()  # Zero-mass support is never selected.
    torch.testing.assert_close(batch["interval_factor"], 4 * sampler.masses[batch["interval"]])


@pytest.mark.parametrize("device", DEVICES)
def test_training_draws_full_conditional_gaussian_and_open_uniform_time(device):
    sampler = make_sampler(device)
    path = PairPath(width=4, layers=1).to(device=device, dtype=torch.float64).freeze()
    # Distinct variances make a scalar/noise broadcasting bug observable.
    with torch.no_grad():
        path.var_net.last.bias.copy_(torch.tensor([.4, -.6], device=device))
    seed = 214
    generator = torch.Generator(device=device).manual_seed(seed)
    global_t, targets, _ = sample_training_batch(sampler, path, 60_000, device, generator)
    generator.manual_seed(seed)
    batch = sampler.sample(60_000, device, generator)
    eps = torch.finfo(torch.float64).eps
    t = eps / 2 + (1 - eps) * torch.rand((60_000, 1), dtype=torch.float64, device=device, generator=generator)
    with torch.no_grad():
        state = path(t, batch["delta"], batch["log_r"], x0=batch["x0"])
    z = (targets["x"].double() - state["xi"]) / state["var"].sqrt()
    assert (t > 0).all() and (t < 1).all()
    assert t.mean().item() == pytest.approx(.5, abs=.005)
    assert t.var().item() == pytest.approx(1 / 12, abs=.0015)
    assert z.mean(0).abs().max().item() < .015
    torch.testing.assert_close(z.var(0), torch.ones(2, device=device, dtype=torch.float64), atol=.025, rtol=0)
    assert (z[:, 0] * z[:, 1]).mean().abs().item() < .015
    assert (z.abs() > 2).double().mean().item() == pytest.approx(.0455, abs=.004)
    assert all(not value.requires_grad for value in targets.values())
    assert global_t.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_validation_generator_does_not_advance_training_rng(device):
    sampler = make_sampler(device)
    path = PairPath(width=4, layers=1).to(device=device, dtype=torch.float64).freeze()
    before = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    first = sample_training_batch(sampler, path, 32, device, torch.Generator(device=device).manual_seed(42))
    second = sample_training_batch(sampler, path, 32, device, torch.Generator(device=device).manual_seed(42))
    after = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    assert torch.equal(before, after)
    torch.testing.assert_close(first[0], second[0])
    for key in first[1]:
        torch.testing.assert_close(first[1][key], second[1][key])


def test_optional_cosine_schedule_and_legacy_constant_default():
    config = dict(learning_rate=.001, steps=100)
    assert matching_learning_rate(config, 99) == .001
    config.update(lr_schedule="cosine", min_learning_rate=.0001)
    assert matching_learning_rate(config, 0) == .001
    assert matching_learning_rate(config, 99) == .0001
    assert matching_learning_rate(config, 30) > matching_learning_rate(config, 60)


def test_two_cycle_cosine_schedule_restarts_once():
    config = dict(learning_rate=.001, min_learning_rate=.0001,
                  steps=100, lr_schedule="cosine_2cycle")
    assert matching_learning_rate(config, 0) == .001
    assert matching_learning_rate(config, 49) == .0001
    assert matching_learning_rate(config, 50) == .001
    assert matching_learning_rate(config, 99) == .0001
    assert matching_learning_rate(config, 60) > matching_learning_rate(config, 80)


def test_two_cycle_cosine_schedule_halves_second_cycle_rates():
    config = dict(learning_rate=.001, min_learning_rate=.0001,
                  steps=100, lr_schedule="cosine_2cycle_half")
    assert matching_learning_rate(config, 0) == .001
    assert matching_learning_rate(config, 49) == .0001
    assert matching_learning_rate(config, 50) == .0005
    assert matching_learning_rate(config, 99) == .00005


def test_training_checkpoint_validation_and_legacy_resume(tmp_path, monkeypatch):
    import src.matching as matching
    (tmp_path / "pair_path.pt").write_bytes(b"test path identity")
    (tmp_path / "couplings").mkdir()
    (tmp_path / "couplings" / "manifest.json").write_text("{}")
    path = PairPath(width=4, layers=1).double().freeze()
    monkeypatch.setattr(matching, "configure", lambda config: torch.device("cpu"))
    monkeypatch.setattr(matching, "init_run", lambda run, config: run)
    monkeypatch.setattr(matching, "CouplingSampler", lambda run, config, device: make_sampler(device))
    monkeypatch.setattr(matching, "load_path", lambda run, config, device: path)
    config = dict(seed=2, dimension=2, matching=dict(hidden_width=4, hidden_layers=1,
                  learning_rate=.001, steps=3, batch_size=16, log_every=1,
                  lr_schedule="cosine", min_learning_rate=.0001,
                  validation_every=2, validation_batch_size=32))
    matching.train_matching(config, tmp_path)
    cp_file = tmp_path / "feedback.pt"
    cp = torch.load(cp_file, weights_only=False)
    assert cp["matching_version"] == matching.MATCHING_VERSION
    assert cp["completed_steps"] == 3
    validation = [json.loads(line) for line in (tmp_path / "matching_validation.jsonl").read_text().splitlines()]
    assert [row["step"] for row in validation] == [1, 2, 3]
    assert all(row["samples"] == 32 and row["loss"] > 0 for row in validation)
    del cp["matching_version"]  # Legacy checkpoints did not carry this field.
    torch.save(cp, cp_file)
    config["matching"]["steps"] = 5
    matching.train_matching(config, tmp_path)
    assert torch.load(cp_file, weights_only=False)["completed_steps"] == 5

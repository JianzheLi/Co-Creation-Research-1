"""Verify the saved distance-path -> distilled cost -> coupling pipeline."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure, file_hash, json_write, load_config, load_path
from src.cost_distillation import load_distilled_oracle
import torch


def audit(run):
    run = Path(run)
    config = load_config(run / "config.yaml")
    device = configure(config)
    path = load_path(run, config, device)
    assert path.parameterization == "axial_distance"
    delta = torch.randn(16, path.dimension, dtype=torch.float64, device=device)
    t = torch.linspace(0.01, 0.99, 16, dtype=torch.float64, device=device)[:, None]
    ell = torch.linspace(-1, 1, 16, dtype=torch.float64, device=device)[:, None]
    expected = torch.cat((t, delta.norm(dim=-1, keepdim=True)/path.delta_scale, ell), -1)
    inputs = {}
    hooks = []
    for name in ("mu_net", "var_net", "mass_net"):
        hooks.append(getattr(path, name).hidden[0].register_forward_pre_hook(
            lambda module, args, name=name: inputs.update({name: args[0].detach()})))
    path(t, delta, ell)
    for hook in hooks:
        hook.remove()
    input_errors = {name: float((values-expected).abs().max()) for name, values in inputs.items()}
    assert len(input_errors) == 3 and max(input_errors.values()) == 0
    endpoints = []
    for local_t in (0., 1.):
        s = path(torch.full_like(t, local_t), delta, ell)
        error = max(float(s["var"].abs().max()),
                    float((s["dotvar"]-(1-2*local_t)*path.sigma**2).abs().max()))
        assert error < 1e-12
        endpoints.append(dict(t=local_t, covariance_eigenvalue_and_slope_max_error=error))
    if config['coupling'].get('cost_evaluation') == 'shared_distance_hermite':
        from src.distance_cost_table import DistanceHermiteGrid
        saved = torch.load(run/'distance_cost_table.pt', map_location=device, weights_only=False)
        assert saved['path_checkpoint_sha256'] == file_hash(run/'pair_path.pt')
        grid = DistanceHermiteGrid(saved['distances'], saved['ratios'], saved['cost'], saved['slope'])
        def forbidden_path(*args, **kwargs):
            raise AssertionError('table query tried to evaluate the path')
        path.forward = forbidden_path
        distances = torch.linspace(0, float(saved['distances'][-1]), 16, dtype=torch.float64, device=device)
        ratio = ell.flatten().exp().detach().requires_grad_(True)
        query = grid.bind(distances)
        values, slopes = query(ratio)
        derivative_error = float((torch.autograd.grad(values.sum(), ratio)[0]-slopes).detach().abs().max())
        assert derivative_error < 1e-10
        validation = json.loads((run/'distance_table_validation.json').read_text(encoding='utf-8'))
        assert validation['validation_passed']
        manifest = json.loads((run/'couplings'/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'converged' and manifest['cost_oracle'] == 'path_quadrature'
        assert manifest['cost_evaluation'] == 'shared_distance_hermite'
        assert manifest['table_sha256'] == file_hash(run/'distance_cost_table.pt')
        assert manifest['path_checkpoint_sha256'] == file_hash(run/'pair_path.pt')
        assert not config.get('distillation', {}).get('enabled') and not (run/'cost_network.pt').exists()
        for interval in manifest['intervals']:
            result = json.loads((run/'couplings'/Path(interval['file']).with_suffix('.json')).read_text(encoding='utf-8'))
            assert result['cost_metadata']['table_sha256'] == manifest['table_sha256']
            assert result['cost_metadata']['evaluation'] == 'shared_distance_hermite'
        report = dict(passed=True, device=str(device), path_input_max_errors=input_errors,
                      endpoint_checks=endpoints, distillation_enabled=False,
                      cost_evaluation='shared_distance_hermite', table_validation=validation,
                      derivative_consistency_max_error=derivative_error,
                      deployed_cost_queries_require_no_path_forward=True,
                      coupling_intervals=len(manifest['intervals']))
        json_write(run/'pipeline_audit.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    oracle = load_distilled_oracle(config, run, path, device)
    # A deployed query must succeed even when teacher forward is forbidden.
    def forbidden(*args, **kwargs):
        raise AssertionError("student query tried to run teacher path")
    path.forward = forbidden
    direction = delta/delta.norm(dim=-1, keepdim=True)
    distance = torch.linspace(0, oracle.distance_max, len(delta), dtype=torch.float64, device=device)[:, None]
    values = oracle.tensor_query(direction*distance, ell.exp(), 2)
    assert all(torch.isfinite(v).all() for v in values)
    rotation = torch.linalg.qr(torch.randn(path.dimension, path.dimension, dtype=torch.float64, device=device)).Q
    rotated = oracle.tensor_query((direction*distance)@rotation.T, ell.exp(), 2)
    rotation_error = max(float((a-b).abs().max()) for a, b in zip(values, rotated))
    assert rotation_error < 1e-9
    validation = json.loads((run / "distillation_validation.json").read_text(encoding="utf-8"))
    assert validation["training_complete"] and validation["optimizer"] == "Adam"
    assert validation["completed_steps"] == config["distillation"]["training_steps"]
    manifest = json.loads((run / "couplings" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "converged"
    assert manifest["cost_oracle"] == "distilled_mlp"
    assert manifest["cost_evaluation"] == "direct_mlp_autodiff"
    assert manifest["cost_network_sha256"] == file_hash(run / "cost_network.pt")
    assert manifest["path_checkpoint_sha256"] == file_hash(run / "pair_path.pt")
    for interval in manifest["intervals"]:
        result = json.loads((run / "couplings" / Path(interval["file"]).with_suffix(".json")).read_text(encoding="utf-8"))
        assert result["cost_metadata"]["cost_oracle"] == "distilled_mlp"
        assert result["cost_metadata"]["interpolation"] == "none"
        assert result["cost_metadata"]["cost_network_sha256"] == manifest["cost_network_sha256"]
    report = dict(passed=True, device=str(device), path_input_order=["time", "distance/delta_scale", "log_mass_ratio"],
                  path_input_max_errors=input_errors, endpoint_checks=endpoints,
                  cost_network_inputs=["distance", "log_mass_ratio"],
                  optimizer="Adam", supervised_targets=["cost", "dC/dr"] +
                    (["d2C/dr2"] if config["distillation"].get("curvature_weight", 0) > 0 else []),
                  deployed_cost_queries_require_no_path_forward=True,
                  cost_rotation_max_error=rotation_error,
                  distillation_validation=validation,
                  coupling_cost_oracle=manifest["cost_oracle"], coupling_blocks=len(manifest["intervals"]),
                  path_checkpoint_sha256=manifest["path_checkpoint_sha256"],
                  cost_network_sha256=manifest["cost_network_sha256"])
    json_write(run / "pipeline_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    audit(parser.parse_args().run_dir)

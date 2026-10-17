import numpy as np
import ot
from scipy.special import logsumexp


def particle_mass_and_weights(logw_gen):
    """Return representable total mass and stable normalized particle weights.

    Keep normalization in log space: exponentiating raw log-weights can turn a
    growth runaway into ``inf`` and downstream ``nan_to_num`` calls can silently
    erase the particles carrying almost all mass.
    """
    logw = np.asarray(logw_gen, dtype=np.float64).reshape(-1)
    if not len(logw) or not np.isfinite(logw).all():
        raise FloatingPointError("nonfinite or empty particle log-weights")
    log_mass = float(logsumexp(logw))
    if not np.isfinite(log_mass):
        raise FloatingPointError("nonfinite predicted log total mass")
    max_log = float(np.log(np.finfo(np.float64).max))
    min_log = float(np.log(np.nextafter(np.float64(0), np.float64(1))))
    if log_mass > max_log:
        raise FloatingPointError(
            f"predicted total mass exceeds float64 range (log total mass={log_mass:.6g})")
    if log_mass < min_log:
        raise FloatingPointError(
            f"predicted total mass underflows float64 (log total mass={log_mass:.6g})")
    mass = float(np.exp(log_mass))
    weights = np.exp(logw - log_mass)
    normalization = float(weights.sum())
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(normalization) or normalization <= 0:
        raise FloatingPointError("predicted particle mass/normalization is not representable")
    weights /= normalization
    return mass, weights, log_mass


def evaluate_snapshot(x_gen, logw_gen, y_obs, n_initial_observed=400, mass_true=None):
    x, y = np.asarray(x_gen, dtype=np.float64), np.asarray(y_obs, dtype=np.float64)
    logw = np.asarray(logw_gen, dtype=np.float64).reshape(-1)
    if len(x) != len(logw) or not len(x) or not len(y):
        raise ValueError("nonempty matching particle/weight arrays required")
    if not all(np.isfinite(a).all() for a in (x, y, logw)):
        raise FloatingPointError("nonfinite evaluation input")
    mass_pred, a, log_mass = particle_mass_and_weights(logw)
    if mass_true is None:
        mass_true = len(y) / float(n_initial_observed)
    b = np.full(len(y), 1 / len(y), dtype=np.float64)
    d2 = np.maximum(ot.dist(x, y, metric="sqeuclidean"), 0)
    w1, info1 = ot.emd2(a, b, np.sqrt(d2), numItermax=1_000_000, log=True)
    w2_squared, info2 = ot.emd2(a, b, d2, numItermax=1_000_000, log=True)
    for info in (info1, info2):
        if info.get("warning") or info.get("result_code", 1) != 1:
            raise RuntimeError(f"POT EMD did not succeed: {info}")
    return dict(W1_normalized=float(w1), W2_normalized=float(np.sqrt(max(w2_squared, 0))),
                mass_pred=mass_pred, mass_true=mass_true, log_total_mass=float(log_mass),
                mass_signed_error=mass_pred - mass_true, mass_abs_error=abs(mass_pred - mass_true),
                mass_relative_error=abs(mass_pred - mass_true) / mass_true,
                ESS=float(1 / np.square(a).sum()), POT_result_code_W1=int(info1.get("result_code", 1)),
                POT_result_code_W2=int(info2.get("result_code", 1)))

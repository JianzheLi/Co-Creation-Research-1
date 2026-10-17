"""Numerical helpers shared by local experiment evaluators."""

from __future__ import annotations

import warnings

import ot


def emd2_with_retry(a, b, cost) -> float:
    """Solve exact empirical OT, retrying if POT reaches its iteration cap."""
    num_iter_max = 1_000_000
    while True:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = float(ot.emd2(a, b, cost, numItermax=num_iter_max))
        not_converged = any(
            "numItermax reached before optimality" in str(w.message) for w in caught
        )
        if not not_converged:
            return value
        if num_iter_max >= 100_000_000:
            raise RuntimeError(
                "POT did not converge after 100 million network-simplex iterations"
            )
        num_iter_max *= 10
        print(
            f"POT reached its iteration cap; retrying full W1 with "
            f"numItermax={num_iter_max:,}",
            flush=True,
        )

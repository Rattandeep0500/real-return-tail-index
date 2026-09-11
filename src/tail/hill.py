from __future__ import annotations

import numpy as np


def hill_estimator(
    sample: np.ndarray,
    k: int,
    *,
    return_inverse: bool = False,
) -> float:
    x = np.asarray(sample, dtype=float)

    if x.ndim != 1:
        raise ValueError("sample must be one-dimensional.")

    if len(x) < 3:
        raise ValueError("sample must contain at least 3 observations.")

    if not np.all(np.isfinite(x)):
        raise ValueError("sample contains NaN or infinite values.")

    if np.any(x <= 0):
        raise ValueError("Hill estimation requires strictly positive observations.")

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")

    if not 1 <= k < len(x):
        raise ValueError("k must satisfy 1 <= k < len(sample).")

    x_sorted = np.sort(x)[::-1]
    threshold = x_sorted[k]

    log_excess = np.log(x_sorted[:k]) - np.log(threshold)
    gamma_hat = np.mean(log_excess)

    if gamma_hat <= 0:
        raise RuntimeError("Estimated Hill gamma is non-positive.")

    if return_inverse:
        return float(gamma_hat)

    return float(1.0 / gamma_hat)


def hill_curve(sample: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    k_values = np.asarray(k_values, dtype=int)
    return np.array(
        [hill_estimator(sample, int(k)) for k in k_values],
        dtype=float,
    )
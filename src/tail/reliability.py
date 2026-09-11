from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_inverse(values):
    values = np.asarray(values, dtype=float)

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum == minimum:
        return np.ones_like(values)

    return 1.0 - (values - minimum) / (maximum - minimum)


def compute_tail_reliability(
    k_values,
    alpha_values,
    bootstrap_std,
    window_size=15,
):
    k_values = np.asarray(k_values, dtype=int)
    alpha_values = np.asarray(alpha_values, dtype=float)
    bootstrap_std = np.asarray(bootstrap_std, dtype=float)

    if not (
        len(k_values)
        == len(alpha_values)
        == len(bootstrap_std)
    ):
        raise ValueError("All inputs must have the same length.")

    if len(alpha_values) < window_size:
        raise ValueError("Not enough observations for the stability window.")

    local_variation = np.full(len(alpha_values), np.nan)

    for i in range(window_size - 1, len(alpha_values)):
        window = alpha_values[
            i - window_size + 1:i + 1
        ]
        local_variation[i] = np.std(window, ddof=1)

    uncertainty_score = normalize_inverse(bootstrap_std)

    stability_score = np.full(
        len(local_variation),
        np.nan,
    )

    valid = np.isfinite(local_variation)

    if np.any(valid):
        minimum = np.min(local_variation[valid])
        maximum = np.max(local_variation[valid])

        if maximum == minimum:
            stability_score[valid] = 1.0
        else:
            stability_score[valid] = (
                1.0
                - (
                    local_variation[valid] - minimum
                )
                / (maximum - minimum)
            )

    reliability = (
        0.5 * stability_score
        + 0.5 * uncertainty_score
    )

    return pd.DataFrame(
        {
            "k": k_values,
            "alpha_hat": alpha_values,
            "bootstrap_std": bootstrap_std,
            "stability": local_variation,
            "stability_score": stability_score,
            "uncertainty_score": uncertainty_score,
            "reliability_score": reliability,
        }
    )


def select_reliable_k(result):
    valid = result.dropna(
        subset=["reliability_score"]
    )

    if valid.empty:
        raise RuntimeError("No valid reliability scores available.")

    best = valid.loc[
        valid["reliability_score"].idxmax()
    ]

    return (
        int(best["k"]),
        float(best["reliability_score"]),
    )
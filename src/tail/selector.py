from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_inverse(values):
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)

    if not np.any(valid):
        return np.full_like(values, np.nan)

    result = np.full_like(values, np.nan)

    minimum = np.min(values[valid])
    maximum = np.max(values[valid])

    if maximum == minimum:
        result[valid] = 1.0
        return result

    result[valid] = 1.0 - (
        (values[valid] - minimum)
        / (maximum - minimum)
    )

    return result


def compute_local_stability(alpha_values, window_size=15):
    alpha_values = np.asarray(alpha_values, dtype=float)

    stability = np.full(
        len(alpha_values),
        np.nan,
    )

    for i in range(window_size - 1, len(alpha_values)):
        window = alpha_values[
            i - window_size + 1:i + 1
        ]

        center = np.mean(window)

        if center == 0:
            stability[i] = np.inf
        else:
            stability[i] = np.std(
                window,
                ddof=1,
            ) / abs(center)

    return stability


def compute_tail_purity(
    k_values,
    sample_size,
    decay=4.0,
):
    k_values = np.asarray(k_values, dtype=float)

    fraction = k_values / sample_size

    return np.exp(-decay * fraction)


def compute_reliability_index(
    k_values,
    alpha_values,
    bootstrap_std,
    sample_size,
    window_size=15,
    stability_weight=0.40,
    uncertainty_weight=0.35,
    purity_weight=0.25,
    purity_decay=4.0,
):
    k_values = np.asarray(k_values, dtype=int)
    alpha_values = np.asarray(alpha_values, dtype=float)
    bootstrap_std = np.asarray(bootstrap_std, dtype=float)

    if not (
        len(k_values)
        == len(alpha_values)
        == len(bootstrap_std)
    ):
        raise ValueError(
            "Input arrays must have equal length."
        )

    stability = compute_local_stability(
        alpha_values,
        window_size,
    )

    stability_score = normalize_inverse(
        stability
    )

    uncertainty_score = normalize_inverse(
        bootstrap_std
    )

    purity_score = compute_tail_purity(
        k_values,
        sample_size,
        purity_decay,
    )

    reliability_score = (
        stability_weight * stability_score
        + uncertainty_weight * uncertainty_score
        + purity_weight * purity_score
    )

    return pd.DataFrame(
        {
            "k": k_values,
            "alpha_hat": alpha_values,
            "bootstrap_std": bootstrap_std,
            "local_cv": stability,
            "stability_score": stability_score,
            "uncertainty_score": uncertainty_score,
            "tail_purity_score": purity_score,
            "reliability_score": reliability_score,
        }
    )


def select_reliable_k(
    result,
    minimum_reliability=0.60,
    minimum_plateau=5,
):
    valid = result.dropna(
        subset=["reliability_score"]
    ).copy()

    if valid.empty:
        return {
            "status": "ABSTAIN",
            "k": None,
            "reliability": None,
            "reason": "No valid candidate thresholds.",
        }

    valid = valid.sort_values(
        "reliability_score",
        ascending=False,
    )

    best = valid.iloc[0]

    if best["reliability_score"] < minimum_reliability:
        return {
            "status": "ABSTAIN",
            "k": None,
            "reliability": float(
                best["reliability_score"]
            ),
            "reason": "Maximum reliability below threshold.",
        }

    top = valid[
        valid["reliability_score"]
        >= best["reliability_score"] * 0.98
    ]

    if len(top) < minimum_plateau:
        return {
            "status": "ABSTAIN",
            "k": None,
            "reliability": float(
                best["reliability_score"]
            ),
            "reason": "Insufficient stable candidate plateau.",
        }

    return {
        "status": "SELECT",
        "k": int(best["k"]),
        "reliability": float(
            best["reliability_score"]
        ),
        "reason": "Reliable threshold selected.",
    }
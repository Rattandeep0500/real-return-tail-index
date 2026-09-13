from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pareto_qq_deviation(sample, alpha_hat, k):
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]

    if len(x) <= k:
        return np.nan

    x = np.sort(x)[::-1]
    tail = x[:k]

    if alpha_hat <= 0:
        return np.nan

    empirical = np.arange(1, k + 1) / (k + 1)
    theoretical = 1.0 - empirical

    empirical_log = np.log(tail)
    theoretical_log = -np.log(
        1.0 - theoretical
    ) / alpha_hat

    empirical_log = (
        empirical_log
        - empirical_log.min()
    )

    theoretical_log = (
        theoretical_log
        - theoretical_log.min()
    )

    denominator = np.mean(
        np.abs(theoretical_log)
    )

    if denominator <= 0:
        return np.nan

    return np.mean(
        np.abs(
            empirical_log
            - theoretical_log
        )
    ) / denominator


def tail_mean_excess_deviation(
    sample,
    k,
):
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]

    if len(x) <= k + 2:
        return np.nan

    x = np.sort(x)[::-1]

    thresholds = x[
        k // 4:k
    ]

    empirical_mean_excess = []

    for threshold in thresholds:
        exceedances = x[
            x > threshold
        ]

        if len(exceedances) < 5:
            continue

        empirical_mean_excess.append(
            np.mean(
                exceedances - threshold
            )
        )

    if len(empirical_mean_excess) < 3:
        return np.nan

    values = np.asarray(
        empirical_mean_excess
    )

    return np.std(values) / (
        abs(np.mean(values))
        + 1e-12
    )


def threshold_stability_score(
    alpha_values,
):
    alpha_values = np.asarray(
        alpha_values,
        dtype=float,
    )

    alpha_values = alpha_values[
        np.isfinite(alpha_values)
        & (alpha_values > 0)
    ]

    if len(alpha_values) < 3:
        return np.nan

    mean_alpha = np.mean(
        alpha_values
    )

    if mean_alpha <= 0:
        return np.nan

    cv = np.std(
        alpha_values,
        ddof=1,
    ) / mean_alpha

    return cv


def hill_linearity_score(
    sample,
    k,
):
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]

    if len(x) <= k:
        return np.nan

    x = np.sort(x)[::-1][:k]

    ranks = np.arange(
        1,
        len(x) + 1,
    )

    log_x = np.log(x)
    log_rank = np.log(
        len(x) / ranks
    )

    if np.std(log_rank) == 0:
        return np.nan

    correlation = np.corrcoef(
        log_rank,
        log_x,
    )[0, 1]

    if not np.isfinite(correlation):
        return np.nan

    return 1.0 - abs(correlation)


def compute_model_validity(
    sample,
    k_values,
    alpha_values,
    window_size=5,
):
    sample = np.asarray(
        sample,
        dtype=float,
    )

    k_values = np.asarray(
        k_values,
        dtype=int,
    )

    alpha_values = np.asarray(
        alpha_values,
        dtype=float,
    )

    records = []

    for i, k in enumerate(
        k_values
    ):
        alpha_hat = alpha_values[i]

        local_start = max(
            0,
            i - window_size + 1,
        )

        local_alphas = (
            alpha_values[
                local_start:i + 1
            ]
        )

        stability = (
            threshold_stability_score(
                local_alphas
            )
        )

        qq_deviation = (
            pareto_qq_deviation(
                sample,
                alpha_hat,
                int(k),
            )
        )

        mean_excess = (
            tail_mean_excess_deviation(
                sample,
                int(k),
            )
        )

        nonlinearity = (
            hill_linearity_score(
                sample,
                int(k),
            )
        )

        records.append(
            {
                "k": int(k),
                "alpha_hat": float(
                    alpha_hat
                ),
                "threshold_stability": stability,
                "pareto_qq_deviation": qq_deviation,
                "mean_excess_deviation": mean_excess,
                "hill_nonlinearity": nonlinearity,
            }
        )

    result = pd.DataFrame(
        records
    )

    for column in [
        "threshold_stability",
        "pareto_qq_deviation",
        "mean_excess_deviation",
        "hill_nonlinearity",
    ]:
        values = result[
            column
        ].to_numpy(
            dtype=float
        )

        valid = np.isfinite(
            values
        )

        result[
            f"{column}_score"
        ] = np.nan

        if np.any(valid):
            minimum = np.min(
                values[valid]
            )

            maximum = np.max(
                values[valid]
            )

            if maximum == minimum:
                result.loc[
                    valid,
                    f"{column}_score",
                ] = 1.0
            else:
                result.loc[
                    valid,
                    f"{column}_score",
                ] = (
                    1.0
                    - (
                        values[valid]
                        - minimum
                    )
                    / (
                        maximum
                        - minimum
                    )
                )

    score_columns = [
        "threshold_stability_score",
        "pareto_qq_deviation_score",
        "mean_excess_deviation_score",
        "hill_nonlinearity_score",
    ]

    result["model_validity_score"] = (
        result[score_columns]
        .mean(axis=1)
    )

    return result


def select_model_validity(
    result,
    minimum_score=0.60,
):
    valid = result.dropna(
        subset=[
            "model_validity_score"
        ]
    )

    if valid.empty:
        return {
            "status": "ABSTAIN",
            "score": np.nan,
            "k": None,
        }

    best = valid.loc[
        valid[
            "model_validity_score"
        ].idxmax()
    ]

    score = float(
        best[
            "model_validity_score"
        ]
    )

    if score < minimum_score:
        return {
            "status": "ABSTAIN",
            "score": score,
            "k": None,
        }

    return {
        "status": "VALID",
        "score": score,
        "k": int(best["k"]),
    }
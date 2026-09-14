from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_tail(sample):
    x = np.asarray(
        sample,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
        & (x > 0)
    ]

    return np.sort(
        x
    )[::-1]


def tail_thresholds(
    sample,
    k_values,
):
    x = _clean_tail(
        sample
    )

    thresholds = []

    for k in k_values:
        if k < len(x):
            thresholds.append(
                x[int(k) - 1]
            )
        else:
            thresholds.append(
                np.nan
            )

    return np.asarray(
        thresholds,
        dtype=float,
    )


def log_excess_ratios(
    sample,
    k_values,
):
    x = _clean_tail(
        sample
    )

    values = []

    for k in k_values:
        k = int(k)

        if k + 1 >= len(x):
            values.append(
                np.nan
            )
            continue

        threshold = x[k]

        if threshold <= 0:
            values.append(
                np.nan
            )
            continue

        excess = (
            np.log(x[:k])
            - np.log(threshold)
        )

        values.append(
            np.mean(excess)
        )

    return np.asarray(
        values,
        dtype=float,
    )


def hill_curve(
    sample,
    k_values,
):
    x = _clean_tail(
        sample
    )

    estimates = []

    for k in k_values:
        k = int(k)

        if k >= len(x):
            estimates.append(
                np.nan
            )
            continue

        threshold = x[k]

        if threshold <= 0:
            estimates.append(
                np.nan
            )
            continue

        values = (
            np.log(x[:k])
            - np.log(threshold)
        )

        denominator = np.sum(
            values
        )

        if denominator <= 0:
            estimates.append(
                np.nan
            )
            continue

        estimates.append(
            k / denominator
        )

    return np.asarray(
        estimates,
        dtype=float,
    )


def hill_stability(
    alpha_values,
):
    alpha_values = np.asarray(
        alpha_values,
        dtype=float,
    )

    valid = np.isfinite(
        alpha_values
    ) & (
        alpha_values > 0
    )

    values = alpha_values[
        valid
    ]

    if len(values) < 3:
        return np.nan

    mean_value = np.mean(
        values
    )

    if mean_value <= 0:
        return np.nan

    return float(
        np.std(
            values,
            ddof=1,
        )
        / mean_value
    )


def pareto_loglog_fit(
    sample,
    k,
):
    x = _clean_tail(
        sample
    )

    k = int(k)

    if k < 5 or k >= len(x):
        return {
            "slope": np.nan,
            "r2": np.nan,
            "max_abs_residual": np.nan,
        }

    tail = x[:k]

    ranks = np.arange(
        1,
        k + 1,
        dtype=float,
    )

    log_x = np.log(
        tail
    )

    log_survival = -np.log(
        ranks / (k + 1.0)
    )

    finite = (
        np.isfinite(log_x)
        & np.isfinite(log_survival)
    )

    if finite.sum() < 5:
        return {
            "slope": np.nan,
            "r2": np.nan,
            "max_abs_residual": np.nan,
        }

    x_values = log_x[
        finite
    ]

    y_values = log_survival[
        finite
    ]

    slope, intercept = np.polyfit(
        x_values,
        y_values,
        1,
    )

    fitted = (
        slope
        * x_values
        + intercept
    )

    residuals = (
        y_values
        - fitted
    )

    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (
            y_values
            - np.mean(y_values)
        ) ** 2
    )

    if ss_tot <= 0:
        r2 = np.nan
    else:
        r2 = 1.0 - (
            ss_res / ss_tot
        )

    return {
        "slope": float(
            slope
        ),
        "r2": float(
            r2
        ),
        "max_abs_residual": float(
            np.max(
                np.abs(
                    residuals
                )
            )
        ),
    }


def threshold_sensitivity(
    alpha_values,
):
    alpha_values = np.asarray(
        alpha_values,
        dtype=float,
    )

    valid = np.isfinite(
        alpha_values
    ) & (
        alpha_values > 0
    )

    if valid.sum() < 3:
        return {
            "relative_range": np.nan,
            "relative_std": np.nan,
            "mean_abs_change": np.nan,
        }

    values = alpha_values[
        valid
    ]

    mean_value = np.mean(
        values
    )

    if mean_value <= 0:
        return {
            "relative_range": np.nan,
            "relative_std": np.nan,
            "mean_abs_change": np.nan,
        }

    changes = np.diff(
        values
    )

    return {
        "relative_range": float(
            (
                np.max(values)
                - np.min(values)
            )
            / mean_value
        ),
        "relative_std": float(
            np.std(
                values,
                ddof=1,
            )
            / mean_value
        ),
        "mean_abs_change": float(
            np.mean(
                np.abs(changes)
            )
            / mean_value
        ),
    }


def mean_excess_linearity(
    sample,
    k_values,
):
    x = _clean_tail(
        sample
    )

    points = []

    for k in k_values:
        k = int(k)

        if k + 1 >= len(x):
            continue

        threshold = x[k]

        if threshold <= 0:
            continue

        exceedances = x[
            :k
        ]

        mean_excess = np.mean(
            exceedances
            - threshold
        )

        if (
            np.isfinite(
                mean_excess
            )
            and mean_excess >= 0
        ):
            points.append(
                (
                    np.log(
                        threshold
                    ),
                    np.log(
                        mean_excess
                        + 1e-12
                    ),
                )
            )

    if len(points) < 4:
        return np.nan

    points = np.asarray(
        points,
        dtype=float,
    )

    slope, intercept = np.polyfit(
        points[:, 0],
        points[:, 1],
        1,
    )

    fitted = (
        slope
        * points[:, 0]
        + intercept
    )

    residuals = (
        points[:, 1]
        - fitted
    )

    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (
            points[:, 1]
            - np.mean(
                points[:, 1]
            )
        ) ** 2
    )

    if ss_tot <= 0:
        return np.nan

    return float(
        1.0
        - ss_res / ss_tot
    )


def score_component(
    value,
    good,
    bad,
):
    if not np.isfinite(
        value
    ):
        return np.nan

    if good <= bad:
        return np.nan

    score = (
        value - bad
    ) / (
        good - bad
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def compute_tail_model_features(
    sample,
    k_values,
):
    k_values = np.asarray(
        k_values,
        dtype=int,
    )

    alphas = hill_curve(
        sample,
        k_values,
    )

    stability = threshold_sensitivity(
        alphas
    )

    selected_k = int(
        k_values[
            np.nanargmin(
                np.abs(
                    alphas
                    - np.nanmedian(
                        alphas
                    )
                )
            )
        ]
    )

    loglog = pareto_loglog_fit(
        sample,
        selected_k,
    )

    me_linearity = (
        mean_excess_linearity(
            sample,
            k_values,
        )
    )

    features = {
        "hill_alpha_median": float(
            np.nanmedian(
                alphas
            )
        ),
        "hill_alpha_min": float(
            np.nanmin(
                alphas
            )
        ),
        "hill_alpha_max": float(
            np.nanmax(
                alphas
            )
        ),
        "hill_relative_range": stability[
            "relative_range"
        ],
        "hill_relative_std": stability[
            "relative_std"
        ],
        "hill_mean_abs_change": stability[
            "mean_abs_change"
        ],
        "pareto_loglog_slope": loglog[
            "slope"
        ],
        "pareto_loglog_r2": loglog[
            "r2"
        ],
        "pareto_loglog_max_residual": loglog[
            "max_abs_residual"
        ],
        "mean_excess_loglog_r2": me_linearity,
        "selected_k": selected_k,
    }

    return (
        features,
        alphas,
    )


def model_validity_score(
    features,
):
    r2 = features[
        "pareto_loglog_r2"
    ]

    relative_range = features[
        "hill_relative_range"
    ]

    relative_std = features[
        "hill_relative_std"
    ]

    residual = features[
        "pareto_loglog_max_residual"
    ]

    mean_excess_r2 = features[
        "mean_excess_loglog_r2"
    ]

    scores = []

    if np.isfinite(r2):
        scores.append(
            np.clip(
                r2,
                0.0,
                1.0,
            )
        )

    if np.isfinite(
        relative_range
    ):
        scores.append(
            np.exp(
                -3.0
                * max(
                    relative_range,
                    0.0,
                )
            )
        )

    if np.isfinite(
        relative_std
    ):
        scores.append(
            np.exp(
                -3.0
                * max(
                    relative_std,
                    0.0,
                )
            )
        )

    if np.isfinite(
        residual
    ):
        scores.append(
            np.exp(
                -residual
            )
        )

    if np.isfinite(
        mean_excess_r2
    ):
        scores.append(
            np.clip(
                mean_excess_r2,
                0.0,
                1.0,
            )
        )

    if not scores:
        return np.nan

    return float(
        np.mean(scores)
    )


def classify_tail_model(
    score,
    high_threshold=0.70,
    low_threshold=0.45,
):
    if not np.isfinite(
        score
    ):
        return "UNKNOWN"

    if score >= high_threshold:
        return "SUPPORTED"

    if score <= low_threshold:
        return "WEAK"

    return "UNCERTAIN"


def analyze_tail_model(
    sample,
    k_values,
):
    features, alphas = (
        compute_tail_model_features(
            sample,
            k_values,
        )
    )

    score = model_validity_score(
        features
    )

    status = classify_tail_model(
        score
    )

    result = dict(
        features
    )

    result[
        "model_validity_score"
    ] = score

    result[
        "model_status"
    ] = status

    return (
        result,
        alphas,
    )
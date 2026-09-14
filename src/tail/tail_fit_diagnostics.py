from __future__ import annotations

import numpy as np
import pandas as pd


def clean_tail(sample):
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return np.sort(x)[::-1]


def select_threshold(sample, k):
    x = clean_tail(sample)
    k = int(k)

    if k < 5 or k >= len(x):
        return np.nan

    return float(x[k])


def hill_estimate(sample, k):
    x = clean_tail(sample)
    k = int(k)

    if k < 2 or k >= len(x):
        return np.nan

    threshold = x[k]

    if threshold <= 0:
        return np.nan

    logs = np.log(x[:k]) - np.log(threshold)
    denominator = np.sum(logs)

    if denominator <= 0:
        return np.nan

    return float(k / denominator)


def pareto_ks_distance(sample, k):
    x = clean_tail(sample)
    k = int(k)

    if k < 5 or k >= len(x):
        return np.nan

    threshold = x[k]

    if threshold <= 0:
        return np.nan

    tail = np.sort(x[:k])

    alpha = hill_estimate(x, k)

    if not np.isfinite(alpha):
        return np.nan

    fitted_cdf = 1.0 - (
        tail / threshold
    ) ** (-alpha)

    empirical_upper = (
        np.arange(1, k + 1, dtype=float)
        / k
    )

    empirical_lower = (
        np.arange(0, k, dtype=float)
        / k
    )

    d_plus = np.max(
        empirical_upper
        - fitted_cdf
    )

    d_minus = np.max(
        fitted_cdf
        - empirical_lower
    )

    return float(
        max(
            d_plus,
            d_minus,
        )
    )


def pareto_loglikelihood(sample, k):
    x = clean_tail(sample)
    k = int(k)

    if k < 5 or k >= len(x):
        return np.nan

    threshold = x[k]

    if threshold <= 0:
        return np.nan

    tail = x[:k]

    alpha = hill_estimate(x, k)

    if not np.isfinite(alpha):
        return np.nan

    log_excess = (
        np.log(tail)
        - np.log(threshold)
    )

    return float(
        k * np.log(alpha)
        - k * np.log(threshold)
        - (alpha + 1.0)
        * np.sum(log_excess)
    )


def threshold_stability(sample, k_values):
    estimates = []

    for k in k_values:
        estimate = hill_estimate(
            sample,
            k,
        )

        if np.isfinite(estimate):
            estimates.append(
                estimate
            )

    if len(estimates) < 3:
        return np.nan

    estimates = np.asarray(
        estimates,
        dtype=float,
    )

    mean_alpha = np.mean(
        estimates
    )

    if mean_alpha <= 0:
        return np.nan

    return float(
        np.std(
            estimates,
            ddof=1,
        )
        / mean_alpha
    )


def bootstrap_hill_stability(
    sample,
    k,
    bootstraps=50,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(42)

    x = clean_tail(sample)

    if len(x) < 10:
        return np.nan

    values = []

    for _ in range(
        bootstraps
    ):
        bootstrap_sample = rng.choice(
            x,
            size=len(x),
            replace=True,
        )

        estimate = hill_estimate(
            bootstrap_sample,
            k,
        )

        if np.isfinite(
            estimate
        ):
            values.append(
                estimate
            )

    if len(values) < 3:
        return np.nan

    values = np.asarray(
        values,
        dtype=float,
    )

    mean_alpha = np.mean(
        values
    )

    if mean_alpha <= 0:
        return np.nan

    return float(
        np.std(
            values,
            ddof=1,
        )
        / mean_alpha
    )


def mean_excess_fit(
    sample,
    k_values,
):
    x = clean_tail(sample)

    points = []

    for k in k_values:
        k = int(k)

        if k >= len(x):
            continue

        threshold = x[k]

        if threshold <= 0:
            continue

        excess = (
            x[:k]
            - threshold
        )

        mean_excess = np.mean(
            excess
        )

        if (
            np.isfinite(
                mean_excess
            )
            and mean_excess > 0
        ):
            points.append(
                (
                    np.log(
                        threshold
                    ),
                    np.log(
                        mean_excess
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


def build_diagnostic_table(
    sample,
    k_values,
    bootstraps=50,
    rng=None,
):
    x = clean_tail(
        sample
    )

    rows = []

    for k in k_values:
        k = int(k)

        if k < 5 or k >= len(x):
            continue

        alpha = hill_estimate(
            x,
            k,
        )

        ks = pareto_ks_distance(
            x,
            k,
        )

        loglik = pareto_loglikelihood(
            x,
            k,
        )

        bootstrap_cv = (
            bootstrap_hill_stability(
                x,
                k,
                bootstraps,
                rng,
            )
        )

        threshold = (
            select_threshold(
                x,
                k,
            )
        )

        rows.append(
            {
                "k": k,
                "k_fraction": (
                    k / len(x)
                ),
                "threshold": threshold,
                "alpha_hat": alpha,
                "pareto_ks": ks,
                "pareto_loglikelihood": loglik,
                "bootstrap_cv": bootstrap_cv,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result) == 0:
        return result

    result[
        "threshold_stability"
    ] = threshold_stability(
        x,
        k_values,
    )

    result[
        "mean_excess_r2"
    ] = mean_excess_fit(
        x,
        k_values,
    )

    return result


def best_pareto_threshold(
    diagnostic_table,
):
    data = diagnostic_table.copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[
            "pareto_ks",
            "bootstrap_cv",
        ]
    )

    if len(data) == 0:
        return None

    data = data[
        data[
            "k_fraction"
        ]
        <= 0.50
    ]

    if len(data) == 0:
        return None

    ks = data[
        "pareto_ks"
    ].to_numpy(
        dtype=float
    )

    cvs = data[
        "bootstrap_cv"
    ].to_numpy(
        dtype=float
    )

    ks_scale = np.median(
        ks
    )

    cv_scale = np.median(
        cvs
    )

    if (
        not np.isfinite(
            ks_scale
        )
        or ks_scale <= 0
    ):
        ks_scale = 1.0

    if (
        not np.isfinite(
            cv_scale
        )
        or cv_scale <= 0
    ):
        cv_scale = 1.0

    data[
        "combined_score"
    ] = (
        0.50
        * data[
            "pareto_ks"
        ]
        / ks_scale
        + 0.50
        * data[
            "bootstrap_cv"
        ]
        / cv_scale
    )

    row = data.loc[
        data[
            "combined_score"
        ].idxmin()
    ]

    return row.to_dict()


def diagnostic_summary(
    sample,
    k_values,
    bootstraps=50,
    rng=None,
):
    table = build_diagnostic_table(
        sample,
        k_values,
        bootstraps,
        rng,
    )

    if len(table) == 0:
        return {
            "selected_k": np.nan,
            "alpha_hat": np.nan,
            "pareto_ks": np.nan,
            "bootstrap_cv": np.nan,
            "threshold_stability": np.nan,
            "mean_excess_r2": np.nan,
        }

    best = best_pareto_threshold(
        table
    )

    if best is None:
        return {
            "selected_k": np.nan,
            "alpha_hat": np.nan,
            "pareto_ks": np.nan,
            "bootstrap_cv": np.nan,
            "threshold_stability": np.nan,
            "mean_excess_r2": np.nan,
        }

    return {
        "selected_k": best[
            "k"
        ],
        "alpha_hat": best[
            "alpha_hat"
        ],
        "pareto_ks": best[
            "pareto_ks"
        ],
        "bootstrap_cv": best[
            "bootstrap_cv"
        ],
        "threshold_stability": best[
            "threshold_stability"
        ],
        "mean_excess_r2": best[
            "mean_excess_r2"
        ],
    }


def validity_score(
    pareto_ks,
    bootstrap_cv,
    threshold_stability,
    mean_excess_r2,
):
    components = []

    if np.isfinite(
        pareto_ks
    ):
        components.append(
            np.exp(
                -8.0
                * max(
                    pareto_ks,
                    0.0,
                )
            )
        )

    if np.isfinite(
        bootstrap_cv
    ):
        components.append(
            np.exp(
                -3.0
                * max(
                    bootstrap_cv,
                    0.0,
                )
            )
        )

    if np.isfinite(
        threshold_stability
    ):
        components.append(
            np.exp(
                -3.0
                * max(
                    threshold_stability,
                    0.0,
                )
            )
        )

    if np.isfinite(
        mean_excess_r2
    ):
        components.append(
            np.clip(
                mean_excess_r2,
                0.0,
                1.0,
            )
        )

    if not components:
        return np.nan

    return float(
        np.mean(
            components
        )
    )


def classify_validity(
    score,
    supported_threshold=0.70,
    weak_threshold=0.45,
):
    if not np.isfinite(
        score
    ):
        return "UNKNOWN"

    if score >= supported_threshold:
        return "SUPPORTED"

    if score <= weak_threshold:
        return "WEAK"

    return "UNCERTAIN"


def analyze_tail_fit(
    sample,
    k_values,
    bootstraps=50,
    rng=None,
):
    summary = diagnostic_summary(
        sample,
        k_values,
        bootstraps,
        rng,
    )

    score = validity_score(
        summary[
            "pareto_ks"
        ],
        summary[
            "bootstrap_cv"
        ],
        summary[
            "threshold_stability"
        ],
        summary[
            "mean_excess_r2"
        ],
    )

    status = classify_validity(
        score
    )

    summary[
        "validity_score"
    ] = score

    summary[
        "validity_status"
    ] = status

    return summary
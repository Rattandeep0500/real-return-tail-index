from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "sp500_real_returns.csv"
)

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "tail_stability_surface.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "tail_stability_surface_summary.csv"
)

K_VALUES = np.arange(
    10,
    81,
    5,
)

WINDOW_SIZE = 120
BOOTSTRAPS = 40
MIN_STABLE_POINTS = 3

KS_THRESHOLD = 0.15
BOOTSTRAP_CV_THRESHOLD = 0.25


def clean_sample(sample):
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


def hill_estimate(sample, k):
    x = clean_sample(
        sample
    )

    k = int(k)

    if k < 2 or k >= len(x):
        return np.nan

    threshold = x[k]

    if threshold <= 0:
        return np.nan

    denominator = np.sum(
        np.log(
            x[:k]
        )
        - np.log(
            threshold
        )
    )

    if denominator <= 0:
        return np.nan

    return float(
        k / denominator
    )


def pareto_ks(sample, k):
    x = clean_sample(
        sample
    )

    k = int(k)

    if k < 5 or k >= len(x):
        return np.nan

    threshold = x[k]

    if threshold <= 0:
        return np.nan

    tail = np.sort(
        x[:k]
    )

    alpha = hill_estimate(
        x,
        k,
    )

    if not np.isfinite(alpha):
        return np.nan

    fitted_cdf = (
        1.0
        - (
            tail
            / threshold
        ) ** (-alpha)
    )

    empirical_upper = (
        np.arange(
            1,
            k + 1,
            dtype=float,
        )
        / k
    )

    empirical_lower = (
        np.arange(
            0,
            k,
            dtype=float,
        )
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


def bootstrap_cv(
    sample,
    k,
    rng,
):
    x = clean_sample(
        sample
    )

    if len(x) < 20:
        return np.nan

    values = []

    for _ in range(
        BOOTSTRAPS
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


def build_surface(
    sample,
    rng,
):
    rows = []

    for k in K_VALUES:
        alpha = hill_estimate(
            sample,
            k,
        )

        ks = pareto_ks(
            sample,
            k,
        )

        cv = bootstrap_cv(
            sample,
            k,
            rng,
        )

        stable = (
            np.isfinite(alpha)
            and np.isfinite(ks)
            and np.isfinite(cv)
            and ks <= KS_THRESHOLD
            and cv <= BOOTSTRAP_CV_THRESHOLD
        )

        rows.append(
            {
                "k": int(k),
                "alpha_hat": alpha,
                "pareto_ks": ks,
                "bootstrap_cv": cv,
                "stable": bool(
                    stable
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_surface(
    surface,
):
    valid = surface.dropna(
        subset=[
            "alpha_hat",
            "pareto_ks",
            "bootstrap_cv",
        ]
    ).copy()

    stable = valid[
        valid[
            "stable"
        ]
    ]

    if len(valid) == 0:
        return {
            "stable_fraction": np.nan,
            "stable_k_min": np.nan,
            "stable_k_max": np.nan,
            "stable_k_center": np.nan,
            "alpha_stability_cv": np.nan,
            "alpha_range": np.nan,
            "ks_min": np.nan,
            "bootstrap_cv_min": np.nan,
            "support_status": "UNKNOWN",
        }

    alpha_values = valid[
        "alpha_hat"
    ].to_numpy(
        dtype=float
    )

    alpha_mean = np.mean(
        alpha_values
    )

    if alpha_mean > 0:
        alpha_cv = (
            np.std(
                alpha_values,
                ddof=1,
            )
            / alpha_mean
        )
    else:
        alpha_cv = np.nan

    alpha_range = (
        np.max(
            alpha_values
        )
        - np.min(
            alpha_values
        )
    )

    if len(stable) >= MIN_STABLE_POINTS:
        stable_k_min = int(
            stable[
                "k"
            ].min()
        )

        stable_k_max = int(
            stable[
                "k"
            ].max()
        )

        stable_k_center = float(
            stable[
                "k"
            ].median()
        )

        support_status = (
            "SUPPORTED"
        )
    else:
        stable_k_min = np.nan
        stable_k_max = np.nan
        stable_k_center = np.nan
        support_status = (
            "WEAK"
        )

    return {
        "stable_fraction": (
            len(stable)
            / len(valid)
        ),
        "stable_k_min": stable_k_min,
        "stable_k_max": stable_k_max,
        "stable_k_center": stable_k_center,
        "alpha_stability_cv": alpha_cv,
        "alpha_range": alpha_range,
        "ks_min": valid[
            "pareto_ks"
        ].min(),
        "bootstrap_cv_min": valid[
            "bootstrap_cv"
        ].min(),
        "support_status": support_status,
    }


def main():
    data = pd.read_csv(
        INPUT_FILE,
        index_col=0,
    )

    data = data.reset_index()

    data = data.rename(
        columns={
            data.columns[0]: "date"
        }
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    if "real_return" not in data.columns:
        raise ValueError(
            "real_return column not found."
        )

    returns = data[
        "real_return"
    ].to_numpy(
        dtype=float
    )

    rng = np.random.default_rng(
        20260914
    )

    surface_rows = []
    summary_rows = []

    max_start = (
        len(returns)
        - WINDOW_SIZE
        + 1
    )

    if max_start <= 0:
        raise ValueError(
            "Not enough observations for the requested window size."
        )

    for start in range(
        max_start
    ):
        end = (
            start
            + WINDOW_SIZE
        )

        window = returns[
            start:end
        ]

        date = data[
            "date"
        ].iloc[
            end - 1
        ]

        window_start = data[
            "date"
        ].iloc[
            start
        ]

        surface = build_surface(
            window,
            rng,
        )

        summary = summarize_surface(
            surface
        )

        summary[
            "date"
        ] = date

        summary[
            "window_start"
        ] = window_start

        summary[
            "window_end"
        ] = date

        summary[
            "window_size"
        ] = WINDOW_SIZE

        summary_rows.append(
            summary
        )

        for _, row in surface.iterrows():
            surface_rows.append(
                {
                    "date": date,
                    "window_start": window_start,
                    "window_size": WINDOW_SIZE,
                    "k": row[
                        "k"
                    ],
                    "alpha_hat": row[
                        "alpha_hat"
                    ],
                    "pareto_ks": row[
                        "pareto_ks"
                    ],
                    "bootstrap_cv": row[
                        "bootstrap_cv"
                    ],
                    "stable": row[
                        "stable"
                    ],
                }
            )

    surface_data = pd.DataFrame(
        surface_rows
    )

    summary_data = pd.DataFrame(
        summary_rows
    )

    surface_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary_data.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    supported_rate = (
        summary_data[
            "support_status"
        ]
        == "SUPPORTED"
    ).mean()

    weak_rate = (
        summary_data[
            "support_status"
        ]
        == "WEAK"
    ).mean()

    mean_stable_fraction = (
        summary_data[
            "stable_fraction"
        ].mean()
    )

    print(
        f"Observations: {len(summary_data)}"
    )

    print(
        f"Window size: {WINDOW_SIZE}"
    )

    print(
        f"k values tested: {len(K_VALUES)}"
    )

    print(
        f"Mean stable fraction: "
        f"{mean_stable_fraction:.6f}"
    )

    print(
        f"Supported windows: "
        f"{supported_rate:.6f}"
    )

    print(
        f"Weak windows: "
        f"{weak_rate:.6f}"
    )

    print(
        f"Surface rows: {len(surface_data)}"
    )

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
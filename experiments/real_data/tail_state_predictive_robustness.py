from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "tables"
    / "tail_state_predictive_predictions.csv"
)

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "tail_state_predictive_robustness.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "tail_state_predictive_robustness_summary.csv"
)


RNG = np.random.default_rng(
    20260914
)

N_BOOTSTRAP = 5000

BLOCK_SIZE = 6

CONFIDENCE_LEVEL = 0.95

TARGETS = [
    "target_1m_extreme",
    "target_1m_severe",
    "target_3m_loss",
    "target_6m_loss",
]


def load_predictions():
    data = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "date",
        "actual",
        "predicted_probability",
        "model",
        "target",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = data.sort_values(
        [
            "target",
            "date",
            "model",
        ]
    ).reset_index(
        drop=True
    )

    return data


def prepare_paired_data(
    data,
    target,
):
    subset = data[
        data["target"]
        == target
    ].copy()

    tail = subset[
        subset["model"]
        == "Tail-State"
    ][
        [
            "date",
            "actual",
            "predicted_probability",
        ]
    ].rename(
        columns={
            "predicted_probability":
                "tail_probability"
        }
    )

    volatility = subset[
        subset["model"]
        == "Volatility-Baseline"
    ][
        [
            "date",
            "actual",
            "predicted_probability",
        ]
    ].rename(
        columns={
            "predicted_probability":
                "volatility_probability"
        }
    )

    paired = tail.merge(
        volatility,
        on="date",
        how="inner",
        suffixes=(
            "_tail",
            "_volatility",
        ),
    )

    paired = paired[
        [
            "date",
            "actual_tail",
            "tail_probability",
            "volatility_probability",
        ]
    ].copy()

    paired = paired.rename(
        columns={
            "actual_tail":
                "actual",
        }
    )

    paired["actual"] = pd.to_numeric(
        paired["actual"],
        errors="coerce",
    )

    paired["tail_probability"] = (
        pd.to_numeric(
            paired[
                "tail_probability"
            ],
            errors="coerce",
        )
    )

    paired["volatility_probability"] = (
        pd.to_numeric(
            paired[
                "volatility_probability"
            ],
            errors="coerce",
        )
    )

    paired = paired.dropna()

    paired = paired.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    return paired


def calculate_auc(
    y,
    scores,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if len(
        np.unique(y)
    ) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            scores,
        )
    )


def paired_auc_difference(
    data,
):
    y = data[
        "actual"
    ].to_numpy(
        dtype=int
    )

    tail_scores = data[
        "tail_probability"
    ].to_numpy(
        dtype=float
    )

    volatility_scores = data[
        "volatility_probability"
    ].to_numpy(
        dtype=float
    )

    tail_auc = calculate_auc(
        y,
        tail_scores,
    )

    volatility_auc = calculate_auc(
        y,
        volatility_scores,
    )

    return (
        tail_auc,
        volatility_auc,
        tail_auc
        - volatility_auc,
    )


def moving_block_indices(
    n,
    block_size,
):
    if n <= 0:
        return np.array(
            [],
            dtype=int,
        )

    if block_size >= n:
        return np.arange(
            n
        )

    starts = RNG.integers(
        0,
        n - block_size + 1,
        size=int(
            np.ceil(
                n / block_size
            )
        ),
    )

    indices = []

    for start in starts:
        indices.extend(
            range(
                start,
                start + block_size,
            )
        )

        if len(indices) >= n:
            break

    return np.asarray(
        indices[
            :n
        ],
        dtype=int,
    )


def bootstrap_auc_difference(
    data,
):
    y = data[
        "actual"
    ].to_numpy(
        dtype=int
    )

    tail_scores = data[
        "tail_probability"
    ].to_numpy(
        dtype=float
    )

    volatility_scores = data[
        "volatility_probability"
    ].to_numpy(
        dtype=float
    )

    n = len(y)

    if n < 20:
        return {
            "bootstrap_mean_delta_auc": np.nan,
            "bootstrap_std_delta_auc": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "prob_delta_positive": np.nan,
            "n_bootstrap_valid": 0,
        }

    values = []

    for _ in range(
        N_BOOTSTRAP
    ):
        indices = moving_block_indices(
            n,
            BLOCK_SIZE,
        )

        y_boot = y[
            indices
        ]

        tail_boot = tail_scores[
            indices
        ]

        volatility_boot = (
            volatility_scores[
                indices
            ]
        )

        if len(
            np.unique(y_boot)
        ) < 2:
            continue

        tail_auc = calculate_auc(
            y_boot,
            tail_boot,
        )

        volatility_auc = calculate_auc(
            y_boot,
            volatility_boot,
        )

        if (
            np.isfinite(
                tail_auc
            )
            and np.isfinite(
                volatility_auc
            )
        ):
            values.append(
                tail_auc
                - volatility_auc
            )

    if len(values) < 100:
        return {
            "bootstrap_mean_delta_auc": np.nan,
            "bootstrap_std_delta_auc": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "prob_delta_positive": np.nan,
            "n_bootstrap_valid": len(values),
        }

    values = np.asarray(
        values,
        dtype=float,
    )

    alpha = (
        1.0
        - CONFIDENCE_LEVEL
    )

    lower = np.quantile(
        values,
        alpha / 2.0,
    )

    upper = np.quantile(
        values,
        1.0 - alpha / 2.0,
    )

    return {
        "bootstrap_mean_delta_auc": float(
            np.mean(values)
        ),
        "bootstrap_std_delta_auc": float(
            np.std(
                values,
                ddof=1,
            )
        ),
        "ci_lower": float(
            lower
        ),
        "ci_upper": float(
            upper
        ),
        "prob_delta_positive": float(
            np.mean(
                values > 0
            )
        ),
        "n_bootstrap_valid": len(
            values
        ),
    }


def main():
    data = load_predictions()

    results = []

    for target in TARGETS:
        paired = prepare_paired_data(
            data,
            target,
        )

        if len(paired) == 0:
            continue

        tail_auc, volatility_auc, delta_auc = (
            paired_auc_difference(
                paired
            )
        )

        bootstrap = (
            bootstrap_auc_difference(
                paired
            )
        )

        results.append(
            {
                "target": target,
                "observations": len(
                    paired
                ),
                "event_rate": paired[
                    "actual"
                ].mean(),
                "tail_state_auc": tail_auc,
                "volatility_baseline_auc": volatility_auc,
                "observed_delta_auc": delta_auc,
                **bootstrap,
            }
        )

    result = pd.DataFrame(
        results
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    significant = (
        (
            result[
                "ci_lower"
            ]
            > 0
        )
        | (
            result[
                "ci_upper"
            ]
            < 0
        )
    )

    result[
        "bootstrap_significant"
    ] = significant

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    overall = pd.DataFrame(
        [
            {
                "targets_tested": len(
                    result
                ),
                "targets_with_positive_delta": int(
                    (
                        result[
                            "observed_delta_auc"
                        ]
                        > 0
                    ).sum()
                ),
                "targets_significantly_positive": int(
                    (
                        (
                            result[
                                "ci_lower"
                            ]
                            > 0
                        )
                    ).sum()
                ),
                "targets_significantly_negative": int(
                    (
                        (
                            result[
                                "ci_upper"
                            ]
                            < 0
                        )
                    ).sum()
                ),
                "mean_observed_delta_auc": result[
                    "observed_delta_auc"
                ].mean(),
                "mean_bootstrap_delta_auc": result[
                    "bootstrap_mean_delta_auc"
                ].mean(),
            }
        ]
    )

    overall.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "M2.3 PREDICTIVE ROBUSTNESS"
    )

    print(
        "=========================="
    )

    print()

    print(
        result.to_string(
            index=False
        )
    )

    print()

    print(
        "Overall robustness summary:"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print()

    print(
        f"Block size: {BLOCK_SIZE}"
    )

    print(
        f"Bootstrap replications: {N_BOOTSTRAP}"
    )

    print(
        f"Confidence level: {CONFIDENCE_LEVEL}"
    )

    print()

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
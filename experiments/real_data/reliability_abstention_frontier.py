from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "tables"
    / "dynamic_reliable_tail.csv"
)

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "reliability_abstention_frontier.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "reliability_abstention_frontier_summary.csv"
)

REAL_RETURN_FILE = (
    ROOT
    / "data"
    / "processed"
    / "sp500_real_returns.csv"
)

RELIABILITY_THRESHOLDS = np.round(
    np.arange(
        0.00,
        1.01,
        0.05,
    ),
    2,
)

SEVERE_LOSS_THRESHOLD = -0.05

EXTREME_LOSS_THRESHOLD = -0.075


def clean_numeric(
    series,
):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return values.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def load_dynamic_data():
    data = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "date",
        "left_alpha",
        "right_alpha",
        "left_joint_reliability",
        "right_joint_reliability",
        "left_reliable",
        "right_reliable",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    numeric_columns = [
        column
        for column in required
        if column != "date"
    ]

    for column in numeric_columns:
        data[column] = clean_numeric(
            data[column]
        )

    data = data.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    return data


def load_returns():
    returns = pd.read_csv(
        REAL_RETURN_FILE,
        index_col=0,
    )

    returns = returns.reset_index()

    returns = returns.rename(
        columns={
            returns.columns[0]: "date"
        }
    )

    returns["date"] = pd.to_datetime(
        returns["date"]
    )

    returns["real_return"] = clean_numeric(
        returns["real_return"]
    )

    returns = returns[
        [
            "date",
            "real_return",
        ]
    ].sort_values(
        "date"
    )

    return returns


def build_features(
    data,
    returns,
):
    data = data.copy()

    data[
        "reliability_score"
    ] = np.minimum(
        data[
            "left_joint_reliability"
        ],
        data[
            "right_joint_reliability"
        ],
    )

    data[
        "mean_side_reliability"
    ] = (
        data[
            [
                "left_joint_reliability",
                "right_joint_reliability",
            ]
        ]
        .mean(
            axis=1
        )
    )

    data[
        "minimum_tail_alpha"
    ] = data[
        [
            "left_alpha",
            "right_alpha",
        ]
    ].min(
        axis=1
    )

    data[
        "maximum_tail_alpha"
    ] = data[
        [
            "left_alpha",
            "right_alpha",
        ]
    ].max(
        axis=1
    )

    data[
        "alpha_asymmetry"
    ] = (
        data[
            "right_alpha"
        ]
        - data[
            "left_alpha"
        ]
    ) / (
        data[
            "right_alpha"
        ]
        + data[
            "left_alpha"
        ]
    )

    data[
        "next_left_alpha"
    ] = data[
        "left_alpha"
    ].shift(
        -1
    )

    data[
        "next_right_alpha"
    ] = data[
        "right_alpha"
    ].shift(
        -1
    )

    data[
        "next_alpha_change"
    ] = np.maximum(
        np.abs(
            data[
                "next_left_alpha"
            ]
            - data[
                "left_alpha"
            ]
        ),
        np.abs(
            data[
                "next_right_alpha"
            ]
            - data[
                "right_alpha"
            ]
        ),
    )

    data[
        "next_alpha_deterioration"
    ] = np.maximum(
        (
            data[
                "left_alpha"
            ]
            - data[
                "next_left_alpha"
            ]
        ),
        (
            data[
                "right_alpha"
            ]
            - data[
                "next_right_alpha"
            ]
        ),
    )

    merged = pd.merge(
        data,
        returns,
        on="date",
        how="left",
    )

    merged[
        "future_severe_loss"
    ] = (
        merged[
            "real_return"
        ].shift(
            -1
        )
        <= SEVERE_LOSS_THRESHOLD
    )

    merged[
        "future_extreme_loss"
    ] = (
        merged[
            "real_return"
        ].shift(
            -1
        )
        <= EXTREME_LOSS_THRESHOLD
    )

    return merged


def evaluate_threshold(
    data,
    threshold,
):
    valid = data[
        np.isfinite(
            data[
                "reliability_score"
            ]
        )
    ].copy()

    retained = valid[
        valid[
            "reliability_score"
        ]
        >= threshold
    ].copy()

    total = len(
        valid
    )

    retained_count = len(
        retained
    )

    coverage = (
        retained_count
        / total
        if total > 0
        else np.nan
    )

    abstention_rate = (
        1.0
        - coverage
        if np.isfinite(
            coverage
        )
        else np.nan
    )

    mean_reliability = (
        retained[
            "reliability_score"
        ].mean()
        if retained_count > 0
        else np.nan
    )

    median_reliability = (
        retained[
            "reliability_score"
        ].median()
        if retained_count > 0
        else np.nan
    )

    mean_next_alpha_change = (
        retained[
            "next_alpha_change"
        ].mean()
        if retained_count > 0
        else np.nan
    )

    median_next_alpha_change = (
        retained[
            "next_alpha_change"
        ].median()
        if retained_count > 0
        else np.nan
    )

    mean_next_alpha_deterioration = (
        retained[
            "next_alpha_deterioration"
        ].mean()
        if retained_count > 0
        else np.nan
    )

    severe_loss_rate = np.nan

    if retained_count > 0:
        valid_severe = retained[
            "future_severe_loss"
        ].dropna()

        if len(valid_severe) > 0:
            severe_loss_rate = (
                valid_severe.mean()
            )

    extreme_loss_rate = np.nan

    if retained_count > 0:
        valid_extreme = retained[
            "future_extreme_loss"
        ].dropna()

        if len(valid_extreme) > 0:
            extreme_loss_rate = (
                valid_extreme.mean()
            )

    left_reliable_fraction = (
        retained[
            "left_reliable"
        ].mean()
        if retained_count > 0
        else np.nan
    )

    right_reliable_fraction = (
        retained[
            "right_reliable"
        ].mean()
        if retained_count > 0
        else np.nan
    )

    return {
        "reliability_threshold": threshold,
        "total_valid_windows": total,
        "retained_windows": retained_count,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "mean_reliability": mean_reliability,
        "median_reliability": median_reliability,
        "mean_next_alpha_change": mean_next_alpha_change,
        "median_next_alpha_change": median_next_alpha_change,
        "mean_next_alpha_deterioration": mean_next_alpha_deterioration,
        "future_severe_loss_rate": severe_loss_rate,
        "future_extreme_loss_rate": extreme_loss_rate,
        "left_reliable_fraction": left_reliable_fraction,
        "right_reliable_fraction": right_reliable_fraction,
    }


def main():
    data = load_dynamic_data()

    returns = load_returns()

    data = build_features(
        data,
        returns,
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid_columns = [
        "reliability_score",
        "left_alpha",
        "right_alpha",
        "next_alpha_change",
        "next_alpha_deterioration",
    ]

    data = data.dropna(
        subset=valid_columns
    ).reset_index(
        drop=True
    )

    rows = []

    for threshold in RELIABILITY_THRESHOLDS:
        rows.append(
            evaluate_threshold(
                data,
                threshold,
            )
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    base = result[
        result[
            "reliability_threshold"
        ]
        == 0.00
    ].iloc[0]

    result[
        "relative_alpha_change_reduction"
    ] = (
        1.0
        - (
            result[
                "mean_next_alpha_change"
            ]
            / base[
                "mean_next_alpha_change"
            ]
        )
    )

    result[
        "relative_severe_loss_rate"
    ] = (
        result[
            "future_severe_loss_rate"
        ]
        / base[
            "future_severe_loss_rate"
        ]
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    efficient_rows = []

    for _, row in result.iterrows():
        if (
            row[
                "retained_windows"
            ]
            >= 20
        ):
            efficient_rows.append(
                row
            )

    efficient = pd.DataFrame(
        efficient_rows
    )

    if len(efficient) > 0:
        best_alpha = efficient.loc[
            efficient[
                "mean_next_alpha_change"
            ].idxmin()
        ]

        best_severe = efficient.loc[
            efficient[
                "future_severe_loss_rate"
            ].idxmin()
        ]
    else:
        best_alpha = None
        best_severe = None

    summary_rows = [
        {
            "metric": "valid_windows",
            "value": len(data),
        },
        {
            "metric": "mean_reliability",
            "value": data[
                "reliability_score"
            ].mean(),
        },
        {
            "metric": "median_reliability",
            "value": data[
                "reliability_score"
            ].median(),
        },
        {
            "metric": "mean_next_alpha_change",
            "value": data[
                "next_alpha_change"
            ].mean(),
        },
        {
            "metric": "future_severe_loss_rate",
            "value": data[
                "future_severe_loss"
            ].mean(),
        },
        {
            "metric": "future_extreme_loss_rate",
            "value": data[
                "future_extreme_loss"
            ].mean(),
        },
    ]

    if best_alpha is not None:
        summary_rows.extend(
            [
                {
                    "metric": "best_threshold_for_alpha_stability",
                    "value": best_alpha[
                        "reliability_threshold"
                    ],
                },
                {
                    "metric": "best_threshold_alpha_stability_coverage",
                    "value": best_alpha[
                        "coverage"
                    ],
                },
                {
                    "metric": "best_threshold_alpha_stability_mean_change",
                    "value": best_alpha[
                        "mean_next_alpha_change"
                    ],
                },
            ]
        )

    if best_severe is not None:
        summary_rows.extend(
            [
                {
                    "metric": "best_threshold_for_severe_loss_rate",
                    "value": best_severe[
                        "reliability_threshold"
                    ],
                },
                {
                    "metric": "best_threshold_severe_loss_coverage",
                    "value": best_severe[
                        "coverage"
                    ],
                },
                {
                    "metric": "best_threshold_severe_loss_rate",
                    "value": best_severe[
                        "future_severe_loss_rate"
                    ],
                },
            ]
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        f"Valid windows: {len(data)}"
    )

    print()

    print(
        "Reliability-abstention frontier:"
    )

    print(
        result.to_string(
            index=False
        )
    )

    print()

    print(
        "Best threshold for alpha stability:"
    )

    if best_alpha is not None:
        print(
            f"threshold={best_alpha['reliability_threshold']:.2f}, "
            f"coverage={best_alpha['coverage']:.4f}, "
            f"abstention={best_alpha['abstention_rate']:.4f}, "
            f"mean_next_alpha_change={best_alpha['mean_next_alpha_change']:.6f}"
        )
    else:
        print(
            "No threshold retained at least 20 windows."
        )

    print()

    print(
        "Best threshold for severe-loss rate:"
    )

    if best_severe is not None:
        print(
            f"threshold={best_severe['reliability_threshold']:.2f}, "
            f"coverage={best_severe['coverage']:.4f}, "
            f"abstention={best_severe['abstention_rate']:.4f}, "
            f"future_severe_loss_rate={best_severe['future_severe_loss_rate']:.6f}"
        )
    else:
        print(
            "No threshold retained at least 20 windows."
        )

    print()

    print(
        f"Detailed results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
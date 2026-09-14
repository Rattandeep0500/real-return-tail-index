from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "tables"
    / "tail_asymmetry.csv"
)

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "tail_stress_index.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "tail_stress_index_summary.csv"
)

OVERALL_FILE = (
    ROOT
    / "tables"
    / "tail_stress_index_overall.csv"
)

MIN_HISTORY = 24

SEVERITY_WEIGHT = 0.50
ASYMMETRY_WEIGHT = 0.20
VOLATILITY_WEIGHT = 0.30

LOW_QUANTILE = 0.25
MODERATE_QUANTILE = 0.50
HIGH_QUANTILE = 0.75

REQUIRED_COLUMNS = [
    "date",
    "left_alpha",
    "right_alpha",
    "left_joint_reliability",
    "right_joint_reliability",
    "tail_asymmetry_magnitude",
    "real_volatility",
]


def load_data():
    data = pd.read_csv(
        INPUT_FILE
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = data[
        REQUIRED_COLUMNS
    ].copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = data.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    return data


def expanding_percentile(
    series,
    index,
    min_history=MIN_HISTORY,
):
    history = series.iloc[
        :index
    ].dropna()

    current = series.iloc[
        index
    ]

    if len(history) < min_history:
        return np.nan

    if not np.isfinite(
        current
    ):
        return np.nan

    return float(
        np.clip(
            np.mean(
                history
                <= current
            ),
            0.0,
            1.0,
        )
    )


def expanding_quantile(
    series,
    index,
    quantile,
    min_history=MIN_HISTORY,
):
    history = series.iloc[
        :index
    ].dropna()

    if len(history) < min_history:
        return np.nan

    return float(
        history.quantile(
            quantile
        )
    )


def build_component_scores(
    data,
):
    data = data.copy()

    data[
        "minimum_tail_alpha"
    ] = np.minimum(
        data[
            "left_alpha"
        ],
        data[
            "right_alpha"
        ],
    )

    data[
        "tail_reliability"
    ] = np.clip(
        np.minimum(
            data[
                "left_joint_reliability"
            ],
            data[
                "right_joint_reliability"
            ],
        ),
        0.0,
        1.0,
    )

    data[
        "alpha_severity_score"
    ] = np.nan

    data[
        "asymmetry_score"
    ] = np.nan

    data[
        "volatility_score"
    ] = np.nan

    for i in range(
        len(data)
    ):
        data.at[
            i,
            "alpha_severity_score",
        ] = (
            1.0
            - expanding_percentile(
                data[
                    "minimum_tail_alpha"
                ],
                i,
            )
        )

        data.at[
            i,
            "asymmetry_score",
        ] = expanding_percentile(
            data[
                "tail_asymmetry_magnitude"
            ],
            i,
        )

        data.at[
            i,
            "volatility_score",
        ] = expanding_percentile(
            data[
                "real_volatility"
            ],
            i,
        )

    weight_sum = (
        SEVERITY_WEIGHT
        + ASYMMETRY_WEIGHT
        + VOLATILITY_WEIGHT
    )

    if weight_sum <= 0:
        raise ValueError(
            "Stress weights must sum to a positive value."
        )

    severity_weight = (
        SEVERITY_WEIGHT
        / weight_sum
    )

    asymmetry_weight = (
        ASYMMETRY_WEIGHT
        / weight_sum
    )

    volatility_weight = (
        VOLATILITY_WEIGHT
        / weight_sum
    )

    data[
        "raw_tail_stress"
    ] = (
        severity_weight
        * data[
            "alpha_severity_score"
        ]
        + asymmetry_weight
        * data[
            "asymmetry_score"
        ]
        + volatility_weight
        * data[
            "volatility_score"
        ]
    )

    data[
        "reliability_weighted_tail_stress"
    ] = (
        data[
            "raw_tail_stress"
        ]
        * data[
            "tail_reliability"
        ]
    )

    data[
        "stress_reliability_penalty"
    ] = (
        1.0
        - data[
            "tail_reliability"
        ]
    )

    return data


def classify_dynamic_stress(
    data,
):
    data = data.copy()

    stress = data[
        "reliability_weighted_tail_stress"
    ]

    data[
        "stress_q25"
    ] = np.nan

    data[
        "stress_q50"
    ] = np.nan

    data[
        "stress_q75"
    ] = np.nan

    data[
        "stress_state"
    ] = "WARMUP"

    for i in range(
        len(data)
    ):
        q25 = expanding_quantile(
            stress,
            i,
            LOW_QUANTILE,
        )

        q50 = expanding_quantile(
            stress,
            i,
            MODERATE_QUANTILE,
        )

        q75 = expanding_quantile(
            stress,
            i,
            HIGH_QUANTILE,
        )

        data.at[
            i,
            "stress_q25",
        ] = q25

        data.at[
            i,
            "stress_q50",
        ] = q50

        data.at[
            i,
            "stress_q75",
        ] = q75

        current = stress.iloc[
            i
        ]

        if (
            not np.isfinite(
                current
            )
            or not np.isfinite(
                q25
            )
            or not np.isfinite(
                q50
            )
            or not np.isfinite(
                q75
            )
        ):
            data.at[
                i,
                "stress_state",
            ] = "WARMUP"

        elif current <= q25:
            data.at[
                i,
                "stress_state",
            ] = "LOW_STRESS"

        elif current <= q50:
            data.at[
                i,
                "stress_state",
            ] = "MODERATE_STRESS"

        elif current <= q75:
            data.at[
                i,
                "stress_state",
            ] = "HIGH_STRESS"

        else:
            data.at[
                i,
                "stress_state",
            ] = "EXTREME_STRESS"

    return data


def build_summary(
    data,
):
    states = [
        "LOW_STRESS",
        "MODERATE_STRESS",
        "HIGH_STRESS",
        "EXTREME_STRESS",
    ]

    valid = data[
        data[
            "stress_state"
        ].isin(
            states
        )
    ].copy()

    rows = []

    for state in states:
        group = valid[
            valid[
                "stress_state"
            ]
            == state
        ]

        rows.append(
            {
                "stress_state": state,
                "observations": len(group),
                "frequency": (
                    len(group)
                    / len(valid)
                    if len(valid) > 0
                    else np.nan
                ),
                "mean_stress": group[
                    "reliability_weighted_tail_stress"
                ].mean(),
                "median_stress": group[
                    "reliability_weighted_tail_stress"
                ].median(),
                "mean_raw_stress": group[
                    "raw_tail_stress"
                ].mean(),
                "mean_reliability": group[
                    "tail_reliability"
                ].mean(),
                "mean_minimum_tail_alpha": group[
                    "minimum_tail_alpha"
                ].mean(),
                "mean_asymmetry": group[
                    "tail_asymmetry_magnitude"
                ].mean(),
                "mean_volatility": group[
                    "real_volatility"
                ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_overall_summary(
    data,
):
    states = [
        "LOW_STRESS",
        "MODERATE_STRESS",
        "HIGH_STRESS",
        "EXTREME_STRESS",
    ]

    valid = data[
        data[
            "stress_state"
        ].isin(
            states
        )
    ].copy()

    if len(valid) == 0:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "observations": len(valid),
                "mean_stress": valid[
                    "reliability_weighted_tail_stress"
                ].mean(),
                "median_stress": valid[
                    "reliability_weighted_tail_stress"
                ].median(),
                "mean_raw_stress": valid[
                    "raw_tail_stress"
                ].mean(),
                "mean_reliability": valid[
                    "tail_reliability"
                ].mean(),
                "mean_alpha_severity": valid[
                    "alpha_severity_score"
                ].mean(),
                "mean_asymmetry_score": valid[
                    "asymmetry_score"
                ].mean(),
                "mean_volatility_score": valid[
                    "volatility_score"
                ].mean(),
                "low_stress_fraction": (
                    (
                        valid[
                            "stress_state"
                        ]
                        == "LOW_STRESS"
                    ).mean()
                ),
                "moderate_stress_fraction": (
                    (
                        valid[
                            "stress_state"
                        ]
                        == "MODERATE_STRESS"
                    ).mean()
                ),
                "high_stress_fraction": (
                    (
                        valid[
                            "stress_state"
                        ]
                        == "HIGH_STRESS"
                    ).mean()
                ),
                "extreme_stress_fraction": (
                    (
                        valid[
                            "stress_state"
                        ]
                        == "EXTREME_STRESS"
                    ).mean()
                ),
            }
        ]
    )


def main():
    data = load_data()

    data = build_component_scores(
        data
    )

    data = classify_dynamic_stress(
        data
    )

    summary = build_summary(
        data
    )

    overall = build_overall_summary(
        data
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    overall.to_csv(
        OVERALL_FILE,
        index=False,
    )

    valid = data[
        data[
            "stress_state"
        ].isin(
            [
                "LOW_STRESS",
                "MODERATE_STRESS",
                "HIGH_STRESS",
                "EXTREME_STRESS",
            ]
        )
    ]

    print(
        f"Observations: {len(data)}"
    )

    print(
        f"Valid stress observations: {len(valid)}"
    )

    print()

    print(
        "Overall stress summary:"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print()

    print(
        "Stress-state summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Stress-state frequency sum: "
        f"{summary['frequency'].sum():.6f}"
    )

    print()

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )

    print(
        f"Overall summary saved to: {OVERALL_FILE}"
    )


if __name__ == "__main__":
    main()
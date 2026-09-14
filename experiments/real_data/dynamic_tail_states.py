from __future__ import annotations

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
    / "dynamic_tail_states.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "dynamic_tail_states_summary.csv"
)


MIN_HISTORY = 24

LOW_RELIABILITY_Q = 0.25
EXTREME_Q = 0.10
ASYMMETRY_Q = 0.75
DETERIORATION_Q = 0.25


REQUIRED_COLUMNS = [
    "date",
    "left_alpha",
    "left_k",
    "left_joint_reliability",
    "right_alpha",
    "right_k",
    "right_joint_reliability",
    "real_volatility",
    "left_alpha_change",
    "right_alpha_change",
    "left_tail_thickness",
    "right_tail_thickness",
]


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


def expanding_median(
    series,
    index,
    min_history=MIN_HISTORY,
):
    history = series.iloc[
        :index
    ].dropna()

    if len(history) < min_history:
        return np.nan

    return float(
        history.median()
    )


def classify_state(
    row,
):
    overall_reliability = row[
        "overall_joint_reliability"
    ]

    if (
        not np.isfinite(
            overall_reliability
        )
    ):
        return "LOW_RELIABILITY"

    if row[
        "reliability_threshold"
    ] != row[
        "reliability_threshold"
    ]:
        return "LOW_RELIABILITY"

    if (
        overall_reliability
        <= row[
            "reliability_threshold"
        ]
    ):
        return "LOW_RELIABILITY"

    left_extreme = (
        np.isfinite(
            row[
                "left_alpha"
            ]
        )
        and np.isfinite(
            row[
                "left_extreme_threshold"
            ]
        )
        and row[
            "left_alpha"
        ]
        <= row[
            "left_extreme_threshold"
        ]
        and row[
            "left_joint_reliability"
        ]
        > row[
            "reliability_threshold"
        ]
    )

    right_extreme = (
        np.isfinite(
            row[
                "right_alpha"
            ]
        )
        and np.isfinite(
            row[
                "right_extreme_threshold"
            ]
        )
        and row[
            "right_alpha"
        ]
        <= row[
            "right_extreme_threshold"
        ]
        and row[
            "right_joint_reliability"
        ]
        > row[
            "reliability_threshold"
        ]
    )

    if (
        left_extreme
        and right_extreme
    ):
        if np.isfinite(
            row[
                "alpha_asymmetry"
            ]
        ):
            if (
                row[
                    "alpha_asymmetry"
                ]
                >= row[
                    "asymmetry_threshold"
                ]
            ):
                return "ASYMMETRIC_TAIL"

        return "TAIL_DETERIORATING"

    if left_extreme:
        return "EXTREME_LEFT_TAIL"

    if right_extreme:
        return "EXTREME_RIGHT_TAIL"

    left_deteriorating = (
        np.isfinite(
            row[
                "left_alpha_change"
            ]
        )
        and np.isfinite(
            row[
                "left_change_threshold"
            ]
        )
        and row[
            "left_alpha_change"
        ]
        <= row[
            "left_change_threshold"
        ]
    )

    right_deteriorating = (
        np.isfinite(
            row[
                "right_alpha_change"
            ]
        )
        and np.isfinite(
            row[
                "right_change_threshold"
            ]
        )
        and row[
            "right_alpha_change"
        ]
        <= row[
            "right_change_threshold"
        ]
    )

    left_thickening = (
        np.isfinite(
            row[
                "left_tail_thickness"
            ]
        )
        and np.isfinite(
            row[
                "left_thickness_threshold"
            ]
        )
        and row[
            "left_tail_thickness"
        ]
        >= row[
            "left_thickness_threshold"
        ]
    )

    right_thickening = (
        np.isfinite(
            row[
                "right_tail_thickness"
            ]
        )
        and np.isfinite(
            row[
                "right_thickness_threshold"
            ]
        )
        and row[
            "right_tail_thickness"
        ]
        >= row[
            "right_thickness_threshold"
        ]
    )

    left_signal = (
        left_deteriorating
        or left_thickening
    )

    right_signal = (
        right_deteriorating
        or right_thickening
    )

    if (
        left_signal
        and right_signal
    ):
        return "TAIL_DETERIORATING"

    if (
        left_signal
        and not right_signal
    ):
        return "EXTREME_LEFT_TAIL"

    if (
        right_signal
        and not left_signal
    ):
        return "EXTREME_RIGHT_TAIL"

    if np.isfinite(
        row[
            "alpha_asymmetry"
        ]
    ) and np.isfinite(
        row[
            "asymmetry_threshold"
        ]
    ):
        if (
            row[
                "alpha_asymmetry"
            ]
            >= row[
                "asymmetry_threshold"
            ]
        ):
            return "ASYMMETRIC_TAIL"

    return "STABLE_TAIL"


def main():
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

    data[
        "overall_joint_reliability"
    ] = np.minimum(
        data[
            "left_joint_reliability"
        ],
        data[
            "right_joint_reliability"
        ],
    )

    data[
        "alpha_asymmetry"
    ] = np.abs(
        data[
            "left_alpha"
        ]
        - data[
            "right_alpha"
        ]
    )

    data[
        "joint_tail_thickness"
    ] = np.maximum(
        data[
            "left_tail_thickness"
        ],
        data[
            "right_tail_thickness"
        ],
    )

    data[
        "reliability_threshold"
    ] = np.nan

    data[
        "left_extreme_threshold"
    ] = np.nan

    data[
        "right_extreme_threshold"
    ] = np.nan

    data[
        "left_change_threshold"
    ] = np.nan

    data[
        "right_change_threshold"
    ] = np.nan

    data[
        "left_thickness_threshold"
    ] = np.nan

    data[
        "right_thickness_threshold"
    ] = np.nan

    data[
        "asymmetry_threshold"
    ] = np.nan

    states = []

    for i in range(
        len(data)
    ):
        data.at[
            i,
            "reliability_threshold",
        ] = expanding_quantile(
            data[
                "overall_joint_reliability"
            ],
            i,
            LOW_RELIABILITY_Q,
        )

        data.at[
            i,
            "left_extreme_threshold",
        ] = expanding_quantile(
            data[
                "left_alpha"
            ],
            i,
            EXTREME_Q,
        )

        data.at[
            i,
            "right_extreme_threshold",
        ] = expanding_quantile(
            data[
                "right_alpha"
            ],
            i,
            EXTREME_Q,
        )

        data.at[
            i,
            "left_change_threshold",
        ] = expanding_quantile(
            data[
                "left_alpha_change"
            ],
            i,
            DETERIORATION_Q,
        )

        data.at[
            i,
            "right_change_threshold",
        ] = expanding_quantile(
            data[
                "right_alpha_change"
            ],
            i,
            DETERIORATION_Q,
        )

        data.at[
            i,
            "left_thickness_threshold",
        ] = expanding_quantile(
            data[
                "left_tail_thickness"
            ],
            i,
            ASYMMETRY_Q,
        )

        data.at[
            i,
            "right_thickness_threshold",
        ] = expanding_quantile(
            data[
                "right_tail_thickness"
            ],
            i,
            ASYMMETRY_Q,
        )

        data.at[
            i,
            "asymmetry_threshold",
        ] = expanding_quantile(
            data[
                "alpha_asymmetry"
            ],
            i,
            ASYMMETRY_Q,
        )

        state = classify_state(
            data.iloc[i]
        )

        if i < MIN_HISTORY:
            state = "WARMUP"

        states.append(
            state
        )

    data[
        "tail_state"
    ] = states

    data[
        "state_changed"
    ] = (
        data[
            "tail_state"
        ]
        != data[
            "tail_state"
        ].shift(1)
    )

    data.loc[
        0,
        "state_changed"
    ] = False

    data[
        "state_duration"
    ] = (
        data[
            "tail_state"
        ]
        .groupby(
            (
                data[
                    "tail_state"
                ]
                != data[
                    "tail_state"
                ].shift(1)
            ).cumsum()
        )
        .cumcount()
        + 1
    )

    data[
        "state_reliability"
    ] = data[
        "overall_joint_reliability"
    ]

    data[
        "state_tail_thickness"
    ] = data[
        "joint_tail_thickness"
    ]

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = (
        data.groupby(
            "tail_state",
            dropna=False,
        )
        .agg(
            observations=(
                "tail_state",
                "size",
            ),
            mean_reliability=(
                "state_reliability",
                "mean",
            ),
            median_reliability=(
                "state_reliability",
                "median",
            ),
            mean_left_alpha=(
                "left_alpha",
                "mean",
            ),
            mean_right_alpha=(
                "right_alpha",
                "mean",
            ),
            mean_volatility=(
                "real_volatility",
                "mean",
            ),
            mean_left_thickness=(
                "left_tail_thickness",
                "mean",
            ),
            mean_right_thickness=(
                "right_tail_thickness",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        "frequency"
    ] = (
        summary[
            "observations"
        ]
        / len(data)
    )

    summary = summary.sort_values(
        "observations",
        ascending=False,
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        f"Observations: {len(data)}"
    )

    print(
        f"Warmup observations: "
        f"{(data['tail_state'] == 'WARMUP').sum()}"
    )

    print()
    print(
        "State counts:"
    )

    print(
        data[
            "tail_state"
        ].value_counts(
            dropna=False
        ).to_string()
    )

    print()
    print(
        "State frequencies:"
    )

    print(
        (
            data[
                "tail_state"
            ]
            .value_counts(
                normalize=True,
                dropna=False,
            )
            .mul(
                100
            )
            .round(
                2
            )
        ).to_string()
    )

    print()
    print(
        f"State transitions: "
        f"{data['state_changed'].sum()}"
    )

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
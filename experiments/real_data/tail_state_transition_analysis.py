from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

STATE_FILE = (
    ROOT
    / "tables"
    / "dynamic_tail_states.csv"
)

RETURN_FILE = (
    ROOT
    / "data"
    / "processed"
    / "sp500_real_returns.csv"
)

TRANSITION_FILE = (
    ROOT
    / "tables"
    / "tail_state_transition_matrix.csv"
)

TRANSITION_COUNTS_FILE = (
    ROOT
    / "tables"
    / "tail_state_transition_counts.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "tail_state_transition_summary.csv"
)

EVENT_FILE = (
    ROOT
    / "tables"
    / "tail_state_future_risk.csv"
)

VALID_STATES = [
    "LOW_RELIABILITY",
    "STABLE_TAIL",
    "TAIL_DETERIORATING",
    "EXTREME_LEFT_TAIL",
    "EXTREME_RIGHT_TAIL",
    "ASYMMETRIC_TAIL",
]


def load_data():
    states = pd.read_csv(
        STATE_FILE
    )

    returns = pd.read_csv(
        RETURN_FILE,
        index_col=0,
    )

    returns = returns.reset_index()

    returns = returns.rename(
        columns={
            returns.columns[0]: "date"
        }
    )

    required_return_columns = [
        "date",
        "real_return",
        "real_log_return",
    ]

    missing_returns = [
        column
        for column in required_return_columns
        if column not in returns.columns
    ]

    if missing_returns:
        raise ValueError(
            f"Missing return columns: {missing_returns}"
        )

    states["date"] = pd.to_datetime(
        states["date"]
    )

    returns["date"] = pd.to_datetime(
        returns["date"]
    )

    data = states.merge(
        returns[
            required_return_columns
        ],
        on="date",
        how="left",
    )

    data = data.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    return data


def build_transition_matrix(
    data,
):
    state_data = data[
        data[
            "tail_state"
        ].isin(
            VALID_STATES
        )
    ].copy()

    state_data[
        "next_state"
    ] = state_data[
        "tail_state"
    ].shift(
        -1
    )

    state_data = state_data[
        state_data[
            "next_state"
        ].isin(
            VALID_STATES
        )
    ]

    transition_counts = pd.crosstab(
        state_data[
            "tail_state"
        ],
        state_data[
            "next_state"
        ],
    )

    transition_counts = (
        transition_counts.reindex(
            index=VALID_STATES,
            columns=VALID_STATES,
            fill_value=0,
        )
    )

    row_totals = (
        transition_counts.sum(
            axis=1
        )
    )

    transition_probabilities = (
        transition_counts.div(
            row_totals.replace(
                0,
                np.nan,
            ),
            axis=0,
        )
    )

    transition_counts.index.name = (
        "current_state"
    )

    transition_probabilities.index.name = (
        "current_state"
    )

    return (
        transition_counts,
        transition_probabilities,
    )


def build_persistence_table(
    data,
):
    state_data = data[
        data[
            "tail_state"
        ].isin(
            VALID_STATES
        )
    ].copy()

    state_data[
        "next_state"
    ] = state_data[
        "tail_state"
    ].shift(
        -1
    )

    valid_transition = (
        state_data[
            "next_state"
        ].isin(
            VALID_STATES
        )
    )

    state_data[
        "is_persistent"
    ] = (
        state_data[
            "tail_state"
        ]
        == state_data[
            "next_state"
        ]
    )

    rows = []

    for state in VALID_STATES:
        group = state_data[
            state_data[
                "tail_state"
            ]
            == state
        ]

        group = group[
            valid_transition.loc[
                group.index
            ]
        ]

        transition_count = len(
            group
        )

        if transition_count > 0:
            persistence = group[
                "is_persistent"
            ].mean()
        else:
            persistence = np.nan

        rows.append(
            {
                "tail_state": state,
                "transition_observations": transition_count,
                "one_step_persistence": persistence,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_state_summary(
    data,
):
    state_data = data[
        data[
            "tail_state"
        ].isin(
            VALID_STATES
        )
    ].copy()

    persistence = build_persistence_table(
        data
    )

    rows = []

    for state in VALID_STATES:
        group = state_data[
            state_data[
                "tail_state"
            ]
            == state
        ].copy()

        if len(group) == 0:
            continue

        persistence_row = persistence[
            persistence[
                "tail_state"
            ]
            == state
        ]

        if len(
            persistence_row
        ) > 0:
            persistence_value = float(
                persistence_row[
                    "one_step_persistence"
                ].iloc[0]
            )
        else:
            persistence_value = np.nan

        rows.append(
            {
                "tail_state": state,
                "observations": len(group),
                "frequency": (
                    len(group)
                    / len(state_data)
                ),
                "mean_current_real_return": group[
                    "real_return"
                ].mean(),
                "median_current_real_return": group[
                    "real_return"
                ].median(),
                "mean_real_volatility": group[
                    "real_volatility"
                ].mean(),
                "mean_reliability": group[
                    "state_reliability"
                ].mean(),
                "median_reliability": group[
                    "state_reliability"
                ].median(),
                "mean_left_alpha": group[
                    "left_alpha"
                ].mean(),
                "mean_right_alpha": group[
                    "right_alpha"
                ].mean(),
                "one_step_persistence": persistence_value,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_forward_returns(
    data,
):
    data = data.copy()

    r = data[
        "real_return"
    ]

    data[
        "future_1m"
    ] = (
        1.0
        + r.shift(
            -1
        )
        - 1.0
    )

    data[
        "future_3m"
    ] = (
        (
            1.0
            + r.shift(
                -1
            )
        )
        * (
            1.0
            + r.shift(
                -2
            )
        )
        * (
            1.0
            + r.shift(
                -3
            )
        )
        - 1.0
    )

    data[
        "future_6m"
    ] = (
        (
            1.0
            + r.shift(
                -1
            )
        )
        * (
            1.0
            + r.shift(
                -2
            )
        )
        * (
            1.0
            + r.shift(
                -3
            )
        )
        * (
            1.0
            + r.shift(
                -4
            )
        )
        * (
            1.0
            + r.shift(
                -5
            )
        )
        * (
            1.0
            + r.shift(
                -6
            )
        )
        - 1.0
    )

    data[
        "future_12m"
    ] = (
        (
            1.0
            + r.shift(
                -1
            )
        )
        * (
            1.0
            + r.shift(
                -2
            )
        )
        * (
            1.0
            + r.shift(
                -3
            )
        )
        * (
            1.0
            + r.shift(
                -4
            )
        )
        * (
            1.0
            + r.shift(
                -5
            )
        )
        * (
            1.0
            + r.shift(
                -6
            )
        )
        * (
            1.0
            + r.shift(
                -7
            )
        )
        * (
            1.0
            + r.shift(
                -8
            )
        )
        * (
            1.0
            + r.shift(
                -9
            )
        )
        * (
            1.0
            + r.shift(
                -10
            )
        )
        * (
            1.0
            + r.shift(
                -11
            )
        )
        * (
            1.0
            + r.shift(
                -12
            )
        )
        - 1.0
    )

    return data


def build_future_risk_table(
    data,
):
    data = build_forward_returns(
        data
    )

    valid_data = data[
        data[
            "tail_state"
        ].isin(
            VALID_STATES
        )
    ].copy()

    historical_returns = data[
        "real_return"
    ].dropna()

    q05 = historical_returns.quantile(
        0.05
    )

    q10 = historical_returns.quantile(
        0.10
    )

    rows = []

    for state in VALID_STATES:
        group = valid_data[
            valid_data[
                "tail_state"
            ]
            == state
        ].copy()

        if len(group) == 0:
            continue

        future_1m = group[
            "future_1m"
        ].dropna()

        future_3m = group[
            "future_3m"
        ].dropna()

        future_6m = group[
            "future_6m"
        ].dropna()

        future_12m = group[
            "future_12m"
        ].dropna()

        rows.append(
            {
                "tail_state": state,
                "observations": len(group),
                "future_1m_observations": len(
                    future_1m
                ),
                "future_1m_mean": (
                    future_1m.mean()
                    if len(
                        future_1m
                    ) > 0
                    else np.nan
                ),
                "future_1m_median": (
                    future_1m.median()
                    if len(
                        future_1m
                    ) > 0
                    else np.nan
                ),
                "future_1m_p05_rate": (
                    np.mean(
                        future_1m
                        <= q05
                    )
                    if len(
                        future_1m
                    ) > 0
                    else np.nan
                ),
                "future_1m_p10_rate": (
                    np.mean(
                        future_1m
                        <= q10
                    )
                    if len(
                        future_1m
                    ) > 0
                    else np.nan
                ),
                "future_3m_observations": len(
                    future_3m
                ),
                "future_3m_mean": (
                    future_3m.mean()
                    if len(
                        future_3m
                    ) > 0
                    else np.nan
                ),
                "future_3m_median": (
                    future_3m.median()
                    if len(
                        future_3m
                    ) > 0
                    else np.nan
                ),
                "future_6m_observations": len(
                    future_6m
                ),
                "future_6m_mean": (
                    future_6m.mean()
                    if len(
                        future_6m
                    ) > 0
                    else np.nan
                ),
                "future_6m_median": (
                    future_6m.median()
                    if len(
                        future_6m
                    ) > 0
                    else np.nan
                ),
                "future_12m_observations": len(
                    future_12m
                ),
                "future_12m_mean": (
                    future_12m.mean()
                    if len(
                        future_12m
                    ) > 0
                    else np.nan
                ),
                "future_12m_median": (
                    future_12m.median()
                    if len(
                        future_12m
                    ) > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    data = load_data()

    (
        transition_counts,
        transition_probabilities,
    ) = build_transition_matrix(
        data
    )

    transition_counts.to_csv(
        TRANSITION_COUNTS_FILE
    )

    transition_probabilities.to_csv(
        TRANSITION_FILE
    )

    state_summary = build_state_summary(
        data
    )

    state_summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    future_risk = build_future_risk_table(
        data
    )

    future_risk.to_csv(
        EVENT_FILE,
        index=False,
    )

    print(
        f"Observations: {len(data)}"
    )

    print()

    print(
        "Transition counts:"
    )

    print(
        transition_counts.to_string()
    )

    print()

    print(
        "Transition probability matrix:"
    )

    print(
        transition_probabilities.round(
            3
        ).to_string()
    )

    print()

    print(
        "State summary:"
    )

    print(
        state_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Future risk by state:"
    )

    print(
        future_risk.to_string(
            index=False
        )
    )

    print()

    print(
        f"Transition counts saved to: "
        f"{TRANSITION_COUNTS_FILE}"
    )

    print(
        f"Transition matrix saved to: "
        f"{TRANSITION_FILE}"
    )

    print(
        f"State summary saved to: "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Future risk saved to: "
        f"{EVENT_FILE}"
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


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

PREDICTION_FILE = (
    ROOT
    / "tables"
    / "tail_state_predictive_predictions.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "tail_state_predictive_summary.csv"
)

MIN_TRAIN = 60

RANDOM_STATE = 42

VALID_STATES = [
    "LOW_RELIABILITY",
    "STABLE_TAIL",
    "TAIL_DETERIORATING",
    "EXTREME_LEFT_TAIL",
    "EXTREME_RIGHT_TAIL",
    "ASYMMETRIC_TAIL",
]

STATE_FEATURES = [
    "overall_joint_reliability",
    "left_alpha",
    "right_alpha",
    "left_alpha_change",
    "right_alpha_change",
    "left_tail_thickness",
    "right_tail_thickness",
    "real_volatility",
    "alpha_asymmetry",
]

RETURN_COLS = [
    "date",
    "real_return",
    "real_log_return",
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

    states["date"] = pd.to_datetime(
        states["date"]
    )

    returns["date"] = pd.to_datetime(
        returns["date"]
    )

    returns = returns[
        RETURN_COLS
    ].copy()

    data = states.merge(
        returns,
        on="date",
        how="left",
    )

    data = data.sort_values(
        "date"
    ).reset_index(
        drop=True
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

    return data


def build_future_targets(
    data,
):
    data = data.copy()

    future_1m = data[
        "real_return"
    ].shift(
        -1
    )

    future_3m = (
        (
            1.0
            + data[
                "real_return"
            ].shift(
                -1
            )
        )
        * (
            1.0
            + data[
                "real_return"
            ].shift(
                -2
            )
        )
        * (
            1.0
            + data[
                "real_return"
            ].shift(
                -3
            )
        )
        - 1.0
    )

    future_6m = (
        (
            1.0
            + data[
                "real_return"
            ].shift(
                -1
            )
        )
        * (
            1.0
            + data[
                "real_return"
            ].shift(
                -2
            )
        )
        * (
            1.0
            + data[
                "real_return"
            ].shift(
                -3
            )
        )
        * (
            1.0
            + data[
                "real_return"
            ].shift(
                -4
            )
        )
        * (
            1.0
            + data[
                "real_return"
            ].shift(
                -5
            )
        )
        * (
            1.0
            + data[
                "real_return"
            ].shift(
                -6
            )
        )
        - 1.0
    )

    data[
        "future_1m"
    ] = future_1m

    data[
        "future_3m"
    ] = future_3m

    data[
        "future_6m"
    ] = future_6m

    historical_returns = data[
        "real_return"
    ].dropna()

    q05 = historical_returns.quantile(
        0.05
    )

    q10 = historical_returns.quantile(
        0.10
    )

    data[
        "target_1m_extreme"
    ] = np.where(
        data[
            "future_1m"
        ].notna(),
        (
            data[
                "future_1m"
            ]
            <= q05
        ).astype(int),
        np.nan,
    )

    data[
        "target_1m_severe"
    ] = np.where(
        data[
            "future_1m"
        ].notna(),
        (
            data[
                "future_1m"
            ]
            <= q10
        ).astype(int),
        np.nan,
    )

    data[
        "target_3m_loss"
    ] = np.where(
        data[
            "future_3m"
        ].notna(),
        (
            data[
                "future_3m"
            ]
            < 0.0
        ).astype(int),
        np.nan,
    )

    data[
        "target_6m_loss"
    ] = np.where(
        data[
            "future_6m"
        ].notna(),
        (
            data[
                "future_6m"
            ]
            < 0.0
        ).astype(int),
        np.nan,
    )

    return data, q05, q10


def encode_features(
    data,
):
    features = data[
        STATE_FEATURES
    ].copy()

    state_dummies = pd.get_dummies(
        data[
            "tail_state"
        ],
        prefix="state",
        dtype=float,
    )

    features = pd.concat(
        [
            features,
            state_dummies,
        ],
        axis=1,
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features


def fit_predict_logistic(
    x_train,
    y_train,
    x_test,
):
    train_mask = (
        np.isfinite(
            x_train
        ).all(
            axis=1
        )
        & np.isfinite(
            y_train
        )
    )

    x_train = x_train[
        train_mask
    ]

    y_train = y_train[
        train_mask
    ]

    if len(
        np.unique(y_train)
    ) < 2:
        return np.nan

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(
        x_train,
        y_train,
    )

    test_mask = np.isfinite(
        x_test
    ).all(
        axis=1
    )

    if not test_mask[0]:
        return np.nan

    probability = model.predict_proba(
        x_test
    )[
        0,
        1,
    ]

    return float(
        probability
    )


def walk_forward_predictions(
    data,
    features,
    target_column,
):
    rows = []

    target = data[
        target_column
    ].to_numpy(
        dtype=float
    )

    for i in range(
        MIN_TRAIN,
        len(data),
    ):
        if not np.isfinite(
            target[i]
        ):
            continue

        train_slice = slice(
            0,
            i,
        )

        x_train = (
            features.iloc[
                train_slice
            ].to_numpy(
                dtype=float
            )
        )

        y_train = target[
            train_slice
        ]

        x_test = (
            features.iloc[
                [i]
            ].to_numpy(
                dtype=float
            )
        )

        valid_train = (
            np.isfinite(
                x_train
            ).all(
                axis=1
            )
            & np.isfinite(
                y_train
            )
        )

        if valid_train.sum() < MIN_TRAIN:
            continue

        if len(
            np.unique(
                y_train[
                    valid_train
                ]
            )
        ) < 2:
            continue

        probability = (
            fit_predict_logistic(
                x_train[
                    valid_train
                ],
                y_train[
                    valid_train
                ],
                x_test,
            )
        )

        if not np.isfinite(
            probability
        ):
            continue

        rows.append(
            {
                "date": data[
                    "date"
                ].iloc[i],
                "actual": int(
                    target[i]
                ),
                "predicted_probability": probability,
                "tail_state": data[
                    "tail_state"
                ].iloc[i],
                "state_reliability": data[
                    "state_reliability"
                ].iloc[i],
                "real_volatility": data[
                    "real_volatility"
                ].iloc[i],
            }
        )

    return pd.DataFrame(
        rows
    )


def volatility_baseline(
    data,
    target_column,
):
    rows = []

    target = data[
        target_column
    ].to_numpy(
        dtype=float
    )

    for i in range(
        MIN_TRAIN,
        len(data),
    ):
        if not np.isfinite(
            target[i]
        ):
            continue

        train_target = target[
            :i
        ]

        train_volatility = data[
            "real_volatility"
        ].iloc[
            :i
        ].to_numpy(
            dtype=float
        )

        test_volatility = data[
            "real_volatility"
        ].iloc[
            i
        ]

        valid = (
            np.isfinite(
                train_target
            )
            & np.isfinite(
                train_volatility
            )
        )

        if valid.sum() < MIN_TRAIN:
            continue

        x_train = (
            train_volatility[
                valid
            ]
            .reshape(
                -1,
                1,
            )
        )

        y_train = (
            train_target[
                valid
            ]
        )

        if len(
            np.unique(
                y_train
            )
        ) < 2:
            continue

        if not np.isfinite(
            test_volatility
        ):
            continue

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        model.fit(
            x_train,
            y_train,
        )

        probability = model.predict_proba(
            np.array(
                [
                    [
                        test_volatility
                    ]
                ]
            )
        )[
            0,
            1,
        ]

        rows.append(
            {
                "date": data[
                    "date"
                ].iloc[i],
                "actual": int(
                    target[i]
                ),
                "predicted_probability": float(
                    probability
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def calculate_metrics(
    predictions,
):
    if len(
        predictions
    ) == 0:
        return {
            "n": 0,
            "event_rate": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "brier_score": np.nan,
        }

    y = predictions[
        "actual"
    ].to_numpy(
        dtype=int
    )

    p = predictions[
        "predicted_probability"
    ].to_numpy(
        dtype=float
    )

    metrics = {
        "n": len(
            predictions
        ),
        "event_rate": float(
            np.mean(y)
        ),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier_score": brier_score_loss(
            y,
            p,
        ),
    }

    if len(
        np.unique(y)
    ) == 2:
        metrics[
            "roc_auc"
        ] = roc_auc_score(
            y,
            p,
        )

        metrics[
            "pr_auc"
        ] = average_precision_score(
            y,
            p,
        )

    return metrics


def build_state_risk_summary(
    data,
    target_column,
):
    valid = data[
        data[
            "tail_state"
        ].isin(
            VALID_STATES
        )
        & data[
            target_column
        ].notna()
    ].copy()

    rows = []

    for state in VALID_STATES:
        group = valid[
            valid[
                "tail_state"
            ]
            == state
        ]

        if len(group) == 0:
            continue

        rows.append(
            {
                "target": target_column,
                "tail_state": state,
                "observations": len(group),
                "event_rate": group[
                    target_column
                ].mean(),
                "mean_reliability": group[
                    "state_reliability"
                ].mean(),
                "mean_volatility": group[
                    "real_volatility"
                ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    data = load_data()

    data, q05, q10 = (
        build_future_targets(
            data
        )
    )

    features = encode_features(
        data
    )

    all_predictions = []
    summary_rows = []
    state_summary_frames = []

    targets = [
        "target_1m_extreme",
        "target_1m_severe",
        "target_3m_loss",
        "target_6m_loss",
    ]

    for target in targets:
        print()
        print(
            f"Running target: {target}"
        )

        tail_predictions = (
            walk_forward_predictions(
                data,
                features,
                target,
            )
        )

        volatility_predictions = (
            volatility_baseline(
                data,
                target,
            )
        )

        tail_metrics = calculate_metrics(
            tail_predictions
        )

        volatility_metrics = (
            calculate_metrics(
                volatility_predictions
            )
        )

        tail_predictions[
            "model"
        ] = "Tail-State"

        volatility_predictions[
            "model"
        ] = "Volatility-Baseline"

        tail_predictions[
            "target"
        ] = target

        volatility_predictions[
            "target"
        ] = target

        all_predictions.append(
            tail_predictions
        )

        all_predictions.append(
            volatility_predictions
        )

        summary_rows.append(
            {
                "target": target,
                "model": "Tail-State",
                **tail_metrics,
            }
        )

        summary_rows.append(
            {
                "target": target,
                "model": "Volatility-Baseline",
                **volatility_metrics,
            }
        )

        state_summary_frames.append(
            build_state_risk_summary(
                data,
                target,
            )
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    state_summary = pd.concat(
        state_summary_frames,
        ignore_index=True,
    )

    predictions.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    state_summary_file = (
        ROOT
        / "tables"
        / "tail_state_predictive_state_risk.csv"
    )

    state_summary.to_csv(
        state_summary_file,
        index=False,
    )

    print()
    print(
        "Historical 5% threshold: "
        f"{q05:.6f}"
    )

    print(
        "Historical 10% threshold: "
        f"{q10:.6f}"
    )

    print()
    print(
        "Predictive results:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "State-level event rates:"
    )

    print(
        state_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Predictions saved to: "
        f"{PREDICTION_FILE}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_FILE}"
    )

    print(
        f"State risk saved to: "
        f"{state_summary_file}"
    )


if __name__ == "__main__":
    main()
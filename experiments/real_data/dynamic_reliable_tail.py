from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.calibration import build_features, predict_reliability
from src.tail.model_validity import compute_model_validity


DATA_PATH = ROOT / "data" / "processed" / "sp500_real_returns.csv"
MODEL_PATH = ROOT / "data" / "metadata" / "tail_reliability_model.joblib"

TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"

WINDOW_SIZE = 120
K_MIN = 10
K_MAX = 80
K_STEP = 5
BOOTSTRAPS = 60
RANDOM_SEED = 42


def load_data():
    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["Unnamed: 0"],
    )

    data = data.rename(
        columns={"Unnamed: 0": "date"}
    )

    data = data.set_index("date")
    data = data.sort_index()

    if "real_return" not in data.columns:
        raise ValueError(
            "real_return column not found."
        )

    return data.dropna(
        subset=["real_return"]
    )


def prepare_tail(sample, side):
    x = np.asarray(
        sample,
        dtype=float,
    )

    if side == "left":
        x = -x
    elif side == "right":
        x = x
    else:
        raise ValueError(
            "side must be left or right."
        )

    x = x[
        np.isfinite(x)
        & (x > 0)
    ]

    return np.sort(x)[::-1]


def estimate_tail_features(
    sample,
    side,
    rng,
):
    tail = prepare_tail(
        sample,
        side,
    )

    if len(tail) <= K_MIN + 1:
        return None

    k_max = min(
        K_MAX,
        len(tail) - 1,
    )

    k_values = np.arange(
        K_MIN,
        k_max + 1,
        K_STEP,
        dtype=int,
    )

    alpha_values = []
    bootstrap_stds = []

    for k in k_values:
        alpha_hat = hill_estimator(
            tail,
            int(k),
        )

        bootstrap_estimates = []

        for _ in range(
            BOOTSTRAPS
        ):
            bootstrap_sample = rng.choice(
                tail,
                size=len(tail),
                replace=True,
            )

            bootstrap_sample = np.sort(
                bootstrap_sample
            )[::-1]

            try:
                estimate = hill_estimator(
                    bootstrap_sample,
                    int(k),
                )

                if np.isfinite(
                    estimate
                ) and estimate > 0:
                    bootstrap_estimates.append(
                        estimate
                    )

            except (
                ValueError,
                RuntimeError,
            ):
                continue

        alpha_values.append(
            alpha_hat
        )

        if len(bootstrap_estimates) >= 2:
            bootstrap_stds.append(
                np.std(
                    bootstrap_estimates,
                    ddof=1,
                )
            )
        else:
            bootstrap_stds.append(
                np.nan
            )

    features = build_features(
        k_values,
        np.asarray(
            alpha_values,
            dtype=float,
        ),
        np.asarray(
            bootstrap_stds,
            dtype=float,
        ),
        len(tail),
    )

    features["k"] = k_values

    return (
        tail,
        features,
    )


def choose_k(features):
    valid = features.dropna(
        subset=[
            "alpha_hat",
            "bootstrap_std",
            "bootstrap_cv",
        ]
    ).copy()

    if valid.empty:
        return None

    alpha = valid[
        "alpha_hat"
    ].to_numpy(
        dtype=float
    )

    bootstrap_cv = valid[
        "bootstrap_cv"
    ].to_numpy(
        dtype=float
    )

    local_cv = valid[
        "local_cv"
    ].to_numpy(
        dtype=float
    )

    valid_local = np.isfinite(
        local_cv
    )

    if np.any(valid_local):
        local_reference = np.nanmedian(
            local_cv
        )
    else:
        local_reference = np.nan

    score = np.zeros(
        len(valid),
        dtype=float,
    )

    score += np.nan_to_num(
        bootstrap_cv,
        nan=np.nanmedian(
            bootstrap_cv
        ),
    )

    if np.isfinite(
        local_reference
    ):
        score += np.nan_to_num(
            local_cv,
            nan=local_reference,
        )

    alpha_log_change = np.abs(
        np.gradient(
            np.log(
                np.maximum(
                    alpha,
                    1e-12,
                )
            )
        )
    )

    score += alpha_log_change

    best_index = int(
        np.argmin(
            score
        )
    )

    return int(
        valid.iloc[
            best_index
        ]["k"]
    )


def evaluate_side(
    sample,
    side,
    reliability_model,
    rng,
):
    output = estimate_tail_features(
        sample,
        side,
        rng,
    )

    if output is None:
        return None

    tail, features = output

    chosen_k = choose_k(
        features
    )

    if chosen_k is None:
        return None

    chosen_rows = features[
        features["k"] == chosen_k
    ]

    if chosen_rows.empty:
        return None

    chosen_index = chosen_rows.index[0]

    alpha_hat = float(
        features.loc[
            chosen_index,
            "alpha_hat",
        ]
    )

    predicted = predict_reliability(
        reliability_model,
        features,
    )

    estimation_reliability = float(
        predicted.loc[
            chosen_index,
            "reliability_probability",
        ]
    )

    validity = compute_model_validity(
        tail,
        features["k"].to_numpy(
            dtype=int
        ),
        features[
            "alpha_hat"
        ].to_numpy(
            dtype=float
        ),
    )

    model_validity = float(
        validity.loc[
            validity["k"] == chosen_k,
            "model_validity_score",
        ].iloc[0]
    )

    joint_reliability = (
        estimation_reliability
        * model_validity
    )

    return {
        "alpha": alpha_hat,
        "k": chosen_k,
        "estimation_reliability": estimation_reliability,
        "model_validity": model_validity,
        "joint_reliability": joint_reliability,
    }


def main():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            DATA_PATH
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            MODEL_PATH
        )

    data = load_data()

    reliability_model = joblib.load(
        MODEL_PATH
    )

    results = []

    total_windows = max(
        0,
        len(data) - WINDOW_SIZE + 1,
    )

    print("=" * 70)
    print(
        "M1.3 - DYNAMIC RELIABLE TAIL ENGINE"
    )
    print("=" * 70)
    print(
        f"Observations: {len(data)}"
    )
    print(
        f"Rolling window: {WINDOW_SIZE}"
    )
    print(
        f"Total windows: {total_windows}"
    )

    for end in range(
        WINDOW_SIZE,
        len(data) + 1,
    ):
        window = data.iloc[
            end - WINDOW_SIZE:end
        ]

        date = window.index[-1]

        row = {
            "date": date,
        }

        for side in [
            "left",
            "right",
        ]:
            side_result = evaluate_side(
                window[
                    "real_return"
                ].to_numpy(),
                side,
                reliability_model,
                rng,
            )

            if side_result is None:
                row[
                    f"{side}_alpha"
                ] = np.nan

                row[
                    f"{side}_k"
                ] = np.nan

                row[
                    f"{side}_estimation_reliability"
                ] = np.nan

                row[
                    f"{side}_model_validity"
                ] = np.nan

                row[
                    f"{side}_joint_reliability"
                ] = np.nan

            else:
                row[
                    f"{side}_alpha"
                ] = side_result[
                    "alpha"
                ]

                row[
                    f"{side}_k"
                ] = side_result[
                    "k"
                ]

                row[
                    f"{side}_estimation_reliability"
                ] = side_result[
                    "estimation_reliability"
                ]

                row[
                    f"{side}_model_validity"
                ] = side_result[
                    "model_validity"
                ]

                row[
                    f"{side}_joint_reliability"
                ] = side_result[
                    "joint_reliability"
                ]

        row[
            "real_volatility"
        ] = window[
            "real_return"
        ].std()

        results.append(
            row
        )

        completed = (
            end - WINDOW_SIZE + 1
        )

        if completed % 10 == 0:
            print(
                f"Progress: "
                f"{completed}/{total_windows}",
                flush=True,
            )

    result = pd.DataFrame(
        results
    ).set_index(
        "date"
    )

    result[
        "left_alpha_change"
    ] = result[
        "left_alpha"
    ].diff()

    result[
        "right_alpha_change"
    ] = result[
        "right_alpha"
    ].diff()

    result[
        "left_tail_thickness"
    ] = 1.0 / result[
        "left_alpha"
    ]

    result[
        "right_tail_thickness"
    ] = 1.0 / result[
        "right_alpha"
    ]

    result[
        "left_reliable"
    ] = (
        result[
            "left_joint_reliability"
        ] >= 0.60
    )

    result[
        "right_reliable"
    ] = (
        result[
            "right_joint_reliability"
        ] >= 0.60
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLES_DIR
        / "dynamic_reliable_tail.csv"
    )

    result.to_csv(
        output_path
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        result.index,
        result["left_alpha"],
        label="Left-tail α",
    )

    plt.plot(
        result.index,
        result["right_alpha"],
        label="Right-tail α",
    )

    plt.xlabel("Date")
    plt.ylabel("Tail index α̂")
    plt.title(
        "Dynamic Real-Return Tail Index"
    )
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()

    alpha_path = (
        FIGURES_DIR
        / "dynamic_reliable_tail_alpha.png"
    )

    plt.savefig(
        alpha_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        result.index,
        result[
            "left_joint_reliability"
        ],
        label="Left-tail joint reliability",
    )

    plt.plot(
        result.index,
        result[
            "right_joint_reliability"
        ],
        label="Right-tail joint reliability",
    )

    plt.axhline(
        0.60,
        linestyle="--",
        linewidth=1.5,
        label="Reliability threshold",
    )

    plt.xlabel("Date")
    plt.ylabel("Joint reliability")
    plt.title(
        "Dynamic Tail Reliability"
    )
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()

    reliability_path = (
        FIGURES_DIR
        / "dynamic_tail_reliability.png"
    )

    plt.savefig(
        reliability_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        result.index,
        result["left_k"],
        label="Left-tail k",
    )

    plt.plot(
        result.index,
        result["right_k"],
        label="Right-tail k",
    )

    plt.xlabel("Date")
    plt.ylabel("Selected k")
    plt.title(
        "Dynamic Hill Threshold"
    )
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()

    k_path = (
        FIGURES_DIR
        / "dynamic_reliable_k.png"
    )

    plt.savefig(
        k_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        result.tail(10).to_string()
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    print(
        f"Alpha figure: {alpha_path}"
    )

    print(
        f"Reliability figure: {reliability_path}"
    )

    print(
        f"k figure: {k_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator


DATA_PATH = ROOT / "data" / "processed" / "sp500_real_returns.csv"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

WINDOW_SIZE = 120
K_MIN = 10
K_MAX = 80
K_STEP = 5


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

    required = [
        "nominal_return",
        "real_return",
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

    return data


def hill_for_tail(sample, side, k):
    x = np.asarray(sample, dtype=float)

    if side == "left":
        x = -x
    elif side != "right":
        raise ValueError(
            "side must be left or right"
        )

    x = x[np.isfinite(x)]
    x = x[x > 0]

    if len(x) <= k:
        return np.nan

    return hill_estimator(
        x,
        k,
    )


def select_k(sample, side):
    x = np.asarray(sample, dtype=float)

    if side == "left":
        x = -x
    else:
        x = x

    x = x[np.isfinite(x)]
    x = x[x > 0]

    k_max = min(
        K_MAX,
        len(x) - 1,
    )

    k_values = np.arange(
        K_MIN,
        k_max + 1,
        K_STEP,
    )

    if len(k_values) == 0:
        return np.nan, np.nan

    estimates = np.array(
        [
            hill_estimator(
                x,
                int(k),
            )
            for k in k_values
        ]
    )

    local_variation = np.full(
        len(estimates),
        np.nan,
    )

    window = min(
        7,
        len(estimates),
    )

    for i in range(
        window - 1,
        len(estimates),
    ):
        local_variation[i] = np.std(
            estimates[
                i - window + 1:i + 1
            ],
            ddof=1,
        )

    valid = np.isfinite(
        local_variation
    )

    if not np.any(valid):
        idx = np.argmin(
            np.abs(
                estimates
                - np.median(estimates)
            )
        )
    else:
        valid_indices = np.where(
            valid
        )[0]

        idx = valid_indices[
            np.argmin(
                local_variation[
                    valid
                ]
            )
        ]

    return (
        int(k_values[idx]),
        float(estimates[idx]),
    )


def rolling_tail_estimates(series):
    results = []

    dates = series.index

    for end in range(
        WINDOW_SIZE,
        len(series) + 1,
    ):
        window = series.iloc[
            end - WINDOW_SIZE:end
        ]

        date = dates[end - 1]

        row = {
            "date": date,
        }

        for side in [
            "left",
            "right",
        ]:
            k, alpha = select_k(
                window.values,
                side,
            )

            row[
                f"{side}_k"
            ] = k

            row[
                f"{side}_alpha"
            ] = alpha

        row["real_volatility"] = (
            window.std()
        )

        results.append(row)

    return pd.DataFrame(
        results
    ).set_index("date")


def main():
    print("=" * 70)
    print("M1.3 — DYNAMIC TAIL ENGINE")
    print("=" * 70)

    data = load_data()

    print(
        f"Observations: {len(data)}"
    )

    print(
        f"Rolling window: {WINDOW_SIZE}"
    )

    print(
        "Calculating dynamic tail indices..."
    )

    result = rolling_tail_estimates(
        data["real_return"]
    )

    result["left_alpha_change"] = (
        result["left_alpha"].diff()
    )

    result["right_alpha_change"] = (
        result["right_alpha"].diff()
    )

    result["left_tail_thickness"] = (
        1.0 / result["left_alpha"]
    )

    result["right_tail_thickness"] = (
        1.0 / result["right_alpha"]
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
        / "dynamic_tail_estimates.csv"
    )

    result.to_csv(
        output_path
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        result.index,
        result["left_alpha"],
        linewidth=1.5,
        label="Left-tail α",
    )

    plt.plot(
        result.index,
        result["right_alpha"],
        linewidth=1.5,
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
        / "dynamic_real_return_tail_index.png"
    )

    plt.savefig(
        alpha_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        result.index,
        result["left_k"],
        linewidth=1.5,
        label="Left-tail k",
    )

    plt.plot(
        result.index,
        result["right_k"],
        linewidth=1.5,
        label="Right-tail k",
    )

    plt.xlabel("Date")
    plt.ylabel("Selected k")
    plt.title(
        "Dynamic Hill Threshold Selection"
    )
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()

    k_path = (
        FIGURES_DIR
        / "dynamic_hill_thresholds.png"
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
        f"Tail-index figure: {alpha_path}"
    )

    print(
        f"Threshold figure: {k_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
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

K_MIN = 10
K_STEP = 5


def prepare_tail(sample, side):
    x = np.asarray(sample, dtype=float)

    if side == "left":
        x = -x
    elif side != "right":
        raise ValueError("side must be 'left' or 'right'")

    x = x[np.isfinite(x)]
    x = x[x > 0]

    if len(x) < 20:
        raise ValueError("Insufficient tail observations.")

    return np.sort(x)[::-1]


def calculate_hill_curve(sample, side):
    tail = prepare_tail(sample, side)

    k_max = len(tail) - 1

    if k_max < K_MIN:
        raise ValueError("Insufficient observations for Hill estimation.")

    k_values = np.arange(
        K_MIN,
        k_max + 1,
        K_STEP,
    )

    estimates = np.array(
        [
            hill_estimator(
                tail,
                int(k),
            )
            for k in k_values
        ]
    )

    return k_values, estimates, len(tail)


def analyze_series(data, column, label):
    output = []

    for side in ["left", "right"]:
        sample = data[column].values

        k_values, estimates, tail_size = calculate_hill_curve(
            sample,
            side,
        )

        for k, alpha in zip(
            k_values,
            estimates,
        ):
            output.append(
                {
                    "series": label,
                    "return_type": column,
                    "tail": side,
                    "k": int(k),
                    "alpha_hat": float(alpha),
                    "tail_size": tail_size,
                }
            )

    return pd.DataFrame(output)


def plot_hill(results, return_type):
    for side in ["left", "right"]:
        subset = results[
            (results["return_type"] == return_type)
            & (results["tail"] == side)
        ]

        plt.figure(figsize=(10, 6))

        for series in subset["series"].unique():
            series_data = subset[
                subset["series"] == series
            ]

            plt.plot(
                series_data["k"],
                series_data["alpha_hat"],
                linewidth=1.5,
                label=series,
            )

        plt.xlabel("k")
        plt.ylabel("Hill tail-index estimate α̂")
        plt.title(
            f"{return_type} — {side.title()} Tail Hill Plot"
        )
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        path = (
            FIGURES_DIR
            / f"hill_{return_type}_{side}.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["Unnamed: 0"],
    )

    data = data.rename(
        columns={
            "Unnamed: 0": "date"
        }
    )

    data = data.set_index("date")
    data = data.sort_index()

    required_columns = [
        "nominal_return",
        "real_return",
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

    nominal = data[
        ["nominal_return"]
    ].dropna()

    real = data[
        ["real_return"]
    ].dropna()

    results_nominal = analyze_series(
        nominal,
        "nominal_return",
        "Nominal",
    )

    results_real = analyze_series(
        real,
        "real_return",
        "Real",
    )

    results = pd.concat(
        [
            results_nominal,
            results_real,
        ],
        ignore_index=True,
    )

    output_path = (
        TABLES_DIR
        / "nominal_vs_real_hill_curves.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    plot_hill(
        results,
        "nominal_return",
    )

    plot_hill(
        results,
        "real_return",
    )

    summary_records = []

    for series in ["Nominal", "Real"]:
        for tail in ["left", "right"]:
            subset = results[
                (results["series"] == series)
                & (results["tail"] == tail)
            ]

            preferred_k = min(
                50,
                subset["k"].max(),
            )

            preferred = subset[
                subset["k"] == preferred_k
            ]

            if preferred.empty:
                preferred = subset.iloc[[0]]

            summary_records.append(
                {
                    "series": series,
                    "tail": tail,
                    "reference_k": int(
                        preferred["k"].iloc[0]
                    ),
                    "reference_alpha": float(
                        preferred["alpha_hat"].iloc[0]
                    ),
                    "tail_size": int(
                        subset["tail_size"].iloc[0]
                    ),
                }
            )

    summary = pd.DataFrame(
        summary_records
    )

    summary_path = (
        TABLES_DIR
        / "nominal_vs_real_tail_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("=" * 70)
    print("M1.2 — NOMINAL VS REAL TAIL DIAGNOSTICS")
    print("=" * 70)
    print(
        summary.to_string(
            index=False
        )
    )
    print()
    print(
        f"Results saved to: {output_path}"
    )
    print(
        f"Summary saved to: {summary_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
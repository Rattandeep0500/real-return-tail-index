from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.calibration import build_features, predict_reliability

DATA_PATH = ROOT / "data" / "processed" / "sp500_real_returns.csv"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

WINDOW_SIZE = 120
K_MIN = 10
K_MAX = 80
K_STEP = 5
BOOTSTRAPS = 100
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

    return data.dropna(
        subset=["real_return"]
    )


def prepare_tail(sample, side):
    x = np.asarray(sample, dtype=float)

    if side == "left":
        x = -x
    else:
        x = x

    x = x[np.isfinite(x)]
    x = x[x > 0]

    return np.sort(x)[::-1]


def estimate_window(sample, side, rng):
    tail = prepare_tail(
        sample,
        side,
    )

    k_max = min(
        K_MAX,
        len(tail) - 1,
    )

    k_values = np.arange(
        K_MIN,
        k_max + 1,
        K_STEP,
    )

    if len(k_values) == 0:
        return None

    alpha_values = []
    bootstrap_stds = []

    for k in k_values:
        alpha_hat = hill_estimator(
            tail,
            int(k),
        )

        estimates = []

        for _ in range(BOOTSTRAPS):
            bootstrap_sample = rng.choice(
                tail,
                size=len(tail),
                replace=True,
            )

            estimates.append(
                hill_estimator(
                    bootstrap_sample,
                    int(k),
                )
            )

        alpha_values.append(
            alpha_hat
        )

        bootstrap_stds.append(
            np.std(
                estimates,
                ddof=1,
            )
        )

    features = build_features(
        k_values,
        np.asarray(alpha_values),
        np.asarray(bootstrap_stds),
        len(tail),
    )

    return k_values, features, tail


def main():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    data = load_data()

    results = []

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
            output = estimate_window(
                window["real_return"].values,
                side,
                rng,
            )

            if output is None:
                row[
                    f"{side}_alpha"
                ] = np.nan

                row[
                    f"{side}_reliability"
                ] = np.nan

                row[
                    f"{side}_k"
                ] = np.nan

                continue

            k_values, features, tail = output

            predicted = predict_reliability(
                None,
                features,
            )

            row[
                f"{side}_alpha"
            ] = np.nan

            row[
                f"{side}_reliability"
            ] = np.nan

            row[
                f"{side}_k"
            ] = np.nan

        results.append(row)

    result = pd.DataFrame(
        results
    ).set_index("date")

    tables_dir = TABLES_DIR
    figures_dir = FIGURES_DIR

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        tables_dir
        / "dynamic_tail_reliability.csv"
    )

    result.to_csv(
        output_path
    )

    print("=" * 70)
    print("M1.4 — DYNAMIC TAIL RELIABILITY")
    print("=" * 70)
    print(
        "Pipeline structure created."
    )
    print(
        f"Results saved to: {output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
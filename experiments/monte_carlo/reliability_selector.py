from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.selector import (
    compute_reliability_index,
    select_reliable_k,
)


TRUE_ALPHA = 3.0
SAMPLE_SIZE = 10000
RANDOM_SEED = 42
K_MIN = 20
K_MAX = 2000
K_STEP = 10
BOOTSTRAPS = 250
WINDOW_SIZE = 15


def simulate_pareto(alpha, n, rng):
    u = rng.uniform(size=n)
    return u ** (-1.0 / alpha)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    sample = simulate_pareto(
        TRUE_ALPHA,
        SAMPLE_SIZE,
        rng,
    )

    k_values = np.arange(
        K_MIN,
        K_MAX + 1,
        K_STEP,
    )

    alpha_values = []
    bootstrap_stds = []

    for k in k_values:
        alpha_hat = hill_estimator(
            sample,
            int(k),
        )

        bootstrap_estimates = []

        for _ in range(BOOTSTRAPS):
            bootstrap_sample = rng.choice(
                sample,
                size=SAMPLE_SIZE,
                replace=True,
            )

            bootstrap_estimates.append(
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
                bootstrap_estimates,
                ddof=1,
            )
        )

    alpha_values = np.asarray(alpha_values)
    bootstrap_stds = np.asarray(
        bootstrap_stds
    )

    result = compute_reliability_index(
        k_values=k_values,
        alpha_values=alpha_values,
        bootstrap_std=bootstrap_stds,
        sample_size=SAMPLE_SIZE,
        window_size=WINDOW_SIZE,
    )

    selection = select_reliable_k(
        result
    )

    tables_dir = ROOT / "tables"
    figures_dir = ROOT / "figures"

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        tables_dir
        / "reliability_aware_selector.csv"
    )

    result.to_csv(
        result_path,
        index=False,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        result["k"],
        result["reliability_score"],
        linewidth=1.5,
        label="Reliability Index",
    )

    if selection["status"] == "SELECT":
        plt.axvline(
            selection["k"],
            linestyle="--",
            linewidth=1.5,
            label=f"Selected k = {selection['k']}",
        )

    plt.xlabel("k")
    plt.ylabel("Reliability Index")
    plt.title(
        "Reliability-Aware Hill Threshold Selection"
    )
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    figure_path = (
        figures_dir
        / "reliability_aware_selector.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 70)
    print("M0.8 — RELIABILITY-AWARE HILL SELECTOR")
    print("=" * 70)
    print(f"True alpha              : {TRUE_ALPHA:.6f}")
    print(f"Status                  : {selection['status']}")

    if selection["k"] is not None:
        selected_alpha = hill_estimator(
            sample,
            selection["k"],
        )

        print(
            f"Selected k              : "
            f"{selection['k']}"
        )

        print(
            f"Selected alpha_hat      : "
            f"{selected_alpha:.6f}"
        )

        print(
            f"Absolute error          : "
            f"{abs(selected_alpha - TRUE_ALPHA):.6f}"
        )

    print(
        f"Reliability             : "
        f"{selection['reliability']}"
    )

    print(
        f"Decision                : "
        f"{selection['reason']}"
    )

    print(
        f"Results saved to        : "
        f"{result_path}"
    )

    print(
        f"Figure saved to         : "
        f"{figure_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
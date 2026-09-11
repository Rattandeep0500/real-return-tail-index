from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.reliability import compute_tail_reliability, select_reliable_k


TRUE_ALPHA = 3.0
SAMPLE_SIZE = 10000
RANDOM_SEED = 42
K_MIN = 20
K_MAX = 2000
K_STEP = 10
BOOTSTRAPS = 300
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
        alpha_hat = hill_estimator(sample, int(k))
        bootstrap_estimates = []

        for _ in range(BOOTSTRAPS):
            bootstrap_sample = rng.choice(
                sample,
                size=len(sample),
                replace=True,
            )

            bootstrap_estimates.append(
                hill_estimator(
                    bootstrap_sample,
                    int(k),
                )
            )

        alpha_values.append(alpha_hat)
        bootstrap_stds.append(
            np.std(
                bootstrap_estimates,
                ddof=1,
            )
        )

    alpha_values = np.asarray(alpha_values)
    bootstrap_stds = np.asarray(bootstrap_stds)

    reliability = compute_tail_reliability(
        k_values=k_values,
        alpha_values=alpha_values,
        bootstrap_std=bootstrap_stds,
        window_size=WINDOW_SIZE,
    )

    selected_k, selected_reliability = select_reliable_k(
        reliability
    )

    selected_alpha = hill_estimator(
        sample,
        selected_k,
    )

    figures_dir = ROOT / "figures"
    tables_dir = ROOT / "tables"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = tables_dir / "tail_reliability_results.csv"
    reliability.to_csv(
        results_path,
        index=False,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        reliability["k"],
        reliability["reliability_score"],
        linewidth=1.5,
        label="Tail reliability",
    )

    plt.axvline(
        selected_k,
        linestyle="--",
        linewidth=1.5,
        label=f"Selected k = {selected_k}",
    )

    plt.xlabel("k")
    plt.ylabel("Reliability score")
    plt.title("Adaptive Tail Reliability Score")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    figure_path = figures_dir / "tail_reliability_score.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 70)
    print("M0.7 — TAIL RELIABILITY ENGINE")
    print("=" * 70)
    print(f"True alpha               : {TRUE_ALPHA:.6f}")
    print(f"Selected k               : {selected_k}")
    print(f"Selected alpha_hat       : {selected_alpha:.6f}")
    print(f"Reliability score        : {selected_reliability:.6f}")
    print(
        f"Absolute estimation error: "
        f"{abs(selected_alpha - TRUE_ALPHA):.6f}"
    )
    print(f"Results saved to         : {results_path}")
    print(f"Figure saved to          : {figure_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
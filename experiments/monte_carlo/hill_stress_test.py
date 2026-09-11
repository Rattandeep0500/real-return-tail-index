from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator


TRUE_ALPHAS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SAMPLE_SIZES = [1000, 5000, 10000, 50000]
K_VALUES = [25, 50, 100, 250, 500]
N_REPLICATIONS = 300
RANDOM_SEED = 42


def simulate_pareto(alpha, n, rng):
    uniforms = rng.uniform(size=n)
    return uniforms ** (-1.0 / alpha)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for alpha in TRUE_ALPHAS:
        for n in SAMPLE_SIZES:
            for k in K_VALUES:
                if k >= n:
                    continue

                estimates = []

                for _ in range(N_REPLICATIONS):
                    sample = simulate_pareto(alpha, n, rng)
                    estimate = hill_estimator(sample, k)
                    estimates.append(estimate)

                estimates = np.asarray(estimates)

                bias = np.mean(estimates) - alpha
                variance = np.var(estimates, ddof=1)
                rmse = np.sqrt(np.mean((estimates - alpha) ** 2))

                records.append(
                    {
                        "true_alpha": alpha,
                        "sample_size": n,
                        "k": k,
                        "mean_alpha_hat": np.mean(estimates),
                        "bias": bias,
                        "variance": variance,
                        "rmse": rmse,
                    }
                )

    results = pd.DataFrame(records)

    tables_dir = ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    results_path = tables_dir / "hill_stress_test_results.csv"
    results.to_csv(results_path, index=False)

    for alpha in TRUE_ALPHAS:
        subset = results[
            (results["true_alpha"] == alpha)
            & (results["sample_size"] == 10000)
        ]

        plt.figure(figsize=(10, 6))

        for n in SAMPLE_SIZES:
            subset_n = results[
                (results["true_alpha"] == alpha)
                & (results["sample_size"] == n)
            ]

            plt.plot(
                subset_n["k"],
                subset_n["rmse"],
                marker="o",
                label=f"n={n}",
            )

        plt.xlabel("k")
        plt.ylabel("RMSE")
        plt.title(f"Hill Estimator RMSE — True α = {alpha}")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

        figure_path = ROOT / "figures" / f"hill_rmse_alpha_{alpha}.png"
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close()

    summary = (
        results.sort_values("rmse")
        .groupby(["true_alpha", "sample_size"], as_index=False)
        .first()
    )

    summary_path = tables_dir / "hill_stress_test_best_k.csv"
    summary.to_csv(summary_path, index=False)

    print("=" * 70)
    print("M0.2 — HILL ESTIMATOR STRESS TEST")
    print("=" * 70)
    print(f"Results saved to: {results_path}")
    print(f"Best-k summary saved to: {summary_path}")
    print()
    print(summary.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
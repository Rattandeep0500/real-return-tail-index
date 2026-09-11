from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator


TRUE_ALPHA = 3.0
SAMPLE_SIZE = 10000
N_REPLICATIONS = 300
RANDOM_SEED = 42
K_VALUES = [25, 50, 100, 250, 500, 1000, 2000, 3000]


def simulate_pareto(alpha, n, rng):
    u = rng.uniform(size=n)
    return u ** (-1.0 / alpha)


def simulate_student_t(alpha, n, rng):
    df = alpha
    x = rng.standard_t(df, size=n)
    return np.abs(x)


def simulate_lognormal(n, rng):
    return np.exp(rng.normal(0, 1, size=n))


def simulate_truncated_pareto(alpha, n, rng):
    x = simulate_pareto(alpha, n, rng)
    return x[x <= 100]


def simulate_mixture(alpha, n, rng):
    pareto = simulate_pareto(alpha, n, rng)
    gaussian = np.abs(rng.normal(1, 0.5, size=n))
    mask = rng.uniform(size=n) < 0.8
    return np.where(mask, pareto, gaussian)


def simulate_regime_mixture(alpha, n, rng):
    n1 = n // 2
    n2 = n - n1

    regime_1 = simulate_pareto(alpha, n1, rng)
    regime_2 = simulate_pareto(alpha * 1.8, n2, rng)

    return np.concatenate([regime_1, regime_2])


def get_simulations(alpha, n, rng):
    return {
        "Pareto": simulate_pareto(alpha, n, rng),
        "Student-t": simulate_student_t(alpha, n, rng),
        "Lognormal": simulate_lognormal(n, rng),
        "Truncated Pareto": simulate_truncated_pareto(alpha, n, rng),
        "Pareto-Gaussian Mixture": simulate_mixture(alpha, n, rng),
        "Regime Mixture": simulate_regime_mixture(alpha, n, rng),
    }


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for distribution_name in [
        "Pareto",
        "Student-t",
        "Lognormal",
        "Truncated Pareto",
        "Pareto-Gaussian Mixture",
        "Regime Mixture",
    ]:
        for replication in range(N_REPLICATIONS):
            sample = get_simulations(
                TRUE_ALPHA,
                SAMPLE_SIZE,
                rng,
            )[distribution_name]

            sample = np.asarray(sample)
            sample = sample[np.isfinite(sample)]
            sample = sample[sample > 0]

            for k in K_VALUES:
                if k >= len(sample):
                    continue

                try:
                    estimate = hill_estimator(sample, k)
                except (ValueError, RuntimeError):
                    continue

                records.append(
                    {
                        "distribution": distribution_name,
                        "replication": replication + 1,
                        "k": k,
                        "alpha_hat": estimate,
                        "bias": estimate - TRUE_ALPHA,
                        "squared_error": (estimate - TRUE_ALPHA) ** 2,
                    }
                )

    results = pd.DataFrame(records)

    tables_dir = ROOT / "tables"
    figures_dir = ROOT / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = tables_dir / "hill_robustness_results.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby(["distribution", "k"])
        .agg(
            mean_alpha_hat=("alpha_hat", "mean"),
            bias=("bias", "mean"),
            variance=("alpha_hat", "var"),
            rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        )
        .reset_index()
    )

    summary_path = tables_dir / "hill_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)

    for distribution in summary["distribution"].unique():
        subset = summary[summary["distribution"] == distribution]

        plt.figure(figsize=(10, 6))
        plt.plot(
            subset["k"],
            subset["mean_alpha_hat"],
            marker="o",
        )
        plt.axhline(
            TRUE_ALPHA,
            linestyle="--",
            linewidth=1.5,
            label=f"Reference α = {TRUE_ALPHA}",
        )
        plt.xlabel("k")
        plt.ylabel("Mean Hill tail-index estimate")
        plt.title(f"Hill Stability — {distribution}")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

        filename = (
            distribution.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        plt.savefig(
            figures_dir / f"hill_robustness_{filename}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    best_k = (
        summary.sort_values("rmse")
        .groupby("distribution", as_index=False)
        .first()
    )

    best_k_path = tables_dir / "hill_robustness_best_k.csv"
    best_k.to_csv(best_k_path, index=False)

    print("=" * 70)
    print("M0.3 — HILL FAILURE & ROBUSTNESS LABORATORY")
    print("=" * 70)
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Best-k results saved to: {best_k_path}")
    print()
    print(best_k.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
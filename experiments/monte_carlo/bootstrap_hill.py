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
K = 500
N_BOOTSTRAPS = 2000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95


def simulate_pareto(alpha, n, rng):
    u = rng.uniform(size=n)
    return u ** (-1.0 / alpha)


def bootstrap_hill(sample, k, n_bootstraps, rng):
    estimates = []

    for _ in range(n_bootstraps):
        bootstrap_sample = rng.choice(
            sample,
            size=len(sample),
            replace=True,
        )

        try:
            estimate = hill_estimator(
                bootstrap_sample,
                k,
            )
            estimates.append(estimate)
        except (ValueError, RuntimeError):
            continue

    return np.asarray(estimates)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    sample = simulate_pareto(
        TRUE_ALPHA,
        SAMPLE_SIZE,
        rng,
    )

    alpha_hat = hill_estimator(sample, K)

    bootstrap_estimates = bootstrap_hill(
        sample,
        K,
        N_BOOTSTRAPS,
        rng,
    )

    lower = np.quantile(
        bootstrap_estimates,
        (1 - CONFIDENCE_LEVEL) / 2,
    )

    upper = np.quantile(
        bootstrap_estimates,
        1 - (1 - CONFIDENCE_LEVEL) / 2,
    )

    bootstrap_mean = np.mean(bootstrap_estimates)
    bootstrap_std = np.std(bootstrap_estimates, ddof=1)

    coverage = lower <= TRUE_ALPHA <= upper

    results = pd.DataFrame(
        {
            "true_alpha": [TRUE_ALPHA],
            "sample_size": [SAMPLE_SIZE],
            "k": [K],
            "alpha_hat": [alpha_hat],
            "bootstrap_mean": [bootstrap_mean],
            "bootstrap_std": [bootstrap_std],
            "ci_lower": [lower],
            "ci_upper": [upper],
            "true_alpha_inside_ci": [coverage],
        }
    )

    tables_dir = ROOT / "tables"
    figures_dir = ROOT / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = tables_dir / "hill_bootstrap_confidence_interval.csv"
    results.to_csv(results_path, index=False)

    plt.figure(figsize=(10, 6))

    plt.hist(
        bootstrap_estimates,
        bins=50,
        density=True,
        alpha=0.75,
    )

    plt.axvline(
        TRUE_ALPHA,
        linestyle="--",
        linewidth=1.5,
        label=f"True α = {TRUE_ALPHA}",
    )

    plt.axvline(
        alpha_hat,
        linestyle=":",
        linewidth=1.5,
        label=f"Original α̂ = {alpha_hat:.3f}",
    )

    plt.axvline(
        lower,
        linestyle="-.",
        linewidth=1.5,
        label=f"95% CI lower = {lower:.3f}",
    )

    plt.axvline(
        upper,
        linestyle="-.",
        linewidth=1.5,
        label=f"95% CI upper = {upper:.3f}",
    )

    plt.xlabel("Bootstrap Hill tail-index estimate")
    plt.ylabel("Density")
    plt.title("Bootstrap Distribution of the Hill Tail Index")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    figure_path = figures_dir / "hill_bootstrap_distribution.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 70)
    print("M0.5 — BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 70)
    print(f"True α                  : {TRUE_ALPHA:.6f}")
    print(f"Original α̂             : {alpha_hat:.6f}")
    print(f"Bootstrap mean          : {bootstrap_mean:.6f}")
    print(f"Bootstrap std           : {bootstrap_std:.6f}")
    print(f"95% CI                  : [{lower:.6f}, {upper:.6f}]")
    print(f"True α inside CI        : {coverage}")
    print(f"Bootstrap samples       : {len(bootstrap_estimates)}")
    print(f"Results saved to        : {results_path}")
    print(f"Figure saved to         : {figure_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
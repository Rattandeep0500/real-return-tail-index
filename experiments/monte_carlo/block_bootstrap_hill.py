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
N_REPLICATIONS = 300
N_BOOTSTRAPS = 500
BLOCK_LENGTH = 50
RANDOM_SEED = 42


def simulate_volatility_clustered(alpha, n, rng):
    innovations = rng.uniform(size=n) ** (-1.0 / alpha)
    volatility = np.empty(n)
    volatility[0] = 1.0

    for t in range(1, n):
        volatility[t] = (
            0.90 * volatility[t - 1]
            + 0.10 * innovations[t - 1]
        )

    return innovations * volatility / np.mean(volatility)


def moving_block_bootstrap(sample, block_length, rng):
    n = len(sample)
    n_blocks = int(np.ceil(n / block_length))

    starts = rng.integers(
        0,
        n - block_length + 1,
        size=n_blocks,
    )

    blocks = [
        sample[start:start + block_length]
        for start in starts
    ]

    bootstrap_sample = np.concatenate(blocks)

    return bootstrap_sample[:n]


def iid_bootstrap(sample, k, rng):
    estimates = []

    for _ in range(N_BOOTSTRAPS):
        bootstrap_sample = rng.choice(
            sample,
            size=len(sample),
            replace=True,
        )

        estimates.append(
            hill_estimator(
                bootstrap_sample,
                k,
            )
        )

    return np.asarray(estimates)


def block_bootstrap(sample, k, block_length, rng):
    estimates = []

    for _ in range(N_BOOTSTRAPS):
        bootstrap_sample = moving_block_bootstrap(
            sample,
            block_length,
            rng,
        )

        estimates.append(
            hill_estimator(
                bootstrap_sample,
                k,
            )
        )

    return np.asarray(estimates)


def confidence_interval(estimates):
    return (
        np.quantile(estimates, 0.025),
        np.quantile(estimates, 0.975),
    )


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    coverage_records = []
    example_sample = None
    example_iid = None
    example_block = None

    for replication in range(N_REPLICATIONS):
        sample = simulate_volatility_clustered(
            TRUE_ALPHA,
            SAMPLE_SIZE,
            rng,
        )

        alpha_hat = hill_estimator(
            sample,
            K,
        )

        iid_estimates = iid_bootstrap(
            sample,
            K,
            rng,
        )

        block_estimates = block_bootstrap(
            sample,
            K,
            BLOCK_LENGTH,
            rng,
        )

        iid_lower, iid_upper = confidence_interval(
            iid_estimates
        )

        block_lower, block_upper = confidence_interval(
            block_estimates
        )

        coverage_records.append(
            {
                "replication": replication + 1,
                "alpha_hat": alpha_hat,
                "iid_lower": iid_lower,
                "iid_upper": iid_upper,
                "iid_coverage": iid_lower <= TRUE_ALPHA <= iid_upper,
                "block_lower": block_lower,
                "block_upper": block_upper,
                "block_coverage": block_lower <= TRUE_ALPHA <= block_upper,
            }
        )

        if replication == 0:
            example_sample = sample
            example_iid = iid_estimates
            example_block = block_estimates

    results = pd.DataFrame(coverage_records)

    iid_coverage = results["iid_coverage"].mean()
    block_coverage = results["block_coverage"].mean()

    iid_width = (
        results["iid_upper"] - results["iid_lower"]
    ).mean()

    block_width = (
        results["block_upper"] - results["block_lower"]
    ).mean()

    summary = pd.DataFrame(
        {
            "method": [
                "IID Bootstrap",
                "Moving Block Bootstrap",
            ],
            "coverage": [
                iid_coverage,
                block_coverage,
            ],
            "mean_ci_width": [
                iid_width,
                block_width,
            ],
        }
    )

    tables_dir = ROOT / "tables"
    figures_dir = ROOT / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = (
        tables_dir / "block_bootstrap_hill_results.csv"
    )

    summary_path = (
        tables_dir / "block_bootstrap_hill_summary.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        example_iid,
        bins=50,
        alpha=0.55,
        label="IID bootstrap",
    )

    plt.hist(
        example_block,
        bins=50,
        alpha=0.55,
        label="Block bootstrap",
    )

    plt.axvline(
        TRUE_ALPHA,
        linestyle="--",
        linewidth=1.5,
        label=f"True α = {TRUE_ALPHA}",
    )

    plt.xlabel("Bootstrap Hill tail-index estimate")
    plt.ylabel("Frequency")
    plt.title("IID vs Block Bootstrap for Dependent Extremes")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    figure_path = (
        figures_dir / "iid_vs_block_bootstrap_hill.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 70)
    print("M0.6 — DEPENDENCE-AWARE HILL INFERENCE")
    print("=" * 70)
    print(f"True α                  : {TRUE_ALPHA:.6f}")
    print(f"Sample size             : {SAMPLE_SIZE}")
    print(f"k                       : {K}")
    print(f"Block length            : {BLOCK_LENGTH}")
    print(f"Replications            : {N_REPLICATIONS}")
    print(f"Bootstrap samples       : {N_BOOTSTRAPS}")
    print()
    print(f"IID coverage            : {iid_coverage:.4f}")
    print(f"Block coverage          : {block_coverage:.4f}")
    print(f"IID mean CI width       : {iid_width:.6f}")
    print(f"Block mean CI width     : {block_width:.6f}")
    print()
    print(f"Results saved to        : {results_path}")
    print(f"Summary saved to        : {summary_path}")
    print(f"Figure saved to         : {figure_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
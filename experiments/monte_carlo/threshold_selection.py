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
RANDOM_SEED = 42

K_MIN = 20
K_MAX = 2000
K_STEP = 10

WINDOW_SIZE = 150
N_BOOTSTRAPS = 300


def simulate_pareto(alpha, n, rng):
    u = rng.uniform(size=n)
    return u ** (-1.0 / alpha)


def calculate_hill_curve(sample):
    k_values = np.arange(K_MIN, K_MAX + 1, K_STEP)
    estimates = np.array(
        [hill_estimator(sample, int(k)) for k in k_values]
    )
    return k_values, estimates


def find_stable_region(sample, k_values, estimates):
    best_start = None
    best_end = None
    best_range = np.inf

    for i in range(len(estimates) - WINDOW_SIZE + 1):
        window = estimates[i:i + WINDOW_SIZE]
        window_range = np.ptp(window)

        if window_range < best_range:
            best_range = window_range
            best_start = i
            best_end = i + WINDOW_SIZE - 1

    if best_start is None:
        raise RuntimeError("No stable region found.")

    stable_k = k_values[best_start:best_end + 1]

    selected_k = int(np.median(stable_k))
    selected_alpha = hill_estimator(sample, selected_k)

    return (
        selected_k,
        selected_alpha,
        int(stable_k[0]),
        int(stable_k[-1]),
        float(best_range),
    )


def bootstrap_k_selection(sample, k_values, rng):
    records = []

    for k in k_values:
        estimates = []

        for _ in range(N_BOOTSTRAPS):
            bootstrap_sample = rng.choice(
                sample,
                size=len(sample),
                replace=True,
            )

            try:
                estimate = hill_estimator(
                    bootstrap_sample,
                    int(k),
                )
                estimates.append(estimate)
            except (ValueError, RuntimeError):
                continue

        if len(estimates) < 10:
            continue

        estimates = np.asarray(estimates)

        records.append(
            {
                "k": int(k),
                "bootstrap_mean": np.mean(estimates),
                "bootstrap_std": np.std(estimates, ddof=1),
                "bootstrap_variance": np.var(estimates, ddof=1),
            }
        )

    result = pd.DataFrame(records)

    if result.empty:
        raise RuntimeError("Bootstrap produced no valid estimates.")

    result["stability_score"] = result["bootstrap_std"]

    best = result.loc[result["stability_score"].idxmin()]

    return result, int(best["k"])


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    sample = simulate_pareto(
        TRUE_ALPHA,
        SAMPLE_SIZE,
        rng,
    )

    k_values, estimates = calculate_hill_curve(sample)

    (
        selected_k,
        selected_alpha,
        stable_start,
        stable_end,
        stable_range,
    ) = find_stable_region(
        sample,
        k_values,
        estimates,
    )

    bootstrap_results, bootstrap_k = bootstrap_k_selection(
        sample,
        k_values,
        rng,
    )

    figures_dir = ROOT / "figures"
    tables_dir = ROOT / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    hill_results = pd.DataFrame(
        {
            "k": k_values,
            "alpha_hat": estimates,
        }
    )

    hill_results.to_csv(
        tables_dir / "threshold_hill_curve.csv",
        index=False,
    )

    bootstrap_results.to_csv(
        tables_dir / "threshold_bootstrap_results.csv",
        index=False,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        k_values,
        estimates,
        linewidth=1.5,
        label="Hill estimate",
    )

    plt.axhline(
        TRUE_ALPHA,
        linestyle="--",
        linewidth=1.5,
        label=f"True α = {TRUE_ALPHA}",
    )

    plt.axvspan(
        stable_start,
        stable_end,
        alpha=0.20,
        label="Selected stability region",
    )

    plt.axvline(
        selected_k,
        linestyle=":",
        linewidth=1.5,
        label=f"Stability k = {selected_k}",
    )

    plt.axvline(
        bootstrap_k,
        linestyle="-.",
        linewidth=1.5,
        label=f"Bootstrap k = {bootstrap_k}",
    )

    plt.xlabel("Number of upper-order statistics k")
    plt.ylabel("Hill tail-index estimate α̂")
    plt.title("Hill Threshold Selection")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    figure_path = figures_dir / "hill_threshold_selection.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 70)
    print("M0.4 — THRESHOLD SELECTION")
    print("=" * 70)
    print(f"True α                   : {TRUE_ALPHA}")
    print(f"Stability-region k       : {stable_start} - {stable_end}")
    print(f"Stability-selected k     : {selected_k}")
    print(f"Stability-selected α̂    : {selected_alpha:.6f}")
    print(f"Stability range          : {stable_range:.6f}")
    print(f"Bootstrap-selected k     : {bootstrap_k}")
    print(f"Figure saved to          : {figure_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
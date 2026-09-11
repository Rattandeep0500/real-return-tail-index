from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator, hill_curve


TRUE_ALPHA = 3.0
SAMPLE_SIZE = 10_000
RANDOM_SEED = 42
K_MIN = 50
K_MAX = 2_000
K_STEP = 10
N_REPLICATIONS = 500


def simulate_pareto(
    alpha: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    if n <= 0:
        raise ValueError("n must be positive.")

    uniforms = rng.uniform(size=n)
    return uniforms ** (-1.0 / alpha)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 70)
    print("REAL-RETURN TAIL INDEX PROJECT")
    print("M0 — Monte Carlo Hill Laboratory")
    print("=" * 70)
    print(f"True tail index α       : {TRUE_ALPHA}")
    print(f"Sample size n           : {SAMPLE_SIZE}")
    print(f"Monte Carlo repetitions : {N_REPLICATIONS}")
    print()

    sample = simulate_pareto(
        alpha=TRUE_ALPHA,
        n=SAMPLE_SIZE,
        rng=rng,
    )

    k_values = np.arange(K_MIN, K_MAX + 1, K_STEP)
    estimates = hill_curve(sample, k_values)

    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / "hill_plot_pareto.png"

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, estimates, linewidth=1.5)
    plt.axhline(
        TRUE_ALPHA,
        linestyle="--",
        linewidth=1.5,
        label=f"True α = {TRUE_ALPHA}",
    )
    plt.xlabel("Number of upper-order statistics k")
    plt.ylabel("Hill tail-index estimate α̂")
    plt.title("Hill Plot — Pareto Simulation")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Hill plot saved to: {figure_path}")
    print()

    selected_k = 500

    alpha_hat = hill_estimator(
        sample,
        selected_k,
    )

    print(f"Selected k             : {selected_k}")
    print(f"Estimated α             : {alpha_hat:.4f}")
    print(f"Estimation error        : {alpha_hat - TRUE_ALPHA:+.4f}")
    print()

    records = []

    for replication in range(N_REPLICATIONS):
        sim_sample = simulate_pareto(
            alpha=TRUE_ALPHA,
            n=SAMPLE_SIZE,
            rng=rng,
        )

        estimate = hill_estimator(
            sim_sample,
            selected_k,
        )

        records.append(
            {
                "replication": replication + 1,
                "alpha_hat": estimate,
                "error": estimate - TRUE_ALPHA,
                "squared_error": (estimate - TRUE_ALPHA) ** 2,
            }
        )

    results = pd.DataFrame(records)

    bias = results["alpha_hat"].mean() - TRUE_ALPHA
    variance = results["alpha_hat"].var(ddof=1)
    rmse = np.sqrt(results["squared_error"].mean())

    print("=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)
    print(f"Mean α̂               : {results['alpha_hat'].mean():.4f}")
    print(f"Bias                  : {bias:+.4f}")
    print(f"Variance              : {variance:.6f}")
    print(f"RMSE                  : {rmse:.4f}")
    print(f"Std. deviation        : {results['alpha_hat'].std(ddof=1):.4f}")

    results_dir = ROOT / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "hill_monte_carlo_results.csv"

    results.to_csv(
        results_path,
        index=False,
    )

    print()
    print(f"Monte Carlo results saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
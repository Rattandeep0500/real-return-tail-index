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
N_REPLICATIONS = 60
N_BOOTSTRAPS = 80
K_VALUES = np.arange(50, 1001, 25)
WINDOW_SIZE = 15
RANDOM_SEED = 42


def simulate_pareto(alpha, n, rng):
    u = rng.uniform(size=n)
    return u ** (-1.0 / alpha)


def simulate_student_t(alpha, n, rng):
    return np.abs(rng.standard_t(df=alpha, size=n))


def simulate_mixture(alpha, n, rng):
    pareto = simulate_pareto(alpha, n, rng)
    gaussian = np.abs(rng.normal(1.0, 0.5, size=n))
    mask = rng.uniform(size=n) < 0.8
    return np.where(mask, pareto, gaussian)


def simulate_regime_mixture(alpha, n, rng):
    n1 = n // 2
    n2 = n - n1

    regime_1 = simulate_pareto(alpha, n1, rng)
    regime_2 = simulate_pareto(alpha + 2.0, n2, rng)

    return np.concatenate(
        [regime_1, regime_2]
    )


def get_sample(name, rng):
    if name == "Pareto":
        return simulate_pareto(
            TRUE_ALPHA,
            SAMPLE_SIZE,
            rng,
        )

    if name == "Student-t":
        return simulate_student_t(
            TRUE_ALPHA,
            SAMPLE_SIZE,
            rng,
        )

    if name == "Pareto-Gaussian Mixture":
        return simulate_mixture(
            TRUE_ALPHA,
            SAMPLE_SIZE,
            rng,
        )

    if name == "Regime Mixture":
        return simulate_regime_mixture(
            TRUE_ALPHA,
            SAMPLE_SIZE,
            rng,
        )

    raise ValueError(f"Unknown distribution: {name}")


def evaluate_sample(sample, rng):
    alpha_values = []
    bootstrap_stds = []

    for k in K_VALUES:
        alpha_hat = hill_estimator(
            sample,
            int(k),
        )

        bootstrap_estimates = []

        for _ in range(N_BOOTSTRAPS):
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

        alpha_values.append(alpha_hat)
        bootstrap_stds.append(
            np.std(
                bootstrap_estimates,
                ddof=1,
            )
        )

    reliability = compute_reliability_index(
        k_values=K_VALUES,
        alpha_values=np.asarray(alpha_values),
        bootstrap_std=np.asarray(bootstrap_stds),
        sample_size=SAMPLE_SIZE,
        window_size=WINDOW_SIZE,
    )

    selection = select_reliable_k(
        reliability
    )

    if selection["status"] == "SELECT":
        selected_k = selection["k"]
        selected_alpha = hill_estimator(
            sample,
            selected_k,
        )
        error = abs(
            selected_alpha - TRUE_ALPHA
        )
    else:
        selected_k = np.nan
        selected_alpha = np.nan
        error = np.nan

    return (
        selection,
        selected_k,
        selected_alpha,
        error,
        reliability,
    )


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    distributions = [
        "Pareto",
        "Student-t",
        "Pareto-Gaussian Mixture",
        "Regime Mixture",
    ]

    records = []
    curves = []

    for distribution in distributions:
        for replication in range(
            N_REPLICATIONS
        ):
            sample = get_sample(
                distribution,
                rng,
            )

            (
                selection,
                selected_k,
                selected_alpha,
                error,
                reliability,
            ) = evaluate_sample(
                sample,
                rng,
            )

            records.append(
                {
                    "distribution": distribution,
                    "replication": replication + 1,
                    "status": selection["status"],
                    "selected_k": selected_k,
                    "selected_alpha": selected_alpha,
                    "reliability": selection["reliability"],
                    "absolute_error": error,
                }
            )

            reliability_copy = reliability.copy()
            reliability_copy["distribution"] = distribution
            reliability_copy["replication"] = (
                replication + 1
            )

            curves.append(
                reliability_copy
            )

    results = pd.DataFrame(records)
    curve_results = pd.concat(
        curves,
        ignore_index=True,
    )

    valid = results.dropna(
        subset=[
            "reliability",
            "absolute_error",
        ]
    )

    calibration_records = []

    for distribution in distributions:
        subset = valid[
            valid["distribution"] == distribution
        ]

        if len(subset) >= 3:
            correlation = subset[
                [
                    "reliability",
                    "absolute_error",
                ]
            ].corr().iloc[0, 1]
        else:
            correlation = np.nan

        calibration_records.append(
            {
                "distribution": distribution,
                "mean_reliability": subset[
                    "reliability"
                ].mean(),
                "mean_absolute_error": subset[
                    "absolute_error"
                ].mean(),
                "median_absolute_error": subset[
                    "absolute_error"
                ].median(),
                "selection_rate": (
                    results[
                        results["distribution"]
                        == distribution
                    ]["status"]
                    == "SELECT"
                ).mean(),
                "reliability_error_correlation": correlation,
            }
        )

    calibration = pd.DataFrame(
        calibration_records
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

    results.to_csv(
        tables_dir / "reliability_calibration_results.csv",
        index=False,
    )

    calibration.to_csv(
        tables_dir / "reliability_calibration_summary.csv",
        index=False,
    )

    curve_results.to_csv(
        tables_dir / "reliability_calibration_curves.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))

    for distribution in distributions:
        subset = valid[
            valid["distribution"] == distribution
        ]

        plt.scatter(
            subset["reliability"],
            subset["absolute_error"],
            alpha=0.45,
            label=distribution,
        )

    plt.xlabel("Reliability score")
    plt.ylabel("Absolute tail-index error")
    plt.title("Tail Reliability Calibration")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    figure_path = (
        figures_dir
        / "tail_reliability_calibration.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 70)
    print("M0.9 — RELIABILITY CALIBRATION")
    print("=" * 70)
    print(calibration.to_string(index=False))
    print()
    print(
        f"Results saved to: "
        f"{tables_dir / 'reliability_calibration_results.csv'}"
    )
    print(
        f"Summary saved to: "
        f"{tables_dir / 'reliability_calibration_summary.csv'}"
    )
    print(
        f"Figure saved to: "
        f"{figure_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
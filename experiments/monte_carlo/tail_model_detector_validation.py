from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.tail_model_detector import analyze_tail_model


RANDOM_SEED = 2026

SAMPLE_SIZE = 120

REPLICATIONS = 300

K_VALUES = np.arange(
    10,
    81,
    5,
)


def simulate_pareto(
    alpha,
    n,
    rng,
):
    u = rng.uniform(
        0.0,
        1.0,
        size=n,
    )

    return u ** (
        -1.0 / alpha
    )


def simulate_student_t(
    alpha,
    n,
    rng,
):
    return np.abs(
        rng.standard_t(
            df=alpha,
            size=n,
        )
    )


def simulate_lognormal(
    n,
    rng,
):
    return np.exp(
        rng.normal(
            0.0,
            1.0,
            size=n,
        )
    )


def simulate_truncated_pareto(
    alpha,
    n,
    rng,
    cutoff=100.0,
):
    values = []

    while len(values) < n:
        batch = simulate_pareto(
            alpha,
            n,
            rng,
        )

        batch = batch[
            batch <= cutoff
        ]

        values.extend(
            batch.tolist()
        )

    return np.asarray(
        values[:n],
        dtype=float,
    )


def simulate_mixture(
    alpha,
    n,
    rng,
):
    n_tail = rng.binomial(
        n,
        0.80,
    )

    n_body = n - n_tail

    tail = simulate_pareto(
        alpha,
        n_tail,
        rng,
    )

    body = np.abs(
        rng.normal(
            1.0,
            0.5,
            size=n_body,
        )
    )

    return np.concatenate(
        [
            tail,
            body,
        ]
    )


def simulate_regime_switch(
    alpha,
    n,
    rng,
):
    split = rng.integers(
        40,
        81,
    )

    alpha_1 = rng.uniform(
        max(
            1.2,
            alpha - 0.7,
        ),
        alpha + 0.2,
    )

    alpha_2 = rng.uniform(
        max(
            1.2,
            alpha - 0.2,
        ),
        alpha + 0.7,
    )

    first = simulate_pareto(
        alpha_1,
        split,
        rng,
    )

    second = simulate_pareto(
        alpha_2,
        n - split,
        rng,
    )

    return np.concatenate(
        [
            first,
            second,
        ]
    )


def simulate_volatility_clustered(
    alpha,
    n,
    rng,
):
    innovations = simulate_pareto(
        alpha,
        n,
        rng,
    )

    sigma = np.empty(n)
    sigma[0] = 1.0

    for t in range(
        1,
        n,
    ):
        sigma[t] = (
            0.94 * sigma[t - 1]
            + 0.06 * innovations[t - 1]
        )

    mean_sigma = sigma.mean()

    if not np.isfinite(
        mean_sigma
    ):
        mean_sigma = 1.0

    if mean_sigma <= 0:
        mean_sigma = 1.0

    return (
        innovations
        * sigma
        / mean_sigma
    )


def generate_sample(
    distribution,
    alpha,
    rng,
):
    if distribution == "Pareto":
        return simulate_pareto(
            alpha,
            SAMPLE_SIZE,
            rng,
        )

    if distribution == "Student-t":
        return simulate_student_t(
            alpha,
            SAMPLE_SIZE,
            rng,
        )

    if distribution == "Lognormal":
        return simulate_lognormal(
            SAMPLE_SIZE,
            rng,
        )

    if distribution == "Truncated-Pareto":
        return simulate_truncated_pareto(
            alpha,
            SAMPLE_SIZE,
            rng,
        )

    if distribution == "Mixture":
        return simulate_mixture(
            alpha,
            SAMPLE_SIZE,
            rng,
        )

    if distribution == "Regime-Switch":
        return simulate_regime_switch(
            alpha,
            SAMPLE_SIZE,
            rng,
        )

    if distribution == "Volatility-Clustered":
        return simulate_volatility_clustered(
            alpha,
            SAMPLE_SIZE,
            rng,
        )

    raise ValueError(
        f"Unknown distribution: {distribution}"
    )


def main():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    distributions = [
        "Pareto",
        "Student-t",
        "Lognormal",
        "Truncated-Pareto",
        "Mixture",
        "Regime-Switch",
        "Volatility-Clustered",
    ]

    rows = []

    total = (
        len(distributions)
        * REPLICATIONS
    )

    completed = 0

    print("=" * 70)
    print(
        "M1.8b - TAIL MODEL DETECTOR VALIDATION"
    )
    print("=" * 70)

    for distribution in distributions:
        for replication in range(
            REPLICATIONS
        ):
            alpha = rng.uniform(
                1.5,
                5.0,
            )

            sample = generate_sample(
                distribution,
                alpha,
                rng,
            )

            features, _ = (
                analyze_tail_model(
                    sample,
                    K_VALUES,
                )
            )

            rows.append(
                {
                    "distribution": distribution,
                    "replication": replication,
                    "simulation_alpha": alpha,
                    "model_validity_score": (
                        features[
                            "model_validity_score"
                        ]
                    ),
                    "model_status": (
                        features[
                            "model_status"
                        ]
                    ),
                    "hill_alpha_median": (
                        features[
                            "hill_alpha_median"
                        ]
                    ),
                    "hill_relative_range": (
                        features[
                            "hill_relative_range"
                        ]
                    ),
                    "hill_relative_std": (
                        features[
                            "hill_relative_std"
                        ]
                    ),
                    "pareto_loglog_r2": (
                        features[
                            "pareto_loglog_r2"
                        ]
                    ),
                    "pareto_loglog_max_residual": (
                        features[
                            "pareto_loglog_max_residual"
                        ]
                    ),
                    "mean_excess_loglog_r2": (
                        features[
                            "mean_excess_loglog_r2"
                        ]
                    ),
                    "selected_k": (
                        features[
                            "selected_k"
                        ]
                    ),
                }
            )

            completed += 1

            if completed % 50 == 0:
                print(
                    f"Progress: "
                    f"{completed}/{total}",
                    flush=True,
                )

    results = pd.DataFrame(
        rows
    )

    summary = (
        results.groupby(
            "distribution"
        )
        .agg(
            mean_validity=(
                "model_validity_score",
                "mean",
            ),
            median_validity=(
                "model_validity_score",
                "median",
            ),
            std_validity=(
                "model_validity_score",
                "std",
            ),
            supported_rate=(
                "model_status",
                lambda x: np.mean(
                    x == "SUPPORTED"
                ),
            ),
            uncertain_rate=(
                "model_status",
                lambda x: np.mean(
                    x == "UNCERTAIN"
                ),
            ),
            weak_rate=(
                "model_status",
                lambda x: np.mean(
                    x == "WEAK"
                ),
            ),
            mean_hill_alpha=(
                "hill_alpha_median",
                "mean",
            ),
            mean_loglog_r2=(
                "pareto_loglog_r2",
                "mean",
            ),
            mean_relative_range=(
                "hill_relative_range",
                "mean",
            ),
            mean_selected_k=(
                "selected_k",
                "mean",
            ),
            observations=(
                "model_validity_score",
                "size",
            ),
        )
        .reset_index()
    )

    tables_dir = ROOT / "tables"

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        tables_dir
        / "tail_model_detector_validation.csv"
    )

    summary_path = (
        tables_dir
        / "tail_model_detector_summary.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: "
        f"{results_path}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
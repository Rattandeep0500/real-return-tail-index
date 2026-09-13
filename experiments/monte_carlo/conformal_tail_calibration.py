from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.conformal_reliability import (
    conformal_radius,
    relative_prediction_interval,
)


RANDOM_SEED = 2026

SAMPLE_SIZE = 120

CALIBRATION_REPLICATIONS = 300
TEST_REPLICATIONS = 300

MIS_COVERAGE = 0.10

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

    if not np.isfinite(mean_sigma):
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

    if distribution == "Mixture":
        return simulate_mixture(
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


def estimate_hill_curve(
    sample,
):
    sample = np.asarray(
        sample,
        dtype=float,
    )

    sample = sample[
        np.isfinite(sample)
        & (sample > 0)
    ]

    sample = np.sort(
        sample
    )[::-1]

    k_values = K_VALUES[
        K_VALUES < len(sample) - 1
    ]

    estimates = []

    for k in k_values:
        try:
            estimate = hill_estimator(
                sample,
                int(k),
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ):
            estimate = np.nan

        estimates.append(
            estimate
        )

    return (
        k_values,
        np.asarray(
            estimates,
            dtype=float,
        ),
    )


def build_calibration_scores(
    distributions,
    rng,
):
    scores = []

    total = (
        len(distributions)
        * CALIBRATION_REPLICATIONS
    )

    completed = 0

    for distribution in distributions:
        for _ in range(
            CALIBRATION_REPLICATIONS
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

            k_values, estimates = (
                estimate_hill_curve(
                    sample
                )
            )

            relative_errors = (
                np.abs(
                    estimates - alpha
                )
                / alpha
            )

            valid = np.isfinite(
                relative_errors
            )

            scores.extend(
                relative_errors[
                    valid
                ].tolist()
            )

            completed += 1

            if completed % 50 == 0:
                print(
                    f"Calibration progress: "
                    f"{completed}/{total}",
                    flush=True,
                )

    if len(scores) == 0:
        raise ValueError(
            "No valid calibration scores."
        )

    return np.asarray(
        scores,
        dtype=float,
    )


def build_test_results(
    distributions,
    radius,
    rng,
):
    records = []

    total = (
        len(distributions)
        * TEST_REPLICATIONS
    )

    completed = 0

    for distribution in distributions:
        for replication in range(
            TEST_REPLICATIONS
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

            k_values, estimates = (
                estimate_hill_curve(
                    sample
                )
            )

            lower, upper = (
                relative_prediction_interval(
                    estimates,
                    radius,
                )
            )

            covered = (
                (alpha >= lower)
                & (alpha <= upper)
            )

            relative_width = (
                upper - lower
            ) / max(
                alpha,
                1e-12,
            )

            for i, k in enumerate(
                k_values
            ):
                if not np.isfinite(
                    estimates[i]
                ):
                    continue

                records.append(
                    {
                        "distribution": distribution,
                        "replication": replication,
                        "true_alpha": float(
                            alpha
                        ),
                        "k": int(k),
                        "alpha_hat": float(
                            estimates[i]
                        ),
                        "lower": float(
                            lower[i]
                        ),
                        "upper": float(
                            upper[i]
                        ),
                        "relative_width": float(
                            relative_width[i]
                        ),
                        "covered": bool(
                            covered[i]
                        ),
                    }
                )

            completed += 1

            if completed % 50 == 0:
                print(
                    f"Test progress: "
                    f"{completed}/{total}",
                    flush=True,
                )

    if not records:
        raise ValueError(
            "No valid test results."
        )

    return pd.DataFrame(
        records
    )


def main():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    distributions = [
        "Pareto",
        "Student-t",
        "Mixture",
        "Volatility-Clustered",
    ]

    tables_dir = (
        ROOT / "tables"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "M1.5b - CONFORMAL TAIL CALIBRATION LABORATORY"
    )
    print("=" * 70)

    print(
        "Generating calibration scores...",
        flush=True,
    )

    calibration_scores = (
        build_calibration_scores(
            distributions,
            rng,
        )
    )

    radius = conformal_radius(
        calibration_scores,
        MIS_COVERAGE,
    )

    print()
    print(
        f"Calibration scores: "
        f"{len(calibration_scores)}"
    )

    print(
        f"Conformal radius: "
        f"{radius:.6f}"
    )

    print(
        f"Target coverage: "
        f"{1.0 - MIS_COVERAGE:.4f}"
    )

    print()
    print(
        "Generating independent test set...",
        flush=True,
    )

    testing = build_test_results(
        distributions,
        radius,
        rng,
    )

    overall_coverage = (
        testing["covered"].mean()
    )

    overall_width = (
        testing[
            "relative_width"
        ].mean()
    )

    summary = (
        testing.groupby(
            "distribution"
        )
        .agg(
            coverage=(
                "covered",
                "mean",
            ),
            mean_relative_width=(
                "relative_width",
                "mean",
            ),
            observations=(
                "covered",
                "size",
            ),
        )
        .reset_index()
    )

    coverage_gap = (
        overall_coverage
        - (
            1.0 - MIS_COVERAGE
        )
    )

    test_path = (
        tables_dir
        / "conformal_tail_calibration_test.csv"
    )

    summary_path = (
        tables_dir
        / "conformal_tail_calibration_summary.csv"
    )

    testing.to_csv(
        test_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        f"Overall test coverage: "
        f"{overall_coverage:.6f}"
    )

    print(
        f"Coverage gap: "
        f"{coverage_gap:+.6f}"
    )

    print(
        f"Mean relative interval width: "
        f"{overall_width:.6f}"
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Test results saved to: "
        f"{test_path}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
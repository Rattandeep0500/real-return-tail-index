from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.adaptive_conformal import (
    fit_adaptive_conformal,
    build_adaptive_certificate,
)


RANDOM_SEED = 2026

SAMPLE_SIZE = 120

CALIBRATION_REPLICATIONS = 500
TEST_REPLICATIONS = 500

BOOTSTRAPS = 30

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


def estimate_features(
    sample,
    rng,
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

    alpha_values = []
    bootstrap_cvs = []

    for k in k_values:
        alpha_hat = hill_estimator(
            sample,
            int(k),
        )

        bootstrap_values = []

        for _ in range(
            BOOTSTRAPS
        ):
            bootstrap_sample = rng.choice(
                sample,
                size=len(sample),
                replace=True,
            )

            bootstrap_sample = np.sort(
                bootstrap_sample
            )[::-1]

            try:
                value = hill_estimator(
                    bootstrap_sample,
                    int(k),
                )

                if np.isfinite(value):
                    bootstrap_values.append(
                        value
                    )

            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
            ):
                continue

        if len(
            bootstrap_values
        ) >= 2:
            bootstrap_std = np.std(
                bootstrap_values,
                ddof=1,
            )

            bootstrap_cv = (
                bootstrap_std
                / max(
                    abs(alpha_hat),
                    1e-12,
                )
            )
        else:
            bootstrap_cv = np.nan

        alpha_values.append(
            alpha_hat
        )

        bootstrap_cvs.append(
            bootstrap_cv
        )

    return pd.DataFrame(
        {
            "k": k_values,
            "alpha_hat": np.asarray(
                alpha_values,
                dtype=float,
            ),
            "bootstrap_cv": np.asarray(
                bootstrap_cvs,
                dtype=float,
            ),
            "sample_size": len(sample),
        }
    )


def build_calibration_dataset(
    distributions,
    rng,
):
    frames = []

    total = (
        len(distributions)
        * CALIBRATION_REPLICATIONS
    )

    completed = 0

    for distribution in distributions:
        for replication in range(
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

            features = estimate_features(
                sample,
                rng,
            )

            features[
                "distribution"
            ] = distribution

            features[
                "replication"
            ] = replication

            features[
                "true_alpha"
            ] = alpha

            frames.append(
                features
            )

            completed += 1

            if completed % 50 == 0:
                print(
                    f"Calibration progress: "
                    f"{completed}/{total}",
                    flush=True,
                )

    if not frames:
        raise ValueError(
            "No calibration data generated."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_test_dataset(
    distributions,
    rng,
):
    frames = []

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

            features = estimate_features(
                sample,
                rng,
            )

            features[
                "distribution"
            ] = distribution

            features[
                "replication"
            ] = replication

            features[
                "true_alpha"
            ] = alpha

            frames.append(
                features
            )

            completed += 1

            if completed % 50 == 0:
                print(
                    f"Test progress: "
                    f"{completed}/{total}",
                    flush=True,
                )

    if not frames:
        raise ValueError(
            "No test data generated."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def evaluate_method(
    test_data,
    model,
    label,
):
    certificate = build_adaptive_certificate(
        test_data[
            "alpha_hat"
        ].to_numpy(
            dtype=float
        ),
        test_data[
            "bootstrap_cv"
        ].to_numpy(
            dtype=float
        ),
        test_data[
            "k"
        ].to_numpy(
            dtype=float
        ),
        test_data[
            "sample_size"
        ].to_numpy(
            dtype=float
        ),
        model,
        max_relative_width=np.inf,
    )

    coverage = (
        (
            test_data[
                "true_alpha"
            ].to_numpy()
            >= certificate[
                "conformal_lower"
            ].to_numpy()
        )
        & (
            test_data[
                "true_alpha"
            ].to_numpy()
            <= certificate[
                "conformal_upper"
            ].to_numpy()
        )
    )

    result = test_data.copy()

    result[
        "conformal_radius"
    ] = certificate[
        "conformal_radius"
    ].to_numpy()

    result[
        "conformal_lower"
    ] = certificate[
        "conformal_lower"
    ].to_numpy()

    result[
        "conformal_upper"
    ] = certificate[
        "conformal_upper"
    ].to_numpy()

    result[
        "relative_interval_width"
    ] = certificate[
        "relative_interval_width"
    ].to_numpy()

    result[
        "radius_source"
    ] = certificate[
        "radius_source"
    ].to_numpy()

    result[
        "covered"
    ] = coverage

    summary = (
        result.groupby(
            "distribution"
        )
        .agg(
            coverage=(
                "covered",
                "mean",
            ),
            mean_relative_width=(
                "relative_interval_width",
                "mean",
            ),
            mean_radius=(
                "conformal_radius",
                "mean",
            ),
            n=(
                "covered",
                "size",
            ),
        )
        .reset_index()
    )

    summary[
        "method"
    ] = label

    return (
        result,
        summary,
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

    print("=" * 70)
    print(
        "M1.7b - ADAPTIVE CONFORMAL VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "Generating calibration dataset...",
        flush=True,
    )

    calibration_data = (
        build_calibration_dataset(
            distributions,
            rng,
        )
    )

    print()
    print(
        f"Calibration observations: "
        f"{len(calibration_data)}"
    )

    print()
    print(
        "Fitting adaptive conformal model...",
        flush=True,
    )

    model = fit_adaptive_conformal(
        calibration_data,
        miscoverage=MIS_COVERAGE,
        alpha_bins=4,
        stability_bins=3,
        k_bins=4,
        min_group_size=40,
    )

    print(
        f"3D groups: "
        f"{len(model['radii_3d'])}"
    )

    print(
        f"2D groups: "
        f"{len(model['radii_2d'])}"
    )

    print(
        f"Global radius: "
        f"{model['global_radius']:.6f}"
    )

    print()
    print(
        "Generating independent test dataset...",
        flush=True,
    )

    test_data = build_test_dataset(
        distributions,
        rng,
    )

    results, summary = evaluate_method(
        test_data,
        model,
        "Adaptive",
    )

    overall_coverage = results[
        "covered"
    ].mean()

    overall_width = results[
        "relative_interval_width"
    ].mean()

    source_counts = results[
        "radius_source"
    ].value_counts()

    tables_dir = ROOT / "tables"

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        tables_dir
        / "adaptive_conformal_test.csv"
    )

    summary_path = (
        tables_dir
        / "adaptive_conformal_summary.csv"
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
        f"Overall coverage: "
        f"{overall_coverage:.6f}"
    )

    print(
        f"Target coverage: "
        f"{1.0 - MIS_COVERAGE:.6f}"
    )

    print(
        f"Mean relative width: "
        f"{overall_width:.6f}"
    )

    print()
    print(
        "Radius source counts:"
    )

    print(
        source_counts.to_string()
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
        f"{results_path}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
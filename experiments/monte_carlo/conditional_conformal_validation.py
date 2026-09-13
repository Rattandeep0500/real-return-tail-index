from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.conditional_conformal import (
    fit_conditional_conformal,
    conditional_certificate,
)


RANDOM_SEED = 2026
SAMPLE_SIZE = 120

CALIBRATION_REPLICATIONS = 500
TEST_REPLICATIONS = 500

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

    bootstrap_cv = []

    for k in k_values:
        alpha_hat = hill_estimator(
            sample,
            int(k),
        )

        estimates.append(
            alpha_hat
        )

        bootstrap_values = []

        for _ in range(30):
            bootstrap_sample = np.random.default_rng().choice(
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
            ):
                continue

        if len(bootstrap_values) >= 2:
            std = np.std(
                bootstrap_values,
                ddof=1,
            )

            cv = std / max(
                abs(alpha_hat),
                1e-12,
            )
        else:
            cv = np.nan

        bootstrap_cv.append(
            cv
        )

    return (
        k_values,
        np.asarray(
            estimates,
            dtype=float,
        ),
        np.asarray(
            bootstrap_cv,
            dtype=float,
        ),
    )


def create_calibration_frame(
    distributions,
    rng,
):
    rows = []

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

            k_values, estimates, cvs = (
                estimate_features(
                    sample
                )
            )

            for i in range(
                len(k_values)
            ):
                if not np.isfinite(
                    estimates[i]
                ):
                    continue

                if not np.isfinite(
                    cvs[i]
                ):
                    continue

                rows.append(
                    {
                        "distribution": distribution,
                        "true_alpha": alpha,
                        "alpha_hat": estimates[i],
                        "bootstrap_cv": cvs[i],
                        "k": int(
                            k_values[i]
                        ),
                    }
                )

            completed += 1

            if completed % 50 == 0:
                print(
                    f"Calibration progress: "
                    f"{completed}/{total}",
                    flush=True,
                )

    return pd.DataFrame(
        rows
    )


def create_test_frame(
    distributions,
    rng,
):
    rows = []

    total = (
        len(distributions)
        * TEST_REPLICATIONS
    )

    completed = 0

    for distribution in distributions:
        for _ in range(
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

            k_values, estimates, cvs = (
                estimate_features(
                    sample
                )
            )

            for i in range(
                len(k_values)
            ):
                if not np.isfinite(
                    estimates[i]
                ):
                    continue

                if not np.isfinite(
                    cvs[i]
                ):
                    continue

                rows.append(
                    {
                        "distribution": distribution,
                        "true_alpha": alpha,
                        "alpha_hat": estimates[i],
                        "bootstrap_cv": cvs[i],
                        "k": int(
                            k_values[i]
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

    return pd.DataFrame(
        rows
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
        "M1.6b - CONDITIONAL CONFORMAL VALIDATION"
    )
    print("=" * 70)

    print(
        "Generating calibration dataset..."
    )

    calibration_data = (
        create_calibration_frame(
            distributions,
            rng,
        )
    )

    print(
        f"Calibration observations: "
        f"{len(calibration_data)}"
    )

    print(
        "Fitting conditional conformal model..."
    )

    model = fit_conditional_conformal(
        calibration_data,
        miscoverage=0.10,
        alpha_bins=4,
        stability_bins=3,
        min_group_size=30,
    )

    print(
        f"Conditional groups: "
        f"{len(model['group_radii'])}"
    )

    print(
        "Generating independent test dataset..."
    )

    test_data = create_test_frame(
        distributions,
        rng,
    )

    certificate = conditional_certificate(
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
        model,
        max_relative_width=np.inf,
    )

    test_data[
        "conformal_radius"
    ] = certificate[
        "conformal_radius"
    ]

    test_data[
        "conformal_lower"
    ] = certificate[
        "conformal_lower"
    ]

    test_data[
        "conformal_upper"
    ] = certificate[
        "conformal_upper"
    ]

    test_data[
        "relative_interval_width"
    ] = certificate[
        "relative_interval_width"
    ]

    test_data[
        "covered"
    ] = (
        test_data[
            "true_alpha"
        ].to_numpy()
        >= test_data[
            "conformal_lower"
        ].to_numpy()
    ) & (
        test_data[
            "true_alpha"
        ].to_numpy()
        <= test_data[
            "conformal_upper"
        ].to_numpy()
    )

    overall_coverage = (
        test_data[
            "covered"
        ].mean()
    )

    overall_width = (
        test_data[
            "relative_interval_width"
        ].mean()
    )

    summary = (
        test_data.groupby(
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
            observations=(
                "covered",
                "size",
            ),
        )
        .reset_index()
    )

    tables_dir = (
        ROOT / "tables"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_path = (
        tables_dir
        / "conditional_conformal_calibration.csv"
    )

    test_path = (
        tables_dir
        / "conditional_conformal_test.csv"
    )

    summary_path = (
        tables_dir
        / "conditional_conformal_summary.csv"
    )

    calibration_data.to_csv(
        calibration_path,
        index=False,
    )

    test_data.to_csv(
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
        f"Target coverage: "
        f"{0.90:.6f}"
    )

    print(
        f"Mean relative width: "
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
        f"Calibration saved to: "
        f"{calibration_path}"
    )

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
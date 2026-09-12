from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.calibration import (
    build_features,
    create_quality_labels,
    fit_reliability_model,
    predict_reliability,
)


TRUE_ALPHAS = [
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
]

SAMPLE_SIZE = 10000
TRAIN_REPLICATIONS = 120
TEST_REPLICATIONS = 80
BOOTSTRAPS = 60
K_VALUES = np.arange(
    50,
    1001,
    25,
)

WINDOW_SIZE = 15
TOLERANCE = 0.10
RANDOM_SEED = 42


def simulate_pareto(
    alpha,
    n,
    rng,
):
    u = rng.uniform(
        size=n
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
    pareto = simulate_pareto(
        alpha,
        n,
        rng,
    )

    gaussian = np.abs(
        rng.normal(
            1.0,
            0.5,
            size=n,
        )
    )

    mask = rng.uniform(
        size=n
    ) < 0.8

    return np.where(
        mask,
        pareto,
        gaussian,
    )


def simulate_regime_mixture(
    alpha,
    n,
    rng,
):
    n1 = n // 2
    n2 = n - n1

    regime_1 = simulate_pareto(
        alpha,
        n1,
        rng,
    )

    regime_2 = simulate_pareto(
        alpha + 2.0,
        n2,
        rng,
    )

    return np.concatenate(
        [
            regime_1,
            regime_2,
        ]
    )


def simulate_distribution(
    name,
    alpha,
    n,
    rng,
):
    if name == "Pareto":
        return simulate_pareto(
            alpha,
            n,
            rng,
        )

    if name == "Student-t":
        return simulate_student_t(
            alpha,
            n,
            rng,
        )

    if name == "Pareto-Gaussian":
        return simulate_mixture(
            alpha,
            n,
            rng,
        )

    if name == "Regime-Mixture":
        return simulate_regime_mixture(
            alpha,
            n,
            rng,
        )

    raise ValueError(
        f"Unknown distribution: {name}"
    )


def evaluate_sample(
    sample,
    rng,
):
    alpha_values = []
    bootstrap_stds = []

    for k in K_VALUES:
        alpha_hat = hill_estimator(
            sample,
            int(k),
        )

        bootstrap_estimates = []

        for _ in range(BOOTSTRAPS):
            bootstrap_sample = rng.choice(
                sample,
                size=len(sample),
                replace=True,
            )

            bootstrap_estimates.append(
                hill_estimator(
                    bootstrap_sample,
                    int(k),
                )
            )

        alpha_values.append(
            alpha_hat
        )

        bootstrap_stds.append(
            np.std(
                bootstrap_estimates,
                ddof=1,
            )
        )

    return np.asarray(
        alpha_values
    ), np.asarray(
        bootstrap_stds
    )


def build_dataset(
    rng,
    replications,
):
    records = []

    distributions = [
        "Pareto",
        "Student-t",
        "Pareto-Gaussian",
        "Regime-Mixture",
    ]

    for distribution in distributions:
        for alpha in TRUE_ALPHAS:
            for replication in range(
                replications
            ):
                sample = simulate_distribution(
                    distribution,
                    alpha,
                    SAMPLE_SIZE,
                    rng,
                )

                alpha_values, bootstrap_stds = (
                    evaluate_sample(
                        sample,
                        rng,
                    )
                )

                features = build_features(
                    K_VALUES,
                    alpha_values,
                    bootstrap_stds,
                    SAMPLE_SIZE,
                    WINDOW_SIZE,
                )

                labels = create_quality_labels(
                    alpha_values,
                    alpha,
                    TOLERANCE,
                )

                for i, k in enumerate(
                    K_VALUES
                ):
                    record = {
                        "distribution": distribution,
                        "true_alpha": alpha,
                        "replication": replication,
                        "k": int(k),
                        "target": int(
                            labels[i]
                        ),
                    }

                    for name in features.columns:
                        record[name] = features.iloc[
                            i
                        ][name]

                    records.append(
                        record
                    )

    return pd.DataFrame(
        records
    )


def main():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print("=" * 70)
    print(
        "M0.10 — DATA-DRIVEN "
        "RELIABILITY CALIBRATION"
    )
    print("=" * 70)

    print("Generating training simulations...")

    training = build_dataset(
        rng,
        TRAIN_REPLICATIONS,
    )

    print(
        f"Training observations: "
        f"{len(training)}"
    )

    feature_columns = [
        "alpha_hat",
        "bootstrap_std",
        "bootstrap_cv",
        "local_cv",
        "local_slope",
        "local_curvature",
        "k_fraction",
    ]

    model, training_index = (
        fit_reliability_model(
            training[
                feature_columns
            ],
            training["target"],
        )
    )

    print("Training calibration model...")

    print("Generating independent test simulations...")

    testing = build_dataset(
        rng,
        TEST_REPLICATIONS,
    )

    test_features = testing[
        feature_columns
    ]

    predicted = predict_reliability(
        model,
        test_features,
    )

    testing[
        "reliability_probability"
    ] = predicted[
        "reliability_probability"
    ]

    valid = testing.dropna(
        subset=[
            "reliability_probability",
            "target",
        ]
    ).copy()

    brier = brier_score_loss(
        valid["target"],
        valid[
            "reliability_probability"
        ],
    )

    auc = roc_auc_score(
        valid["target"],
        valid[
            "reliability_probability"
        ],
    )

    calibration_bins = pd.qcut(
        valid[
            "reliability_probability"
        ],
        q=10,
        duplicates="drop",
    )

    calibration = (
        valid.groupby(
            calibration_bins,
            observed=True,
        )
        .agg(
            mean_predicted=(
                "reliability_probability",
                "mean",
            ),
            observed_accuracy=(
                "target",
                "mean",
            ),
            count=(
                "target",
                "size",
            ),
        )
        .reset_index()
    )

    test_results_path = (
        ROOT
        / "tables"
        / "data_driven_reliability_test.csv"
    )

    calibration_path = (
        ROOT
        / "tables"
        / "reliability_calibration_curve.csv"
    )

    testing.to_csv(
        test_results_path,
        index=False,
    )

    calibration.to_csv(
        calibration_path,
        index=False,
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    plt.plot(
        calibration[
            "mean_predicted"
        ],
        calibration[
            "observed_accuracy"
        ],
        marker="o",
        linewidth=1.5,
        label="Model",
    )

    plt.xlabel(
        "Predicted reliability probability"
    )

    plt.ylabel(
        "Observed accuracy"
    )

    plt.title(
        "Reliability Calibration"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    figure_path = (
        ROOT
        / "figures"
        / "reliability_calibration_curve.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    summary = (
        testing.groupby(
            "distribution"
        )
        .agg(
            mean_reliability=(
                "reliability_probability",
                "mean",
            ),
            actual_accuracy=(
                "target",
                "mean",
            ),
            n=(
                "target",
                "size",
            ),
        )
        .reset_index()
    )

    summary_path = (
        ROOT
        / "tables"
        / "data_driven_reliability_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        f"Brier score             : "
        f"{brier:.6f}"
    )

    print(
        f"ROC AUC                 : "
        f"{auc:.6f}"
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Test results saved to   : "
        f"{test_results_path}"
    )

    print(
        f"Calibration saved to    : "
        f"{calibration_path}"
    )

    print(
        f"Summary saved to        : "
        f"{summary_path}"
    )

    print(
        f"Figure saved to         : "
        f"{figure_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
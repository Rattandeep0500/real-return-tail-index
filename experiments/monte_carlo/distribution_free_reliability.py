from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.calibration import (
    build_features,
    create_quality_labels,
    fit_reliability_model,
    predict_reliability,
)


SAMPLE_SIZES = [2000, 5000, 10000, 25000]
ALPHAS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

TRAIN_REPLICATIONS = 100
TEST_REPLICATIONS = 60
BOOTSTRAPS = 40

K_VALUES = np.arange(40, 1001, 40)

WINDOW_SIZE = 15
TOLERANCE = 0.10
RANDOM_SEED = 42


def simulate_pareto(alpha, n, rng):
    u = rng.uniform(size=n)
    return u ** (-1.0 / alpha)


def simulate_student_t(df, n, rng):
    return np.abs(
        rng.standard_t(
            df=df,
            size=n,
        )
    )


def simulate_log_normal(n, rng):
    return np.exp(
        rng.normal(
            0.0,
            1.0,
            size=n,
        )
    )


def simulate_truncated_pareto(alpha, n, rng, cutoff):
    sample = []

    while len(sample) < n:
        batch = simulate_pareto(
            alpha,
            n,
            rng,
        )

        batch = batch[
            batch <= cutoff
        ]

        sample.extend(batch.tolist())

    return np.asarray(
        sample[:n]
    )


def simulate_mixture(alpha, n, rng):
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


def simulate_regime_switch(alpha, n, rng):
    split = rng.integers(
        int(0.30 * n),
        int(0.70 * n),
    )

    alpha_1 = rng.uniform(
        max(1.2, alpha - 0.8),
        alpha + 0.2,
    )

    alpha_2 = rng.uniform(
        max(1.2, alpha - 0.2),
        alpha + 1.2,
    )

    regime_1 = simulate_pareto(
        alpha_1,
        split,
        rng,
    )

    regime_2 = simulate_pareto(
        alpha_2,
        n - split,
        rng,
    )

    return np.concatenate(
        [
            regime_1,
            regime_2,
        ]
    )


def simulate_volatility_clustered(alpha, n, rng):
    innovations = simulate_pareto(
        alpha,
        n,
        rng,
    )

    sigma = np.empty(n)
    sigma[0] = 1.0

    for t in range(1, n):
        sigma[t] = (
            0.94 * sigma[t - 1]
            + 0.06 * innovations[t - 1]
        )

    result = (
        innovations
        * sigma
        / np.mean(sigma)
    )

    return result


def generate_sample(
    distribution,
    alpha,
    n,
    rng,
):
    if distribution == "Pareto":
        return simulate_pareto(
            alpha,
            n,
            rng,
        )

    if distribution == "Student-t":
        return simulate_student_t(
            alpha,
            n,
            rng,
        )

    if distribution == "Lognormal":
        return simulate_log_normal(
            n,
            rng,
        )

    if distribution == "Truncated-Pareto":
        return simulate_truncated_pareto(
            alpha,
            n,
            rng,
            cutoff=100.0,
        )

    if distribution == "Mixture":
        return simulate_mixture(
            alpha,
            n,
            rng,
        )

    if distribution == "Regime-Switch":
        return simulate_regime_switch(
            alpha,
            n,
            rng,
        )

    if distribution == "Volatility-Clustered":
        return simulate_volatility_clustered(
            alpha,
            n,
            rng,
        )

    raise ValueError(
        f"Unknown distribution: {distribution}"
    )


def adaptive_k_values(n):
    return K_VALUES[
        K_VALUES < n
    ]


def evaluate_sample(
    sample,
    rng,
):
    k_values = adaptive_k_values(
        len(sample)
    )

    alpha_values = []
    bootstrap_stds = []

    for k in k_values:
        alpha_hat = hill_estimator(
            sample,
            int(k),
        )

        estimates = []

        for _ in range(
            BOOTSTRAPS
        ):
            bootstrap_sample = rng.choice(
                sample,
                size=len(sample),
                replace=True,
            )

            try:
                estimates.append(
                    hill_estimator(
                        bootstrap_sample,
                        int(k),
                    )
                )
            except (
                ValueError,
                RuntimeError,
            ):
                continue

        alpha_values.append(
            alpha_hat
        )

        bootstrap_stds.append(
            np.std(
                estimates,
                ddof=1,
            )
        )

    features = build_features(
        k_values,
        np.asarray(
            alpha_values
        ),
        np.asarray(
            bootstrap_stds
        ),
        len(sample),
        WINDOW_SIZE,
    )

    return (
        k_values,
        features,
    )


def build_dataset(
    distributions,
    replications,
    rng,
):
    records = []

    for distribution in distributions:
        for alpha in ALPHAS:
            for replication in range(
                replications
            ):
                n = int(
                    rng.choice(
                        SAMPLE_SIZES
                    )
                )

                sample = generate_sample(
                    distribution,
                    alpha,
                    n,
                    rng,
                )

                k_values, features = (
                    evaluate_sample(
                        sample,
                        rng,
                    )
                )

                labels = create_quality_labels(
                    features[
                        "alpha_hat"
                    ].values,
                    alpha,
                    TOLERANCE,
                )

                for i, k in enumerate(
                    k_values
                ):
                    row = {
                        "distribution": distribution,
                        "true_alpha": alpha,
                        "sample_size": n,
                        "replication": replication,
                        "k": int(k),
                        "target": int(
                            labels[i]
                        ),
                    }

                    for column in features.columns:
                        row[column] = features.iloc[
                            i
                        ][column]

                    records.append(row)

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
        "Lognormal",
        "Truncated-Pareto",
        "Mixture",
        "Regime-Switch",
        "Volatility-Clustered",
    ]

    training_distributions = distributions
    testing_distributions = distributions

    print("=" * 70)
    print(
        "M0.11 — DISTRIBUTION-FREE "
        "RELIABILITY VALIDATION"
    )
    print("=" * 70)

    print("Generating training universe...")

    training = build_dataset(
        training_distributions,
        TRAIN_REPLICATIONS,
        rng,
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

    model, _ = fit_reliability_model(
        training[
            feature_columns
        ],
        training["target"],
    )

    print(
        "Generating independent "
        "test universe..."
    )

    testing = build_dataset(
        testing_distributions,
        TEST_REPLICATIONS,
        rng,
    )

    testing = testing.copy()

    predictions = predict_reliability(
        model,
        testing[
            feature_columns
        ],
    )

    testing[
        "reliability_probability"
    ] = predictions[
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

    threshold = 0.80

    valid[
        "predicted_good"
    ] = (
        valid[
            "reliability_probability"
        ] >= threshold
    ).astype(int)

    precision = (
        valid.loc[
            valid["predicted_good"] == 1,
            "target",
        ].mean()
    )

    summary = (
        valid.groupby(
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
            mean_alpha=(
                "alpha_hat",
                "mean",
            ),
            mean_bootstrap_std=(
                "bootstrap_std",
                "mean",
            ),
            n=(
                "target",
                "size",
            ),
        )
        .reset_index()
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

    testing_path = (
        tables_dir
        / "distribution_free_reliability_test.csv"
    )

    summary_path = (
        tables_dir
        / "distribution_free_reliability_summary.csv"
    )

    testing.to_csv(
        testing_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    plt.figure(
        figsize=(10, 6)
    )

    for distribution in distributions:
        subset = valid[
            valid["distribution"]
            == distribution
        ]

        plt.scatter(
            subset[
                "reliability_probability"
            ],
            np.abs(
                subset["alpha_hat"]
                - subset["true_alpha"]
            )
            / subset["true_alpha"],
            alpha=0.20,
            label=distribution,
        )

    plt.axhline(
        TOLERANCE,
        linestyle="--",
        linewidth=1.5,
        label="10% error threshold",
    )

    plt.xlabel(
        "Predicted reliability probability"
    )

    plt.ylabel(
        "Absolute relative tail-index error"
    )

    plt.title(
        "Distribution-Free Reliability Validation"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    figure_path = (
        figures_dir
        / "distribution_free_reliability.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        f"Brier score             : "
        f"{brier:.6f}"
    )

    print(
        f"ROC AUC                 : "
        f"{auc:.6f}"
    )

    print(
        f"Precision at 0.80      : "
        f"{precision:.6f}"
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Test results saved to  : "
        f"{testing_path}"
    )

    print(
        f"Summary saved to       : "
        f"{summary_path}"
    )

    print(
        f"Figure saved to        : "
        f"{figure_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
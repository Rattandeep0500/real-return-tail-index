from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.calibration import (
    build_features,
    fit_reliability_model,
    predict_reliability,
)
from src.tail.model_validity import compute_model_validity


RANDOM_SEED = 314159

WINDOW_SIZE = 120

ALPHAS = [
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
]

BOOTSTRAPS = 100

K_VALUES = np.arange(
    10,
    81,
    5,
)

TRAIN_REPLICATIONS = 300

TEST_REPLICATIONS = 150

TOLERANCE = 0.10


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
    tail_count = rng.binomial(
        n,
        0.8,
    )

    body_count = n - tail_count

    tail = simulate_pareto(
        alpha,
        tail_count,
        rng,
    )

    body = np.abs(
        rng.normal(
            1.0,
            0.5,
            size=body_count,
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

    mean_sigma = np.mean(
        sigma
    )

    return (
        innovations
        * sigma
        / mean_sigma
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
        max(1.2, alpha - 0.7),
        alpha + 0.2,
    )

    alpha_2 = rng.uniform(
        max(1.2, alpha - 0.2),
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

    effective_alpha = min(
        alpha_1,
        alpha_2,
    )

    return (
        np.concatenate(
            [
                first,
                second,
            ]
        ),
        effective_alpha,
    )


def generate_sample(
    distribution,
    alpha,
    rng,
):
    if distribution == "Pareto":
        return (
            simulate_pareto(
                alpha,
                WINDOW_SIZE,
                rng,
            ),
            alpha,
            True,
        )

    if distribution == "Student-t":
        return (
            simulate_student_t(
                alpha,
                WINDOW_SIZE,
                rng,
            ),
            alpha,
            True,
        )

    if distribution == "Mixture":
        return (
            simulate_mixture(
                alpha,
                WINDOW_SIZE,
                rng,
            ),
            alpha,
            True,
        )

    if distribution == "Volatility-Clustered":
        return (
            simulate_volatility_clustered(
                alpha,
                WINDOW_SIZE,
                rng,
            ),
            alpha,
            True,
        )

    if distribution == "Regime-Switch":
        sample, effective_alpha = (
            simulate_regime_switch(
                alpha,
                WINDOW_SIZE,
                rng,
            )
        )

        return (
            sample,
            effective_alpha,
            True,
        )

    raise ValueError(
        f"Unknown distribution: {distribution}"
    )


def calculate_features(
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
    bootstrap_stds = []

    for k in k_values:
        alpha_hat = hill_estimator(
            sample,
            int(k),
        )

        bootstrap_estimates = []

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
                    bootstrap_estimates.append(
                        value
                    )

            except (
                ValueError,
                RuntimeError,
            ):
                continue

        alpha_values.append(
            alpha_hat
        )

        if len(
            bootstrap_estimates
        ) >= 2:
            bootstrap_stds.append(
                np.std(
                    bootstrap_estimates,
                    ddof=1,
                )
            )
        else:
            bootstrap_stds.append(
                np.nan
            )

    features = build_features(
        k_values,
        np.asarray(alpha_values),
        np.asarray(bootstrap_stds),
        len(sample),
        window_size=15,
    )

    features[
        "k"
    ] = k_values

    return features


def create_labels(
    alpha_values,
    target_alpha,
):
    relative_error = (
        np.abs(
            alpha_values
            - target_alpha
        )
        / target_alpha
    )

    return (
        np.isfinite(
            relative_error
        )
        & (
            relative_error
            <= TOLERANCE
        )
    ).astype(int)


def build_dataset(
    distributions,
    replications,
    rng,
):
    records = []

    total = (
        len(distributions)
        * len(ALPHAS)
        * replications
    )

    completed = 0

    for distribution in distributions:
        for alpha in ALPHAS:
            for replication in range(
                replications
            ):
                sample, target_alpha, valid_tail = (
                    generate_sample(
                        distribution,
                        alpha,
                        rng,
                    )
                )

                features = calculate_features(
                    sample,
                    rng,
                )

                labels = create_labels(
                    features[
                        "alpha_hat"
                    ].to_numpy(
                        dtype=float
                    ),
                    target_alpha,
                )

                for i in range(
                    len(features)
                ):
                    row = {
                        "distribution": distribution,
                        "simulation_alpha": alpha,
                        "target_alpha": target_alpha,
                        "tail_valid": int(
                            valid_tail
                        ),
                        "replication": replication,
                        "k": int(
                            features.iloc[
                                i
                            ]["k"]
                        ),
                        "target": int(
                            labels[i]
                        ),
                    }

                    for column in features.columns:
                        row[column] = features.iloc[
                            i
                        ][column]

                    records.append(
                        row
                    )

                completed += 1

                if completed % 25 == 0:
                    print(
                        f"Progress: "
                        f"{completed}/{total}",
                        flush=True,
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
        "Regime-Switch",
    ]

    feature_columns = [
        "alpha_hat",
        "bootstrap_std",
        "bootstrap_cv",
        "local_cv",
        "local_slope",
        "local_curvature",
        "k_fraction",
    ]

    print("=" * 70)
    print(
        "M1.4 - DEPLOYMENT-CALIBRATED "
        "TAIL RELIABILITY"
    )
    print("=" * 70)

    print(
        "Generating deployment-scale training universe...",
        flush=True,
    )

    training = build_dataset(
        distributions,
        TRAIN_REPLICATIONS,
        rng,
    )

    print(
        f"Training observations: "
        f"{len(training)}",
        flush=True,
    )

    model, _ = fit_reliability_model(
        training[
            feature_columns
        ],
        training["target"],
    )

    model_path = (
        ROOT
        / "data"
        / "metadata"
        / "deployment_tail_reliability_model.joblib"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Frozen deployment model: "
        f"{model_path}",
        flush=True,
    )

    print(
        "Generating independent deployment-scale test universe...",
        flush=True,
    )

    testing = build_dataset(
        distributions,
        TEST_REPLICATIONS,
        rng,
    )

    testing[
        "reliability_probability"
    ] = predict_reliability(
        model,
        testing[
            feature_columns
        ],
    )[
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

    summary = (
        valid.groupby(
            "distribution"
        )
        .agg(
            predicted=(
                "reliability_probability",
                "mean",
            ),
            observed=(
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

    tables_dir = (
        ROOT / "tables"
    )

    figures_dir = (
        ROOT / "figures"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_path = (
        tables_dir
        / "deployment_reliability_test.csv"
    )

    summary_path = (
        tables_dir
        / "deployment_reliability_summary.csv"
    )

    testing.to_csv(
        test_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    calibration_bins = pd.qcut(
        valid[
            "reliability_probability"
        ],
        q=10,
        duplicates="drop",
    )

    calibration = (
        valid.assign(
            reliability_bin=calibration_bins
        )
        .groupby(
            "reliability_bin",
            observed=True,
        )
        .agg(
            predicted=(
                "reliability_probability",
                "mean",
            ),
            observed=(
                "target",
                "mean",
            ),
            observations=(
                "target",
                "size",
            ),
        )
        .reset_index()
    )

    calibration_path = (
        tables_dir
        / "deployment_reliability_calibration.csv"
    )

    calibration[
        [
            "predicted",
            "observed",
            "observations",
        ]
    ].to_csv(
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
        calibration["predicted"],
        calibration["observed"],
        marker="o",
        linewidth=1.5,
        label="Deployment model",
    )

    plt.xlabel(
        "Predicted reliability"
    )

    plt.ylabel(
        "Observed reliability"
    )

    plt.title(
        "Deployment-Calibrated Tail Reliability"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    figure_path = (
        figures_dir
        / "deployment_tail_reliability_calibration.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        f"Brier score             : {brier:.6f}"
    )

    print(
        f"ROC AUC                 : {auc:.6f}"
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Test results saved to   : {test_path}"
    )

    print(
        f"Summary saved to        : {summary_path}"
    )

    print(
        f"Calibration saved to    : {calibration_path}"
    )

    print(
        f"Figure saved to         : {figure_path}"
    )

    print(
        f"Frozen model            : {model_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
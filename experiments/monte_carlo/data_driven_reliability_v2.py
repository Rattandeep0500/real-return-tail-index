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

TRUE_ALPHAS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SAMPLE_SIZES = [2000, 5000, 10000, 25000]

TRAIN_REPLICATIONS = 40
TEST_REPLICATIONS = 25
BOOTSTRAPS = 15

K_VALUES = np.arange(40, 1001, 40)

WINDOW_SIZE = 15
TOLERANCE = 0.10
RANDOM_SEED = 123


def simulate_pareto(alpha, n, rng):
    u = rng.uniform(0.0, 1.0, n)
    return u ** (-1.0 / alpha)


def simulate_student_t(alpha, n, rng):
    return np.abs(rng.standard_t(df=alpha, size=n))


def simulate_lognormal(n, rng):
    return np.exp(rng.normal(0.0, 1.0, size=n))


def simulate_truncated_pareto(alpha, n, rng, cutoff=100.0):
    values = []

    while len(values) < n:
        batch = simulate_pareto(alpha, n, rng)
        batch = batch[batch <= cutoff]
        values.extend(batch.tolist())

    return np.asarray(values[:n], dtype=float)


def simulate_mixture(alpha, n, rng):
    n_tail = rng.binomial(n, 0.80)
    n_body = n - n_tail

    tail = simulate_pareto(alpha, n_tail, rng)

    body = np.abs(
        rng.normal(
            1.0,
            0.5,
            size=n_body,
        )
    )

    return np.concatenate([tail, body])


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
            [first, second]
        ),
        effective_alpha,
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

    volatility = np.empty(n)
    volatility[0] = 1.0

    for t in range(1, n):
        volatility[t] = (
            0.94 * volatility[t - 1]
            + 0.06 * innovations[t - 1]
        )

    mean_volatility = volatility.mean()

    if not np.isfinite(mean_volatility):
        mean_volatility = 1.0

    if mean_volatility <= 0:
        mean_volatility = 1.0

    return (
        innovations
        * volatility
        / mean_volatility
    )


def generate_sample(
    distribution,
    alpha,
    n,
    rng,
):
    if distribution == "Pareto":
        return (
            simulate_pareto(
                alpha,
                n,
                rng,
            ),
            alpha,
            True,
        )

    if distribution == "Student-t":
        return (
            simulate_student_t(
                alpha,
                n,
                rng,
            ),
            alpha,
            True,
        )

    if distribution == "Mixture":
        return (
            simulate_mixture(
                alpha,
                n,
                rng,
            ),
            alpha,
            True,
        )

    if distribution == "Volatility-Clustered":
        return (
            simulate_volatility_clustered(
                alpha,
                n,
                rng,
            ),
            alpha,
            True,
        )

    if distribution == "Truncated-Pareto":
        return (
            simulate_truncated_pareto(
                alpha,
                n,
                rng,
            ),
            np.nan,
            False,
        )

    if distribution == "Lognormal":
        return (
            simulate_lognormal(
                n,
                rng,
            ),
            np.nan,
            False,
        )

    if distribution == "Regime-Switch":
        sample, effective_alpha = simulate_regime_switch(
            alpha,
            n,
            rng,
        )

        return (
            sample,
            effective_alpha,
            True,
        )

    raise ValueError(
        f"Unknown distribution: {distribution}"
    )


def hill_features(sample, rng):
    sample = np.asarray(
        sample,
        dtype=float,
    )

    sample = sample[
        np.isfinite(sample)
        & (sample > 0)
    ]

    if len(sample) < 20:
        return pd.DataFrame()

    k_values = K_VALUES[
        K_VALUES < len(sample) - 1
    ]

    alpha_values = []
    bootstrap_stds = []

    sorted_sample = np.sort(
        sample
    )[::-1]

    for k_value in k_values:
        k = int(k_value)

        try:
            alpha_hat = hill_estimator(
                sorted_sample,
                k,
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ):
            alpha_hat = np.nan

        alpha_values.append(
            alpha_hat
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
                bootstrap_alpha = hill_estimator(
                    bootstrap_sample,
                    k,
                )

                if (
                    np.isfinite(
                        bootstrap_alpha
                    )
                    and bootstrap_alpha > 0
                ):
                    bootstrap_estimates.append(
                        bootstrap_alpha
                    )

            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
            ):
                continue

        if len(
            bootstrap_estimates
        ) >= 2:
            bootstrap_std = np.std(
                bootstrap_estimates,
                ddof=1,
            )
        else:
            bootstrap_std = np.nan

        bootstrap_stds.append(
            bootstrap_std
        )

    return build_features(
        k_values,
        np.asarray(
            alpha_values,
            dtype=float,
        ),
        np.asarray(
            bootstrap_stds,
            dtype=float,
        ),
        len(sample),
        WINDOW_SIZE,
    )


def create_labels(
    features,
    target_alpha,
    tail_valid,
):
    if not tail_valid:
        return np.zeros(
            len(features),
            dtype=int,
        )

    alpha_hat = features[
        "alpha_hat"
    ].to_numpy(
        dtype=float
    )

    relative_error = (
        np.abs(
            alpha_hat
            - target_alpha
        )
        / target_alpha
    )

    labels = (
        np.isfinite(
            relative_error
        )
        & (
            relative_error
            <= TOLERANCE
        )
    )

    return labels.astype(
        int
    )


def build_dataset(
    distributions,
    replications,
    rng,
):
    records = []

    total = (
        len(distributions)
        * len(TRUE_ALPHAS)
        * replications
    )

    completed = 0

    for distribution in distributions:
        for alpha in TRUE_ALPHAS:
            for replication in range(
                replications
            ):
                n = int(
                    rng.choice(
                        SAMPLE_SIZES
                    )
                )

                sample, target_alpha, tail_valid = (
                    generate_sample(
                        distribution,
                        alpha,
                        n,
                        rng,
                    )
                )

                features = hill_features(
                    sample,
                    rng,
                )

                if features.empty:
                    completed += 1
                    continue

                labels = create_labels(
                    features,
                    target_alpha,
                    tail_valid,
                )

                k_values = K_VALUES[
                    K_VALUES < len(sample) - 1
                ]

                row_count = min(
                    len(features),
                    len(k_values),
                    len(labels),
                )

                for i in range(
                    row_count
                ):
                    row = {
                        "distribution": distribution,
                        "simulation_alpha": float(alpha),
                        "target_alpha": (
                            float(target_alpha)
                            if np.isfinite(
                                target_alpha
                            )
                            else np.nan
                        ),
                        "tail_valid": int(
                            tail_valid
                        ),
                        "sample_size": int(n),
                        "replication": int(
                            replication
                        ),
                        "k": int(
                            k_values[i]
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

                if completed % 10 == 0:
                    print(
                        f"Progress: {completed}/{total}",
                        flush=True,
                    )

    if not records:
        raise ValueError(
            "No observations were generated."
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
        "Lognormal",
        "Truncated-Pareto",
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
        "M0.10b - VALIDITY-AWARE RELIABILITY CALIBRATION"
    )
    print("=" * 70)

    print(
        "Generating training universe...",
        flush=True,
    )

    training = build_dataset(
        distributions,
        TRAIN_REPLICATIONS,
        rng,
    )

    training = training.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print(
        f"Training observations: {len(training)}",
        flush=True,
    )

    print(
        "Training reliability model...",
        flush=True,
    )

    model, valid_mask = fit_reliability_model(
        training[
            feature_columns
        ],
        training["target"],
    )

    model_path = (
        ROOT
        / "data"
        / "metadata"
        / "tail_reliability_model_v2.joblib"
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
        f"Frozen model saved: {model_path}",
        flush=True,
    )

    print(
        "Generating independent test universe...",
        flush=True,
    )

    testing = build_dataset(
        distributions,
        TEST_REPLICATIONS,
        rng,
    )

    testing = testing.replace(
        [np.inf, -np.inf],
        np.nan,
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
            predicted_reliability=(
                "reliability_probability",
                "mean",
            ),
            observed_reliability=(
                "target",
                "mean",
            ),
            tail_valid=(
                "tail_valid",
                "first",
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
        / "data_driven_reliability_v2_test.csv"
    )

    summary_path = (
        tables_dir
        / "data_driven_reliability_v2_summary.csv"
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

    calibration_curve = (
        valid.assign(
            reliability_bin=calibration_bins
        )
        .groupby(
            "reliability_bin",
            observed=True,
        )
        .agg(
            predicted_reliability=(
                "reliability_probability",
                "mean",
            ),
            observed_reliability=(
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
        / "reliability_calibration_curve_v2.csv"
    )

    calibration_curve[
        [
            "predicted_reliability",
            "observed_reliability",
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
        calibration_curve[
            "predicted_reliability"
        ],
        calibration_curve[
            "observed_reliability"
        ],
        marker="o",
        linewidth=1.5,
        label="Reliability model",
    )

    plt.xlabel(
        "Predicted reliability probability"
    )

    plt.ylabel(
        "Observed reliability"
    )

    plt.title(
        "Validity-Aware Tail Reliability Calibration"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()

    figure_path = (
        figures_dir
        / "validity_aware_reliability_calibration.png"
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
        f"Calibration curve saved : {calibration_path}"
    )

    print(
        f"Calibration figure      : {figure_path}"
    )

    print(
        f"Frozen model            : {model_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
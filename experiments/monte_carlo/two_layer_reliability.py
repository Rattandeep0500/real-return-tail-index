from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tail.hill import hill_estimator
from src.tail.calibration import build_features, predict_reliability
from src.tail.model_validity import compute_model_validity


RANDOM_SEED = 42
SAMPLE_SIZE = 10000
BOOTSTRAPS = 100
K_VALUES = np.arange(40, 1001, 40)
WINDOW_SIZE = 15


def simulate_pareto(alpha, n, rng):
    u = rng.uniform(
        0.0,
        1.0,
        size=n,
    )
    return u ** (-1.0 / alpha)


def simulate_student_t(df, n, rng):
    return np.abs(
        rng.standard_t(
            df=df,
            size=n,
        )
    )


def simulate_lognormal(n, rng):
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

        bootstrap_stds.append(
            np.std(
                bootstrap_estimates,
                ddof=1,
            )
        )

    features = build_features(
        k_values,
        np.asarray(alpha_values),
        np.asarray(bootstrap_stds),
        len(sample),
        WINDOW_SIZE,
    )

    return k_values, features


def main():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    model_path = (
        ROOT
        / "data"
        / "metadata"
        / "tail_reliability_model.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Reliability model not found: {model_path}"
        )

    reliability_model = joblib.load(
        model_path
    )

    distributions = {
        "Pareto": simulate_pareto(
            3.0,
            SAMPLE_SIZE,
            rng,
        ),
        "Student-t": simulate_student_t(
            3.0,
            SAMPLE_SIZE,
            rng,
        ),
        "Lognormal": simulate_lognormal(
            SAMPLE_SIZE,
            rng,
        ),
        "Truncated-Pareto": simulate_truncated_pareto(
            3.0,
            SAMPLE_SIZE,
            rng,
        ),
    }

    all_results = []

    for name, sample in distributions.items():
        k_values, features = calculate_features(
            sample,
            rng,
        )

        estimated = predict_reliability(
            reliability_model,
            features,
        )

        reliability_probability = estimated[
            "reliability_probability"
        ].to_numpy(
            dtype=float
        )

        validity = compute_model_validity(
            sample,
            k_values,
            features[
                "alpha_hat"
            ].to_numpy(
                dtype=float
            ),
            WINDOW_SIZE,
        )

        result = features.copy()

        result[
            "k"
        ] = k_values

        result[
            "estimation_reliability"
        ] = reliability_probability

        result[
            "model_validity"
        ] = validity[
            "model_validity_score"
        ].to_numpy(
            dtype=float
        )

        result[
            "joint_reliability"
        ] = (
            result[
                "estimation_reliability"
            ]
            * result[
                "model_validity"
            ]
        )

        result[
            "distribution"
        ] = name

        all_results.append(
            result
        )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary = (
        combined.groupby(
            "distribution"
        )
        .agg(
            mean_estimation_reliability=(
                "estimation_reliability",
                "mean",
            ),
            mean_model_validity=(
                "model_validity",
                "mean",
            ),
            mean_joint_reliability=(
                "joint_reliability",
                "mean",
            ),
            max_joint_reliability=(
                "joint_reliability",
                "max",
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

    results_path = (
        tables_dir
        / "two_layer_reliability_results.csv"
    )

    summary_path = (
        tables_dir
        / "two_layer_reliability_summary.csv"
    )

    combined.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    plt.figure(
        figsize=(10, 6)
    )

    for name in combined[
        "distribution"
    ].unique():

        subset = combined[
            combined[
                "distribution"
            ] == name
        ]

        plt.plot(
            subset["k"],
            subset[
                "joint_reliability"
            ],
            marker="o",
            linewidth=1.5,
            label=name,
        )

    plt.xlabel("k")
    plt.ylabel("Joint reliability")
    plt.title(
        "Two-Layer Tail Reliability"
    )
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()

    figure_path = (
        figures_dir
        / "two_layer_tail_reliability.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 70)
    print(
        "M0.12 - TWO-LAYER TAIL RELIABILITY"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: {results_path}"
    )

    print(
        f"Summary saved to: {summary_path}"
    )

    print(
        f"Figure saved to: {figure_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
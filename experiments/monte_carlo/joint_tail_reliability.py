from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src.tail.adaptive_conformal import (
    fit_adaptive_conformal,
    build_adaptive_certificate,
)

from src.tail.tail_fit_diagnostics import (
    analyze_tail_fit,
)


CALIBRATION_FILE = (
    ROOT
    / "tables"
    / "adaptive_conformal_test.csv"
)

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "joint_tail_reliability.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "joint_tail_reliability_summary.csv"
)


RNG = np.random.default_rng(
    20260914
)


N_REPLICATIONS = 300
N_TEST = 120
BOOTSTRAPS = 30

TARGET_COVERAGE = 0.90

K_VALUES = np.arange(
    10,
    81,
    5,
)


DISTRIBUTIONS = [
    "Pareto",
    "Mixture",
    "Regime-Switch",
    "Student-t",
    "Truncated-Pareto",
    "Volatility-Clustered",
    "Lognormal",
]


VALID_DISTRIBUTIONS = {
    "Pareto",
    "Mixture",
    "Regime-Switch",
    "Student-t",
    "Volatility-Clustered",
}


TRUE_ALPHA = {
    "Pareto": 3.0,
    "Mixture": 3.0,
    "Regime-Switch": 2.5,
    "Student-t": 3.0,
    "Truncated-Pareto": np.nan,
    "Volatility-Clustered": np.nan,
    "Lognormal": np.nan,
}


def generate_sample(
    distribution,
    n,
):
    if distribution == "Pareto":
        return (
            RNG.pareto(
                3.0,
                n,
            )
            + 1.0
        )

    if distribution == "Mixture":
        selector = RNG.random(
            n
        )

        pareto = (
            RNG.pareto(
                3.0,
                n,
            )
            + 1.0
        )

        lognormal = np.exp(
            RNG.normal(
                0.0,
                0.8,
                n,
            )
        )

        return np.where(
            selector < 0.70,
            pareto,
            lognormal,
        )

    if distribution == "Regime-Switch":
        selector = RNG.random(
            n
        )

        regime_a = (
            RNG.pareto(
                2.5,
                n,
            )
            + 1.0
        )

        regime_b = (
            RNG.pareto(
                4.0,
                n,
            )
            + 1.0
        )

        return np.where(
            selector < 0.50,
            regime_a,
            regime_b,
        )

    if distribution == "Student-t":
        return (
            np.abs(
                RNG.standard_t(
                    3.0,
                    n,
                )
            )
            + 1.0
        )

    if distribution == "Truncated-Pareto":
        x = (
            RNG.pareto(
                3.0,
                n,
            )
            + 1.0
        )

        return np.minimum(
            x,
            8.0,
        )

    if distribution == "Volatility-Clustered":
        shocks = RNG.normal(
            0.0,
            1.0,
            n,
        )

        volatility = np.empty(
            n,
            dtype=float,
        )

        volatility[0] = 0.5

        for i in range(
            1,
            n,
        ):
            volatility[i] = (
                0.10
                + 0.85
                * volatility[i - 1]
                + 0.10
                * abs(
                    shocks[i - 1]
                )
            )

        return (
            np.abs(
                shocks
                * volatility
            )
            + 1.0
        )

    if distribution == "Lognormal":
        return np.exp(
            RNG.normal(
                0.0,
                1.0,
                n,
            )
        )

    raise ValueError(
        f"Unknown distribution: {distribution}"
    )


def build_calibration_model():
    calibration = pd.read_csv(
        CALIBRATION_FILE
    )

    required = [
        "k",
        "alpha_hat",
        "bootstrap_cv",
        "sample_size",
        "true_alpha",
    ]

    missing = [
        column
        for column in required
        if column not in calibration.columns
    ]

    if missing:
        raise ValueError(
            f"Missing calibration columns: {missing}"
        )

    calibration = calibration[
        required
    ].copy()

    calibration = calibration.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    calibration = calibration.dropna()

    if len(calibration) < 100:
        raise RuntimeError(
            "Insufficient calibration observations."
        )

    model = fit_adaptive_conformal(
        calibration_frame=calibration,
        alpha_true_column="true_alpha",
        alpha_hat_column="alpha_hat",
        bootstrap_cv_column="bootstrap_cv",
        k_column="k",
        sample_size_column="sample_size",
        miscoverage=(
            1.0
            - TARGET_COVERAGE
        ),
        alpha_bins=4,
        stability_bins=3,
        k_bins=4,
        min_group_size=40,
    )

    return model, calibration


def calculate_uncertainty_certificate(
    alpha_hat,
    bootstrap_cv,
    k,
    sample_size,
    model,
):
    certificate = (
        build_adaptive_certificate(
            alpha_hat=np.array(
                [alpha_hat],
                dtype=float,
            ),
            bootstrap_cv=np.array(
                [bootstrap_cv],
                dtype=float,
            ),
            k=np.array(
                [k],
                dtype=float,
            ),
            sample_size=np.array(
                [sample_size],
                dtype=float,
            ),
            model=model,
            max_relative_width=1.0,
        )
    )

    return {
        "certificate": float(
            certificate[
                "certificate_score"
            ].iloc[0]
        ),
        "radius": float(
            certificate[
                "conformal_radius"
            ].iloc[0]
        ),
        "lower": float(
            certificate[
                "conformal_lower"
            ].iloc[0]
        ),
        "upper": float(
            certificate[
                "conformal_upper"
            ].iloc[0]
        ),
        "relative_width": float(
            certificate[
                "relative_interval_width"
            ].iloc[0]
        ),
        "radius_source": (
            certificate[
                "radius_source"
            ].iloc[0]
        ),
        "abstain": bool(
            certificate[
                "abstain"
            ].iloc[0]
        ),
    }


def main():
    print(
        "Loading adaptive-conformal calibration..."
    )

    conformal_model, calibration = (
        build_calibration_model()
    )

    print(
        f"Calibration observations: {len(calibration)}"
    )

    print(
        f"Global conformal radius: {conformal_model['global_radius']:.6f}"
    )

    total = (
        len(DISTRIBUTIONS)
        * N_REPLICATIONS
    )

    rows = []

    completed = 0

    print(
        f"Test progress: 0/{total}"
    )

    for distribution in DISTRIBUTIONS:
        true_alpha = TRUE_ALPHA[
            distribution
        ]

        tail_valid = int(
            distribution
            in VALID_DISTRIBUTIONS
        )

        for replication in range(
            N_REPLICATIONS
        ):
            sample = generate_sample(
                distribution,
                N_TEST,
            )

            diagnostic = (
                analyze_tail_fit(
                    sample,
                    K_VALUES,
                    BOOTSTRAPS,
                    RNG,
                )
            )

            alpha_hat = diagnostic[
                "alpha_hat"
            ]

            selected_k = diagnostic[
                "selected_k"
            ]

            bootstrap_cv = diagnostic[
                "bootstrap_cv"
            ]

            model_validity = diagnostic[
                "validity_score"
            ]

            if not np.isfinite(
                alpha_hat
            ):
                completed += 1
                continue

            if not np.isfinite(
                selected_k
            ):
                completed += 1
                continue

            if not np.isfinite(
                bootstrap_cv
            ):
                completed += 1
                continue

            if not np.isfinite(
                model_validity
            ):
                completed += 1
                continue

            selected_k = int(
                selected_k
            )

            uncertainty = (
                calculate_uncertainty_certificate(
                    alpha_hat=alpha_hat,
                    bootstrap_cv=bootstrap_cv,
                    k=selected_k,
                    sample_size=N_TEST,
                    model=conformal_model,
                )
            )

            uncertainty_certificate = (
                uncertainty[
                    "certificate"
                ]
            )

            joint_reliability = (
                uncertainty_certificate
                * model_validity
            )

            covered = np.nan

            if np.isfinite(
                true_alpha
            ):
                covered = bool(
                    uncertainty[
                        "lower"
                    ]
                    <= true_alpha
                    <= uncertainty[
                        "upper"
                    ]
                )

            rows.append(
                {
                    "distribution": distribution,
                    "replication": replication,
                    "k": selected_k,
                    "alpha_hat": alpha_hat,
                    "bootstrap_cv": bootstrap_cv,
                    "true_alpha": true_alpha,
                    "model_validity": model_validity,
                    "uncertainty_certificate": uncertainty_certificate,
                    "joint_reliability": joint_reliability,
                    "conformal_radius": uncertainty[
                        "radius"
                    ],
                    "conformal_lower": uncertainty[
                        "lower"
                    ],
                    "conformal_upper": uncertainty[
                        "upper"
                    ],
                    "relative_interval_width": uncertainty[
                        "relative_width"
                    ],
                    "radius_source": uncertainty[
                        "radius_source"
                    ],
                    "abstain": uncertainty[
                        "abstain"
                    ],
                    "covered": covered,
                    "tail_valid": tail_valid,
                }
            )

            completed += 1

            if completed % 50 == 0:
                print(
                    f"Test progress: {completed}/{total}"
                )

    result = pd.DataFrame(
        rows
    )

    if len(result) == 0:
        raise RuntimeError(
            "No valid test observations were produced."
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary_rows = []

    for distribution, group in (
        result.groupby(
            "distribution"
        )
    ):
        coverage = (
            group[
                "covered"
            ]
            .dropna()
        )

        summary_rows.append(
            {
                "distribution": distribution,
                "n": len(group),
                "coverage": (
                    coverage.mean()
                    if len(
                        coverage
                    ) > 0
                    else np.nan
                ),
                "mean_model_validity": group[
                    "model_validity"
                ].mean(),
                "mean_uncertainty_certificate": group[
                    "uncertainty_certificate"
                ].mean(),
                "mean_joint_reliability": group[
                    "joint_reliability"
                ].mean(),
                "mean_relative_interval_width": group[
                    "relative_interval_width"
                ].mean(),
                "abstention_rate": group[
                    "abstain"
                ].mean(),
            }
        )

    auc_data = result[
        [
            "tail_valid",
            "model_validity",
            "uncertainty_certificate",
            "joint_reliability",
        ]
    ].dropna()

    y_true = auc_data[
        "tail_valid"
    ].to_numpy(
        dtype=int
    )

    model_scores = auc_data[
        "model_validity"
    ].to_numpy(
        dtype=float
    )

    uncertainty_scores = auc_data[
        "uncertainty_certificate"
    ].to_numpy(
        dtype=float
    )

    joint_scores = auc_data[
        "joint_reliability"
    ].to_numpy(
        dtype=float
    )

    if len(
        np.unique(
            y_true
        )
    ) == 2:
        model_auc = roc_auc_score(
            y_true,
            model_scores,
        )

        uncertainty_auc = (
            roc_auc_score(
                y_true,
                uncertainty_scores,
            )
        )

        joint_auc = roc_auc_score(
            y_true,
            joint_scores,
        )

        model_pr_auc = (
            average_precision_score(
                y_true,
                model_scores,
            )
        )

        uncertainty_pr_auc = (
            average_precision_score(
                y_true,
                uncertainty_scores,
            )
        )

        joint_pr_auc = (
            average_precision_score(
                y_true,
                joint_scores,
            )
        )
    else:
        model_auc = np.nan
        uncertainty_auc = np.nan
        joint_auc = np.nan
        model_pr_auc = np.nan
        uncertainty_pr_auc = np.nan
        joint_pr_auc = np.nan

    finite_coverage = result[
        "covered"
    ].dropna()

    overall_coverage = (
        finite_coverage.mean()
        if len(
            finite_coverage
        ) > 0
        else np.nan
    )

    summary_rows.append(
        {
            "distribution": "OVERALL",
            "n": len(result),
            "coverage": overall_coverage,
            "mean_model_validity": result[
                "model_validity"
            ].mean(),
            "mean_uncertainty_certificate": result[
                "uncertainty_certificate"
            ].mean(),
            "mean_joint_reliability": result[
                "joint_reliability"
            ].mean(),
            "mean_relative_interval_width": result[
                "relative_interval_width"
            ].mean(),
            "abstention_rate": result[
                "abstain"
            ].mean(),
            "model_roc_auc": model_auc,
            "uncertainty_roc_auc": uncertainty_auc,
            "joint_roc_auc": joint_auc,
            "model_pr_auc": model_pr_auc,
            "uncertainty_pr_auc": uncertainty_pr_auc,
            "joint_pr_auc": joint_pr_auc,
            "calibration_observations": len(
                calibration
            ),
            "global_radius": conformal_model[
                "global_radius"
            ],
            "radius_source_counts": str(
                result[
                    "radius_source"
                ]
                .value_counts()
                .to_dict()
            ),
        }
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        "M1.9 JOINT TAIL RELIABILITY"
    )
    print(
        "================================"
    )
    print(
        f"Calibration observations: {len(calibration)}"
    )
    print(
        f"Global conformal radius: {conformal_model['global_radius']:.6f}"
    )
    print(
        f"Test observations: {len(result)}"
    )
    print(
        f"Overall coverage: {overall_coverage:.6f}"
    )
    print(
        f"Model-validity ROC AUC: {model_auc:.6f}"
    )
    print(
        f"Uncertainty ROC AUC: {uncertainty_auc:.6f}"
    )
    print(
        f"Joint ROC AUC: {joint_auc:.6f}"
    )
    print(
        f"Model-validity PR AUC: {model_pr_auc:.6f}"
    )
    print(
        f"Uncertainty PR AUC: {uncertainty_pr_auc:.6f}"
    )
    print(
        f"Joint PR AUC: {joint_pr_auc:.6f}"
    )
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )
    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
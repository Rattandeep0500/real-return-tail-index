from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_SCRIPT = (
    ROOT
    / "experiments"
    / "real_data"
    / "tail_model_competition.py"
)

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "tail_model_identifiability_vs_k.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "tail_model_identifiability_vs_k_summary.csv"
)

SEED = 20260914

REPLICATES = 40

SAMPLE_SIZES = [
    500,
    1000,
    2000,
    5000,
]

K_VALUES = [
    20,
    40,
    60,
    80,
    120,
    160,
]

PARETO_ALPHA = 3.0

TRUNCATED_ALPHA = 3.0

TRUNCATED_UPPER = 8.0

LOGNORMAL_MU = 0.0

LOGNORMAL_SIGMA = 0.65

STUDENT_DF = 5.0

STUDENT_SCALE = 1.0


def load_production_module():
    spec = importlib.util.spec_from_file_location(
        "tail_model_competition",
        PRODUCTION_SCRIPT,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def generate_pareto(
    rng,
    n,
):
    u = rng.uniform(
        0.0,
        1.0,
        n,
    )

    return (
        1.0
        * (1.0 - u)
        ** (-1.0 / PARETO_ALPHA)
    )


def generate_truncated_pareto(
    rng,
    n,
):
    lower_ratio = (
        1.0
        / TRUNCATED_UPPER
    ) ** TRUNCATED_ALPHA

    u = rng.uniform(
        0.0,
        1.0,
        n,
    )

    return (
        1.0
        * (
            1.0
            - u
            * (
                1.0
                - lower_ratio
            )
        )
        ** (
            -1.0
            / TRUNCATED_ALPHA
        )
    )


def generate_lognormal(
    rng,
    n,
):
    return rng.lognormal(
        mean=LOGNORMAL_MU,
        sigma=LOGNORMAL_SIGMA,
        size=n,
    )


def generate_student_t(
    rng,
    n,
):
    return np.abs(
        rng.standard_t(
            df=STUDENT_DF,
            size=n,
        )
        * STUDENT_SCALE
    )


def generate_mixture(
    rng,
    n,
):
    indicator = (
        rng.uniform(
            0.0,
            1.0,
            n,
        )
        < 0.70
    )

    lognormal_part = generate_lognormal(
        rng,
        n,
    )

    pareto_part = generate_pareto(
        rng,
        n,
    )

    return np.where(
        indicator,
        lognormal_part,
        pareto_part,
    )


def generate_sample(
    distribution,
    rng,
    n,
):
    if distribution == "Pareto":
        return generate_pareto(
            rng,
            n,
        )

    if distribution == "Truncated-Pareto":
        return generate_truncated_pareto(
            rng,
            n,
        )

    if distribution == "Lognormal":
        return generate_lognormal(
            rng,
            n,
        )

    if distribution == "Student-t":
        return generate_student_t(
            rng,
            n,
        )

    if distribution == "Mixture":
        return generate_mixture(
            rng,
            n,
        )

    raise ValueError(
        f"Unknown distribution: {distribution}"
    )


def run_competition(
    production,
    sample,
    k,
):
    sample = production.clean_values(
        sample
    )

    if k >= len(sample):
        return None

    sample = np.sort(
        sample
    )[::-1]

    threshold = float(
        sample[k]
    )

    tail = sample[:k]

    fits = production.fit_all_models(
        tail,
        threshold,
    )

    selection = production.select_best_model(
        fits
    )

    if selection is None:
        return None

    comparison = (
        selection["comparison"]
        .copy()
    )

    comparison = comparison.sort_values(
        "aic"
    ).reset_index(
        drop=True
    )

    return {
        "selected_model": str(
            selection[
                "selected_model"
            ]
        ),
        "selected_ks": float(
            selection[
                "selected_ks"
            ]
        ),
        "aic_gap": float(
            selection[
                "aic_gap"
            ]
        ),
        "candidate_count": int(
            selection[
                "candidate_count"
            ]
        ),
    }


def main():
    production = (
        load_production_module()
    )

    rng = np.random.default_rng(
        SEED
    )

    distributions = [
        "Pareto",
        "Truncated-Pareto",
        "Lognormal",
        "Student-t",
        "Mixture",
    ]

    rows = []

    total_attempted = (
        len(distributions)
        * len(SAMPLE_SIZES)
        * len(K_VALUES)
        * REPLICATES
    )

    completed = 0

    for distribution in distributions:
        for sample_size in SAMPLE_SIZES:
            for replicate in range(
                REPLICATES
            ):
                sample = generate_sample(
                    distribution,
                    rng,
                    sample_size,
                )

                for k in K_VALUES:
                    if k >= sample_size:
                        continue

                    result = run_competition(
                        production,
                        sample,
                        k,
                    )

                    if result is None:
                        continue

                    completed += 1

                    rows.append(
                        {
                            "distribution": distribution,
                            "sample_size": sample_size,
                            "k": k,
                            "k_fraction": (
                                k
                                / sample_size
                            ),
                            "replicate": replicate,
                            "selected_model": result[
                                "selected_model"
                            ],
                            "selected_ks": result[
                                "selected_ks"
                            ],
                            "aic_gap": result[
                                "aic_gap"
                            ],
                            "candidate_count": result[
                                "candidate_count"
                            ],
                            "correct_selection": (
                                result[
                                    "selected_model"
                                ]
                                == distribution
                            ),
                        }
                    )

    result = pd.DataFrame(
        rows
    )

    if len(result) == 0:
        raise RuntimeError(
            "No validation results were produced."
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary_rows = []

    for distribution in distributions:
        for sample_size in SAMPLE_SIZES:
            for k in K_VALUES:
                group = result[
                    (
                        result[
                            "distribution"
                        ]
                        == distribution
                    )
                    & (
                        result[
                            "sample_size"
                        ]
                        == sample_size
                    )
                    & (
                        result["k"]
                        == k
                    )
                ]

                if len(group) == 0:
                    continue

                summary_rows.append(
                    {
                        "distribution": distribution,
                        "sample_size": sample_size,
                        "k": k,
                        "k_fraction": (
                            k
                            / sample_size
                        ),
                        "observations": len(group),
                        "recovery_rate": group[
                            "correct_selection"
                        ].mean(),
                        "mean_aic_gap": group[
                            "aic_gap"
                        ].mean(),
                        "median_aic_gap": group[
                            "aic_gap"
                        ].median(),
                        "mean_selected_ks": group[
                            "selected_ks"
                        ].mean(),
                    }
                )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    recovery_by_n = (
        summary[
            summary[
                "distribution"
            ].isin(
                [
                    "Pareto",
                    "Truncated-Pareto",
                    "Lognormal",
                    "Student-t",
                ]
            )
        ]
        .groupby(
            [
                "distribution",
                "sample_size",
            ],
            as_index=False,
        )[
            "recovery_rate"
        ]
        .mean()
    )

    recovery_by_k = (
        summary[
            summary[
                "distribution"
            ].isin(
                [
                    "Pareto",
                    "Truncated-Pareto",
                    "Lognormal",
                    "Student-t",
                ]
            )
        ]
        .groupby(
            [
                "distribution",
                "k",
            ],
            as_index=False,
        )[
            "recovery_rate"
        ]
        .mean()
    )

    print(
        f"Validation runs attempted: {total_attempted}"
    )

    print(
        f"Validation runs completed: {completed}"
    )

    print()

    print(
        "Overall exact-model recovery:"
    )

    exact = result[
        result[
            "distribution"
        ].isin(
            [
                "Pareto",
                "Truncated-Pareto",
                "Lognormal",
                "Student-t",
            ]
        )
    ]

    overall_recovery = (
        exact
        .groupby(
            "distribution"
        )[
            "correct_selection"
        ]
        .mean()
    )

    print(
        overall_recovery.to_string()
    )

    print()

    print(
        "Recovery by sample size:"
    )

    print(
        recovery_by_n.to_string(
            index=False
        )
    )

    print()

    print(
        "Recovery by k:"
    )

    print(
        recovery_by_k.to_string(
            index=False
        )
    )

    print()

    print(
        "Best recovery configuration by distribution:"
    )

    for distribution in [
        "Pareto",
        "Truncated-Pareto",
        "Lognormal",
        "Student-t",
    ]:
        group = summary[
            summary[
                "distribution"
            ]
            == distribution
        ]

        best = group.loc[
            group[
                "recovery_rate"
            ].idxmax()
        ]

        print(
            f"{distribution}: "
            f"recovery={best['recovery_rate']:.4f}, "
            f"n={int(best['sample_size'])}, "
            f"k={int(best['k'])}, "
            f"k/n={best['k_fraction']:.4f}"
        )

    print()

    print(
        f"Detailed results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
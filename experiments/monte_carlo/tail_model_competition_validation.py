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
    / "tail_model_competition_validation.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "tail_model_competition_validation_summary.csv"
)

SEED = 20260914
REPLICATES = 150
SAMPLE_SIZE = 120
K_VALUES = [20, 30, 40]

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
        * (1.0 - u) ** (
            -1.0 / PARETO_ALPHA
        )
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
    indicator = rng.uniform(
        0.0,
        1.0,
        n,
    ) < 0.70

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
    name,
    rng,
    n,
):
    if name == "Pareto":
        return generate_pareto(
            rng,
            n,
        )

    if name == "Truncated-Pareto":
        return generate_truncated_pareto(
            rng,
            n,
        )

    if name == "Lognormal":
        return generate_lognormal(
            rng,
            n,
        )

    if name == "Student-t":
        return generate_student_t(
            rng,
            n,
        )

    if name == "Mixture":
        return generate_mixture(
            rng,
            n,
        )

    raise ValueError(
        f"Unknown distribution: {name}"
    )


def run_single(
    production,
    sample,
    k,
):
    sample = production.clean_values(
        sample
    )

    sample = np.sort(
        sample
    )[::-1]

    if k >= len(sample):
        return None

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

    selected_model = (
        selection[
            "selected_model"
        ]
    )

    selected_row = (
        selection[
            "comparison"
        ]
        .iloc[0]
    )

    return {
        "selected_model": selected_model,
        "selected_aic": float(
            selected_row["aic"]
        ),
        "selected_ks": float(
            selected_row["ks"]
        ),
        "aic_gap": float(
            selection["aic_gap"]
        ),
        "candidate_count": int(
            selection["candidate_count"]
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

    total_runs = (
        len(distributions)
        * REPLICATES
        * len(K_VALUES)
    )

    run_number = 0

    for distribution in distributions:
        for replicate in range(
            REPLICATES
        ):
            sample = generate_sample(
                distribution,
                rng,
                SAMPLE_SIZE,
            )

            for k in K_VALUES:
                run_number += 1

                result = run_single(
                    production,
                    sample,
                    k,
                )

                if result is None:
                    continue

                rows.append(
                    {
                        "distribution": distribution,
                        "replicate": replicate,
                        "k": k,
                        "sample_size": SAMPLE_SIZE,
                        "selected_model": result[
                            "selected_model"
                        ],
                        "selected_aic": result[
                            "selected_aic"
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
        group = result[
            result["distribution"]
            == distribution
        ]

        if len(group) == 0:
            continue

        selection_counts = (
            group[
                "selected_model"
            ]
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        for k in K_VALUES:
            k_group = group[
                group["k"]
                == k
            ]

            if len(k_group) == 0:
                continue

            summary_rows.append(
                {
                    "distribution": distribution,
                    "k": k,
                    "observations": len(
                        k_group
                    ),
                    "mean_aic_gap": k_group[
                        "aic_gap"
                    ].mean(),
                    "mean_selected_ks": k_group[
                        "selected_ks"
                    ].mean(),
                    "pareto_selection_rate": selection_counts.get(
                        "Pareto",
                        0.0,
                    ),
                    "lognormal_selection_rate": selection_counts.get(
                        "Lognormal",
                        0.0,
                    ),
                    "student_t_selection_rate": selection_counts.get(
                        "Student-t",
                        0.0,
                    ),
                    "truncated_pareto_selection_rate": selection_counts.get(
                        "Truncated-Pareto",
                        0.0,
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

    recovery_rows = []

    exact_distributions = [
        "Pareto",
        "Truncated-Pareto",
        "Lognormal",
        "Student-t",
    ]

    for distribution in exact_distributions:
        group = result[
            result["distribution"]
            == distribution
        ]

        if len(group) == 0:
            continue

        recovery_rows.append(
            {
                "distribution": distribution,
                "overall_recovery_rate": (
                    group[
                        "selected_model"
                    ]
                    == distribution
                ).mean(),
                "mean_aic_gap": group[
                    "aic_gap"
                ].mean(),
                "mean_selected_ks": group[
                    "selected_ks"
                ].mean(),
            }
        )

    recovery = pd.DataFrame(
        recovery_rows
    )

    print(
        f"Validation runs attempted: {total_runs}"
    )

    print(
        f"Validation runs completed: {len(result)}"
    )

    print()

    print(
        "Exact-model recovery:"
    )

    print(
        recovery.to_string(
            index=False
        )
    )

    print()

    print(
        "Selection rates by distribution and k:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Overall selected-model distribution:"
    )

    print(
        result[
            "selected_model"
        ]
        .value_counts(
            normalize=True
        )
        .to_string()
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
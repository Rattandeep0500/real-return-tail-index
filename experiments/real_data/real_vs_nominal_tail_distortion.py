from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "sp500_real_returns.csv"
)

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "real_vs_nominal_tail_distortion.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "real_vs_nominal_tail_distortion_summary.csv"
)

OVERALL_FILE = (
    ROOT
    / "tables"
    / "real_vs_nominal_tail_distortion_overall.csv"
)

FIGURE_FILE = (
    ROOT
    / "figures"
    / "real_vs_nominal_tail_distortion.png"
)

WINDOW_SIZE = 120

K_VALUES = list(
    range(
        10,
        81,
        5,
    )
)

REFERENCE_K = 30


def hill_alpha(
    sample,
    k,
):
    sample = np.asarray(
        sample,
        dtype=float,
    )

    sample = sample[
        np.isfinite(sample)
        & (sample > 0)
    ]

    if len(sample) <= k:
        return np.nan

    sample = np.sort(
        sample
    )[::-1]

    threshold = sample[k]

    if threshold <= 0:
        return np.nan

    values = sample[:k]

    logs = np.log(
        values / threshold
    )

    denominator = np.sum(
        logs
    )

    if denominator <= 0:
        return np.nan

    return float(
        k / denominator
    )


def get_column(
    data,
    candidates,
):
    for column in candidates:
        if column in data.columns:
            return column

    raise ValueError(
        f"None of these columns were found: {candidates}"
    )


def summarize_distortion(
    data,
):
    rows = []

    for side in [
        "left",
        "right",
    ]:
        group = data[
            data["side"] == side
        ].copy()

        if len(group) == 0:
            continue

        for k in K_VALUES:
            k_group = group[
                group["k"] == k
            ].copy()

            if len(k_group) == 0:
                continue

            distortion = (
                k_group[
                    "alpha_distortion"
                ]
                .dropna()
            )

            if len(distortion) == 0:
                continue

            abs_distortion = np.abs(
                distortion
            )

            rows.append(
                {
                    "side": side,
                    "k": k,
                    "observations": len(
                        distortion
                    ),
                    "mean_real_alpha": k_group[
                        "real_alpha"
                    ].mean(),
                    "mean_nominal_alpha": k_group[
                        "nominal_alpha"
                    ].mean(),
                    "mean_alpha_distortion": distortion.mean(),
                    "median_alpha_distortion": distortion.median(),
                    "std_alpha_distortion": distortion.std(),
                    "mean_abs_alpha_distortion": abs_distortion.mean(),
                    "q10_alpha_distortion": distortion.quantile(
                        0.10
                    ),
                    "q90_alpha_distortion": distortion.quantile(
                        0.90
                    ),
                    "fraction_abs_gt_010": (
                        abs_distortion > 0.10
                    ).mean(),
                    "fraction_abs_gt_025": (
                        abs_distortion > 0.25
                    ).mean(),
                    "fraction_abs_gt_050": (
                        abs_distortion > 0.50
                    ).mean(),
                    "fraction_real_thicker": (
                        distortion < 0
                    ).mean(),
                    "fraction_real_thinner": (
                        distortion > 0
                    ).mean(),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_overall_summary(
    data,
):
    rows = []

    for side in [
        "left",
        "right",
    ]:
        group = data[
            data["side"] == side
        ].copy()

        distortion = group[
            "alpha_distortion"
        ].dropna()

        abs_distortion = np.abs(
            distortion
        )

        if len(distortion) == 0:
            continue

        rows.append(
            {
                "side": side,
                "observations": len(
                    distortion
                ),
                "mean_alpha_distortion": distortion.mean(),
                "median_alpha_distortion": distortion.median(),
                "mean_abs_alpha_distortion": abs_distortion.mean(),
                "std_alpha_distortion": distortion.std(),
                "fraction_abs_gt_010": (
                    abs_distortion > 0.10
                ).mean(),
                "fraction_abs_gt_025": (
                    abs_distortion > 0.25
                ).mean(),
                "fraction_abs_gt_050": (
                    abs_distortion > 0.50
                ).mean(),
                "fraction_real_thicker": (
                    distortion < 0
                ).mean(),
                "fraction_real_thinner": (
                    distortion > 0
                ).mean(),
                "mean_real_alpha": group[
                    "real_alpha"
                ].mean(),
                "mean_nominal_alpha": group[
                    "nominal_alpha"
                ].mean(),
            }
        )

    all_distortion = data[
        "alpha_distortion"
    ].dropna()

    abs_all = np.abs(
        all_distortion
    )

    rows.append(
        {
            "side": "overall",
            "observations": len(
                all_distortion
            ),
            "mean_alpha_distortion": all_distortion.mean(),
            "median_alpha_distortion": all_distortion.median(),
            "mean_abs_alpha_distortion": abs_all.mean(),
            "std_alpha_distortion": all_distortion.std(),
            "fraction_abs_gt_010": (
                abs_all > 0.10
            ).mean(),
            "fraction_abs_gt_025": (
                abs_all > 0.25
            ).mean(),
            "fraction_abs_gt_050": (
                abs_all > 0.50
            ).mean(),
            "fraction_real_thicker": (
                all_distortion < 0
            ).mean(),
            "fraction_real_thinner": (
                all_distortion > 0
            ).mean(),
            "mean_real_alpha": data[
                "real_alpha"
            ].mean(),
            "mean_nominal_alpha": data[
                "nominal_alpha"
            ].mean(),
        }
    )

    return pd.DataFrame(
        rows
    )


def create_reference_plot(
    data,
):
    reference = data[
        data["k"] == REFERENCE_K
    ].copy()

    if len(reference) == 0:
        return

    reference = reference.sort_values(
        [
            "date",
            "side",
        ]
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    for side in [
        "left",
        "right",
    ]:
        group = reference[
            reference["side"] == side
        ]

        if len(group) == 0:
            continue

        axis.plot(
            group["date"],
            group["alpha_distortion"],
            label=f"{side} tail",
        )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_title(
        f"Real-vs-Nominal Hill Tail Distortion at k={REFERENCE_K}"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Real alpha - Nominal alpha"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURE_FILE,
        dpi=200,
    )

    plt.close(
        figure
    )


def main():
    data = pd.read_csv(
        INPUT_FILE,
        index_col=0,
    )

    data = data.reset_index()

    data = data.rename(
        columns={
            data.columns[0]: "date"
        }
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    nominal_column = get_column(
        data,
        [
            "nominal_return",
            "nominal_returns",
            "return_nominal",
        ],
    )

    real_column = get_column(
        data,
        [
            "real_return",
            "real_returns",
            "return_real",
        ],
    )

    nominal_returns = pd.to_numeric(
        data[
            nominal_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    real_returns = pd.to_numeric(
        data[
            real_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    dates = data[
        "date"
    ].to_numpy()

    rows = []

    total_windows = (
        len(data)
        - WINDOW_SIZE
        + 1
    )

    for start in range(
        total_windows
    ):
        end = (
            start
            + WINDOW_SIZE
        )

        nominal_window = nominal_returns[
            start:end
        ]

        real_window = real_returns[
            start:end
        ]

        date = pd.Timestamp(
            dates[end - 1]
        )

        for side in [
            "left",
            "right",
        ]:
            if side == "left":
                nominal_magnitude = -nominal_window[
                    nominal_window < 0
                ]

                real_magnitude = -real_window[
                    real_window < 0
                ]

            else:
                nominal_magnitude = nominal_window[
                    nominal_window > 0
                ]

                real_magnitude = real_window[
                    real_window > 0
                ]

            for k in K_VALUES:
                nominal_alpha = hill_alpha(
                    nominal_magnitude,
                    k,
                )

                real_alpha = hill_alpha(
                    real_magnitude,
                    k,
                )

                if (
                    not np.isfinite(
                        nominal_alpha
                    )
                    or not np.isfinite(
                        real_alpha
                    )
                ):
                    continue

                distortion = (
                    real_alpha
                    - nominal_alpha
                )

                rows.append(
                    {
                        "date": date,
                        "side": side,
                        "k": k,
                        "nominal_alpha": nominal_alpha,
                        "real_alpha": real_alpha,
                        "alpha_distortion": distortion,
                        "absolute_alpha_distortion": abs(
                            distortion
                        ),
                    }
                )

    result = pd.DataFrame(
        rows
    )

    if len(result) == 0:
        raise RuntimeError(
            "No valid rolling tail estimates were produced."
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = summarize_distortion(
        result
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    overall = build_overall_summary(
        result
    )

    overall.to_csv(
        OVERALL_FILE,
        index=False,
    )

    create_reference_plot(
        result
    )

    reference = result[
        result["k"] == REFERENCE_K
    ]

    print(
        f"Rolling windows: {total_windows}"
    )

    print(
        f"Distortion observations: {len(result)}"
    )

    print()

    print(
        f"Reference k: {REFERENCE_K}"
    )

    print()

    print(
        "Overall real-vs-nominal distortion:"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print()

    print(
        "Distortion summary by side and k:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Reference-k summary:"
    )

    reference_overall = (
        reference
        .groupby(
            "side"
        )[
            "alpha_distortion"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
    )

    print(
        reference_overall.to_string()
    )

    print()

    print(
        f"Detailed results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )

    print(
        f"Overall summary saved to: {OVERALL_FILE}"
    )

    print(
        f"Figure saved to: {FIGURE_FILE}"
    )


if __name__ == "__main__":
    main()
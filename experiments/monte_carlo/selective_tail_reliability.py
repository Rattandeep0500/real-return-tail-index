from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


INPUT_PATH = (
    ROOT
    / "tables"
    / "adaptive_conformal_test.csv"
)

TABLES_DIR = ROOT / "tables"

WIDTH_THRESHOLDS = [
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    1.00,
    1.25,
    1.50,
    2.00,
    3.00,
]


def evaluate_threshold(
    data,
    threshold,
):
    accepted = (
        data[
            "relative_interval_width"
        ]
        <= threshold
    )

    accepted_data = data[
        accepted
    ].copy()

    acceptance_rate = (
        accepted.mean()
    )

    if len(accepted_data) == 0:
        return {
            "width_threshold": threshold,
            "acceptance_rate": 0.0,
            "abstention_rate": 1.0,
            "coverage": np.nan,
            "accepted_observations": 0,
        }

    coverage = (
        accepted_data[
            "covered"
        ].mean()
    )

    return {
        "width_threshold": threshold,
        "acceptance_rate": float(
            acceptance_rate
        ),
        "abstention_rate": float(
            1.0 - acceptance_rate
        ),
        "coverage": float(
            coverage
        ),
        "accepted_observations": int(
            len(accepted_data)
        ),
    }


def evaluate_by_distribution(
    data,
    threshold,
):
    accepted = (
        data[
            "relative_interval_width"
        ]
        <= threshold
    )

    rows = []

    for distribution, subset in data.groupby(
        "distribution"
    ):
        subset = subset[
            subset[
                "relative_interval_width"
            ]
            <= threshold
        ]

        if len(subset) == 0:
            coverage = np.nan
        else:
            coverage = float(
                subset[
                    "covered"
                ].mean()
            )

        total = len(
            data[
                data[
                    "distribution"
                ] == distribution
            ]
        )

        rows.append(
            {
                "width_threshold": threshold,
                "distribution": distribution,
                "coverage": coverage,
                "acceptance_rate": (
                    len(subset) / total
                    if total > 0
                    else 0.0
                ),
                "accepted_observations": len(
                    subset
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}"
        )

    data = pd.read_csv(
        INPUT_PATH
    )

    required_columns = [
        "distribution",
        "relative_interval_width",
        "covered",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[
            "relative_interval_width",
            "covered",
        ]
    )

    data[
        "covered"
    ] = data[
        "covered"
    ].astype(bool)

    overall_rows = []
    distribution_frames = []

    for threshold in WIDTH_THRESHOLDS:
        overall_rows.append(
            evaluate_threshold(
                data,
                threshold,
            )
        )

        distribution_frames.append(
            evaluate_by_distribution(
                data,
                threshold,
            )
        )

    overall = pd.DataFrame(
        overall_rows
    )

    by_distribution = pd.concat(
        distribution_frames,
        ignore_index=True,
    )

    best_row = (
        overall[
            overall[
                "coverage"
            ].notna()
        ]
        .sort_values(
            [
                "abstention_rate",
                "coverage",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .iloc[0]
    )

    minimum_coverage = 0.90

    qualified = overall[
        overall[
            "coverage"
        ]
        >= minimum_coverage
    ]

    if len(qualified) > 0:
        selected = qualified.sort_values(
            "acceptance_rate",
            ascending=False,
        ).iloc[0]

        selected_threshold = float(
            selected[
                "width_threshold"
            ]
        )
    else:
        selected_threshold = np.nan

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_path = (
        TABLES_DIR
        / "selective_tail_reliability_overall.csv"
    )

    distribution_path = (
        TABLES_DIR
        / "selective_tail_reliability_by_distribution.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    by_distribution.to_csv(
        distribution_path,
        index=False,
    )

    print("=" * 70)
    print(
        "M1.7c - SELECTIVE TAIL RELIABILITY"
    )
    print("=" * 70)

    print()
    print(
        "Coverage versus abstention:"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print()

    print(
        "Distribution-specific results:"
    )

    print(
        by_distribution.to_string(
            index=False
        )
    )

    print()

    if np.isfinite(
        selected_threshold
    ):
        print(
            f"Selected threshold for "
            f">=90% coverage: "
            f"{selected_threshold:.2f}"
        )

        selected_row = overall[
            overall[
                "width_threshold"
            ]
            == selected_threshold
        ].iloc[0]

        print(
            f"Acceptance rate: "
            f"{selected_row['acceptance_rate']:.6f}"
        )

        print(
            f"Abstention rate: "
            f"{selected_row['abstention_rate']:.6f}"
        )

        print(
            f"Coverage: "
            f"{selected_row['coverage']:.6f}"
        )
    else:
        print(
            "No threshold achieved "
            "the 90% coverage target."
        )

    print()

    print(
        f"Overall results saved to: "
        f"{overall_path}"
    )

    print(
        f"Distribution results saved to: "
        f"{distribution_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
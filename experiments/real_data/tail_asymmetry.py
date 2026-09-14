from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "tables"
    / "dynamic_reliable_tail.csv"
)

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "tail_asymmetry.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "tail_asymmetry_summary.csv"
)

REGIME_FILE = (
    ROOT
    / "tables"
    / "tail_asymmetry_regime_summary.csv"
)

REQUIRED_COLUMNS = [
    "date",
    "left_alpha",
    "right_alpha",
    "left_joint_reliability",
    "right_joint_reliability",
    "left_tail_thickness",
    "right_tail_thickness",
    "real_volatility",
]


def load_data():
    data = pd.read_csv(
        INPUT_FILE
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = data[
        REQUIRED_COLUMNS
    ].copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = data.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    return data


def calculate_asymmetry(
    data,
):
    left = data[
        "left_alpha"
    ].to_numpy(
        dtype=float
    )

    right = data[
        "right_alpha"
    ].to_numpy(
        dtype=float
    )

    left_reliability = data[
        "left_joint_reliability"
    ].to_numpy(
        dtype=float
    )

    right_reliability = data[
        "right_joint_reliability"
    ].to_numpy(
        dtype=float
    )

    denominator = (
        left
        + right
    )

    valid = (
        np.isfinite(left)
        & np.isfinite(right)
        & (left > 0)
        & (right > 0)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    signed_asymmetry = np.full(
        len(data),
        np.nan,
        dtype=float,
    )

    signed_asymmetry[
        valid
    ] = (
        right[valid]
        - left[valid]
    ) / denominator[valid]

    magnitude = np.abs(
        signed_asymmetry
    )

    reliability = np.minimum(
        left_reliability,
        right_reliability,
    )

    reliability = np.clip(
        reliability,
        0.0,
        1.0,
    )

    reliability_weighted = (
        magnitude
        * reliability
    )

    left_heavier = (
        signed_asymmetry > 0.10
    )

    right_heavier = (
        signed_asymmetry < -0.10
    )

    approximately_symmetric = (
        magnitude <= 0.10
    )

    regime = np.full(
        len(data),
        "UNKNOWN",
        dtype=object,
    )

    regime[
        approximately_symmetric
        & valid
    ] = "SYMMETRIC"

    regime[
        left_heavier
        & valid
    ] = "LEFT_HEAVIER"

    regime[
        right_heavier
        & valid
    ] = "RIGHT_HEAVIER"

    data[
        "tail_asymmetry"
    ] = signed_asymmetry

    data[
        "tail_asymmetry_magnitude"
    ] = magnitude

    data[
        "asymmetry_reliability"
    ] = reliability

    data[
        "reliability_weighted_asymmetry"
    ] = reliability_weighted

    data[
        "left_tail_heavier"
    ] = signed_asymmetry > 0

    data[
        "right_tail_heavier"
    ] = signed_asymmetry < 0

    data[
        "approximately_symmetric"
    ] = approximately_symmetric

    data[
        "asymmetry_regime"
    ] = regime

    return data


def build_summary(
    data,
):
    valid = data[
        data[
            "tail_asymmetry"
        ].notna()
    ].copy()

    if len(valid) == 0:
        return pd.DataFrame()

    asymmetry = valid[
        "tail_asymmetry"
    ]

    magnitude = valid[
        "tail_asymmetry_magnitude"
    ]

    weighted = valid[
        "reliability_weighted_asymmetry"
    ]

    return pd.DataFrame(
        [
            {
                "observations": len(valid),
                "mean_asymmetry": asymmetry.mean(),
                "median_asymmetry": asymmetry.median(),
                "mean_absolute_asymmetry": magnitude.mean(),
                "median_absolute_asymmetry": magnitude.median(),
                "mean_reliability_weighted_asymmetry": weighted.mean(),
                "median_reliability_weighted_asymmetry": weighted.median(),
                "left_heavier_fraction": (
                    (
                        valid[
                            "asymmetry_regime"
                        ]
                        == "LEFT_HEAVIER"
                    ).mean()
                ),
                "right_heavier_fraction": (
                    (
                        valid[
                            "asymmetry_regime"
                        ]
                        == "RIGHT_HEAVIER"
                    ).mean()
                ),
                "symmetric_fraction": (
                    (
                        valid[
                            "asymmetry_regime"
                        ]
                        == "SYMMETRIC"
                    ).mean()
                ),
                "mean_left_alpha": valid[
                    "left_alpha"
                ].mean(),
                "mean_right_alpha": valid[
                    "right_alpha"
                ].mean(),
                "mean_left_reliability": valid[
                    "left_joint_reliability"
                ].mean(),
                "mean_right_reliability": valid[
                    "right_joint_reliability"
                ].mean(),
                "mean_real_volatility": valid[
                    "real_volatility"
                ].mean(),
            }
        ]
    )


def build_regime_summary(
    data,
):
    valid = data[
        data[
            "asymmetry_regime"
        ].isin(
            [
                "LEFT_HEAVIER",
                "RIGHT_HEAVIER",
                "SYMMETRIC",
            ]
        )
    ].copy()

    rows = []

    regimes = [
        "LEFT_HEAVIER",
        "RIGHT_HEAVIER",
        "SYMMETRIC",
    ]

    for regime in regimes:
        group = valid[
            valid[
                "asymmetry_regime"
            ]
            == regime
        ]

        rows.append(
            {
                "asymmetry_regime": regime,
                "observations": len(group),
                "frequency": (
                    len(group)
                    / len(valid)
                    if len(valid) > 0
                    else np.nan
                ),
                "mean_asymmetry": group[
                    "tail_asymmetry"
                ].mean(),
                "mean_absolute_asymmetry": group[
                    "tail_asymmetry_magnitude"
                ].mean(),
                "mean_reliability_weighted_asymmetry": group[
                    "reliability_weighted_asymmetry"
                ].mean(),
                "mean_left_alpha": group[
                    "left_alpha"
                ].mean(),
                "mean_right_alpha": group[
                    "right_alpha"
                ].mean(),
                "mean_reliability": group[
                    "asymmetry_reliability"
                ].mean(),
                "mean_real_volatility": group[
                    "real_volatility"
                ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    data = load_data()

    data = calculate_asymmetry(
        data
    )

    summary = build_summary(
        data
    )

    regime_summary = (
        build_regime_summary(
            data
        )
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    regime_summary.to_csv(
        REGIME_FILE,
        index=False,
    )

    valid = data[
        data[
            "asymmetry_regime"
        ].isin(
            [
                "LEFT_HEAVIER",
                "RIGHT_HEAVIER",
                "SYMMETRIC",
            ]
        )
    ]

    print(
        f"Observations: {len(data)}"
    )

    print(
        f"Valid asymmetry observations: {len(valid)}"
    )

    print()

    print(
        "Asymmetry summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Mutually exclusive asymmetry regimes:"
    )

    print(
        regime_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Regime frequency sum: "
        f"{regime_summary['frequency'].sum():.6f}"
    )

    print()

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )

    print(
        f"Regime summary saved to: {REGIME_FILE}"
    )


if __name__ == "__main__":
    main()
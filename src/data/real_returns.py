from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def align_market_and_cpi(
    market: pd.DataFrame,
    cpi: pd.DataFrame,
) -> pd.DataFrame:
    market_monthly = market.copy()
    cpi_monthly = cpi.copy()

    market_monthly.index = pd.to_datetime(
        market_monthly.index
    )

    cpi_monthly.index = pd.to_datetime(
        cpi_monthly.index
    )

    market_monthly.index = (
        market_monthly.index
        .to_period("M")
        .to_timestamp("M")
    )

    cpi_monthly.index = (
        cpi_monthly.index
        .to_period("M")
        .to_timestamp("M")
    )

    combined = market_monthly.join(
        cpi_monthly,
        how="inner",
        lsuffix="_market",
        rsuffix="_cpi",
    )

    required = [
        "nominal_return",
        "cpi",
        "monthly_inflation",
        "annual_inflation",
    ]

    missing = [
        column
        for column in required
        if column not in combined.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return combined[
        [
            "Close",
            "nominal_return",
            "cpi",
            "monthly_inflation",
            "annual_inflation",
        ]
    ].dropna()


def calculate_real_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    if np.any(
        1.0 + result["nominal_return"] <= 0
    ):
        raise ValueError(
            "Nominal returns contain values <= -100%."
        )

    if np.any(
        1.0 + result["monthly_inflation"] <= 0
    ):
        raise ValueError(
            "Inflation contains values <= -100%."
        )

    result["real_return"] = (
        (1.0 + result["nominal_return"])
        / (1.0 + result["monthly_inflation"])
        - 1.0
    )

    result["nominal_log_return"] = np.log(
        1.0 + result["nominal_return"]
    )

    result["real_log_return"] = (
        np.log(
            1.0 + result["nominal_return"]
        )
        - np.log(
            1.0 + result["monthly_inflation"]
        )
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result.dropna(
        subset=[
            "real_return",
            "real_log_return",
        ]
    )


def save_real_returns(
    data: pd.DataFrame,
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(path)
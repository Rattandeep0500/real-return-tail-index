from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.market_data import download_market_data, resample_monthly
from src.data.inflation import download_cpi, prepare_monthly_cpi
from src.data.real_returns import (
    align_market_and_cpi,
    calculate_real_returns,
    save_real_returns,
)


TICKER = "^GSPC"
START = "2000-01-01"
END = "2026-09-01"


def main():
    print("=" * 70)
    print("M1.1 — REAL-RETURN DATA ENGINE")
    print("=" * 70)

    print("Downloading market data...")

    market_daily = download_market_data(
        TICKER,
        START,
        END,
    )

    market_monthly = resample_monthly(
        market_daily
    )

    print(
        f"Market observations: {len(market_monthly)}"
    )

    print("Downloading CPI...")

    cpi = download_cpi(
        START,
        END,
    )

    cpi_monthly = prepare_monthly_cpi(
        cpi
    )

    print(
        f"CPI observations: {len(cpi_monthly)}"
    )

    print("Aligning market and inflation data...")

    aligned = align_market_and_cpi(
        market_monthly,
        cpi_monthly,
    )

    print(
        f"Aligned observations: {len(aligned)}"
    )

    real_returns = calculate_real_returns(
        aligned
    )

    output_path = (
        ROOT
        / "data"
        / "processed"
        / "sp500_real_returns.csv"
    )

    save_real_returns(
        real_returns,
        output_path,
    )

    print()
    print(
        f"Date range: "
        f"{real_returns.index.min().date()} "
        f"to "
        f"{real_returns.index.max().date()}"
    )

    print(
        f"Mean nominal return: "
        f"{real_returns['nominal_return'].mean():.6f}"
    )

    print(
        f"Mean real return: "
        f"{real_returns['real_return'].mean():.6f}"
    )

    print(
        f"Std nominal return: "
        f"{real_returns['nominal_return'].std():.6f}"
    )

    print(
        f"Std real return: "
        f"{real_returns['real_return'].std():.6f}"
    )

    print()
    print(
        real_returns[
            [
                "nominal_return",
                "monthly_inflation",
                "annual_inflation",
                "real_return",
            ]
        ].tail(10).to_string()
    )

    print()
    print(
        f"Dataset saved to: {output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
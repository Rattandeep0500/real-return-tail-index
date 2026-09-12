from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def download_market_data(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No market data returned for {ticker}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        raise ValueError(
            "Market dataset does not contain Close."
        )

    data = data[["Close"]].copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="first")]

    data["daily_return"] = data["Close"].pct_change()

    data["log_return"] = np.log(
        data["Close"] / data["Close"].shift(1)
    )

    return data


def resample_monthly(
    data: pd.DataFrame,
) -> pd.DataFrame:
    monthly = (
        data["Close"]
        .resample("ME")
        .last()
        .to_frame()
    )

    monthly["nominal_return"] = (
        monthly["Close"].pct_change()
    )

    return monthly.dropna(
        subset=["nominal_return"]
    )


def save_market_data(
    data: pd.DataFrame,
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(path)
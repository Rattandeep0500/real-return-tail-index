from __future__ import annotations

from pathlib import Path

import pandas as pd


FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=CPIAUCSL"
)


def download_cpi(
    start: str,
    end: str,
) -> pd.DataFrame:
    cpi = pd.read_csv(
        FRED_URL,
        parse_dates=["observation_date"],
    )

    cpi = cpi.rename(
        columns={
            "observation_date": "date",
            "CPIAUCSL": "cpi",
        }
    )

    cpi["date"] = pd.to_datetime(
        cpi["date"]
    )

    cpi = cpi.set_index("date")

    cpi = cpi.loc[
        start:end,
        ["cpi"],
    ]

    cpi["cpi"] = pd.to_numeric(
        cpi["cpi"],
        errors="coerce",
    )

    cpi = cpi.dropna()

    cpi = cpi[
        cpi["cpi"] > 0
    ]

    return cpi.sort_index()


def prepare_monthly_cpi(
    cpi: pd.DataFrame,
) -> pd.DataFrame:
    monthly = cpi.copy()

    monthly["monthly_inflation"] = (
        monthly["cpi"].pct_change()
    )

    monthly["annual_inflation"] = (
        monthly["cpi"].pct_change(12)
    )

    return monthly.dropna(
        subset=["monthly_inflation"]
    )


def save_inflation_data(
    data: pd.DataFrame,
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(path)
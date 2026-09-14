from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "tables"
    / "tail_model_detector_validation.csv"
)

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "tail_validity_auc.csv"
)

VALID_DISTRIBUTIONS = {
    "Pareto",
    "Mixture",
    "Regime-Switch",
    "Volatility-Clustered",
}

INVALID_DISTRIBUTIONS = {
    "Lognormal",
    "Student-t",
    "Truncated-Pareto",
}

SCORE_COLUMNS = [
    "validity_score",
    "mean_validity",
    "tail_validity",
    "model_validity_score",
    "validity",
]


def find_score_column(data):
    for column in SCORE_COLUMNS:
        if column in data.columns:
            return column

    numeric_candidates = [
        column
        for column in data.columns
        if "valid" in column.lower()
    ]

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise ValueError(
        "Could not identify the validity-score column. "
        f"Available columns: {list(data.columns)}"
    )


def main():
    data = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        "distribution"
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    score_column = find_score_column(
        data
    )

    data = data[
        data["distribution"].isin(
            VALID_DISTRIBUTIONS
            | INVALID_DISTRIBUTIONS
        )
    ].copy()

    data[score_column] = pd.to_numeric(
        data[score_column],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "distribution",
            score_column,
        ]
    )

    data["tail_valid"] = (
        data["distribution"]
        .isin(VALID_DISTRIBUTIONS)
        .astype(int)
    )

    y_true = data[
        "tail_valid"
    ].to_numpy(
        dtype=int
    )

    y_score = data[
        score_column
    ].to_numpy(
        dtype=float
    )

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Both valid and invalid classes are required."
        )

    roc_auc = roc_auc_score(
        y_true,
        y_score,
    )

    pr_auc = average_precision_score(
        y_true,
        y_score,
    )

    result = pd.DataFrame(
        [
            {
                "score_column": score_column,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "valid_observations": int(
                    np.sum(y_true == 1)
                ),
                "invalid_observations": int(
                    np.sum(y_true == 0)
                ),
                "total_observations": len(
                    y_true
                ),
            }
        ]
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Score column: {score_column}"
    )

    print(
        f"ROC AUC: {roc_auc:.6f}"
    )

    print(
        f"PR AUC: {pr_auc:.6f}"
    )

    print(
        f"Valid observations: {np.sum(y_true == 1)}"
    )

    print(
        f"Invalid observations: {np.sum(y_true == 0)}"
    )

    print(
        f"Total observations: {len(y_true)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "alpha_hat",
    "bootstrap_std",
    "bootstrap_cv",
    "local_cv",
    "local_slope",
    "local_curvature",
    "k_fraction",
]


def build_features(
    k_values,
    alpha_values,
    bootstrap_std,
    sample_size,
    window_size=15,
):
    k_values = np.asarray(k_values, dtype=float)
    alpha_values = np.asarray(alpha_values, dtype=float)
    bootstrap_std = np.asarray(bootstrap_std, dtype=float)

    local_cv = np.full(len(alpha_values), np.nan)
    local_slope = np.full(len(alpha_values), np.nan)
    local_curvature = np.full(len(alpha_values), np.nan)

    for i in range(window_size - 1, len(alpha_values)):
        window = alpha_values[
            i - window_size + 1:i + 1
        ]

        center = np.mean(window)

        if np.isfinite(center) and center != 0:
            local_cv[i] = (
                np.std(window, ddof=1)
                / abs(center)
            )

        if len(window) >= 3 and np.all(np.isfinite(window)):
            x = np.arange(
                len(window),
                dtype=float,
            )

            coefficients = np.polyfit(
                x,
                window,
                2,
            )

            local_curvature[i] = (
                2.0 * coefficients[0]
            )

            local_slope[i] = (
                2.0
                * coefficients[0]
                * (len(window) - 1)
                + coefficients[1]
            )

    bootstrap_cv = np.divide(
        bootstrap_std,
        np.abs(alpha_values),
        out=np.full_like(
            bootstrap_std,
            np.nan,
        ),
        where=np.abs(alpha_values) > 1e-12,
    )

    k_fraction = (
        k_values / float(sample_size)
    )

    return pd.DataFrame(
        {
            "alpha_hat": alpha_values,
            "bootstrap_std": bootstrap_std,
            "bootstrap_cv": bootstrap_cv,
            "local_cv": local_cv,
            "local_slope": local_slope,
            "local_curvature": local_curvature,
            "k_fraction": k_fraction,
        }
    )


def create_quality_labels(
    alpha_hat,
    true_alpha,
    tolerance=0.10,
):
    alpha_hat = np.asarray(
        alpha_hat,
        dtype=float,
    )

    relative_error = (
        np.abs(alpha_hat - true_alpha)
        / abs(true_alpha)
    )

    return (
        relative_error <= tolerance
    ).astype(int)


def build_calibration_model():
    base_model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3,
    )


def fit_reliability_model(
    feature_frame,
    labels,
):
    features = feature_frame[
        FEATURE_NAMES
    ].copy()

    labels = np.asarray(
        labels,
        dtype=int,
    )

    if len(features) != len(labels):
        raise ValueError(
            "Features and labels must have equal length."
        )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if len(features) == 0:
        raise ValueError(
            "No calibration observations available."
        )

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Calibration data contains only one target class."
        )

    model = build_calibration_model()

    model.fit(
        features,
        labels,
    )

    return model, np.ones(
        len(features),
        dtype=bool,
    )


def predict_reliability(
    model,
    feature_frame,
):
    result = feature_frame.copy()

    features = result[
        FEATURE_NAMES
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result["reliability_probability"] = (
        model.predict_proba(
            features
        )[:, 1]
    )

    return result
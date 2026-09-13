from __future__ import annotations

import numpy as np
import pandas as pd


def compute_nonconformity(
    alpha_hat,
    alpha_true,
):
    alpha_hat = np.asarray(
        alpha_hat,
        dtype=float,
    )

    alpha_true = np.asarray(
        alpha_true,
        dtype=float,
    )

    if alpha_hat.shape != alpha_true.shape:
        raise ValueError(
            "alpha_hat and alpha_true must have identical shapes."
        )

    return (
        np.abs(
            alpha_hat - alpha_true
        )
        / np.maximum(
            np.abs(alpha_true),
            1e-12,
        )
    )


def make_tail_bins(
    alpha_hat,
    n_bins=4,
):
    alpha_hat = np.asarray(
        alpha_hat,
        dtype=float,
    )

    finite = np.isfinite(
        alpha_hat
    )

    if finite.sum() < n_bins:
        return np.zeros(
            len(alpha_hat),
            dtype=int,
        )

    values = alpha_hat[
        finite
    ]

    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.quantile(
        values,
        quantiles,
    )

    edges = np.unique(
        edges
    )

    if len(edges) < 2:
        return np.zeros(
            len(alpha_hat),
            dtype=int,
        )

    bins = np.searchsorted(
        edges[1:-1],
        alpha_hat,
        side="right",
    )

    bins[~finite] = -1

    return bins.astype(int)


def make_stability_bins(
    bootstrap_cv,
    n_bins=3,
):
    values = np.asarray(
        bootstrap_cv,
        dtype=float,
    )

    finite = np.isfinite(
        values
    )

    if finite.sum() < n_bins:
        result = np.zeros(
            len(values),
            dtype=int,
        )
        result[~finite] = -1
        return result

    finite_values = values[
        finite
    ]

    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.quantile(
        finite_values,
        quantiles,
    )

    edges = np.unique(
        edges
    )

    if len(edges) < 2:
        result = np.zeros(
            len(values),
            dtype=int,
        )
        result[~finite] = -1
        return result

    result = np.searchsorted(
        edges[1:-1],
        values,
        side="right",
    )

    result[~finite] = -1

    return result.astype(int)


def fit_conditional_conformal(
    calibration_frame,
    alpha_true_column="true_alpha",
    alpha_hat_column="alpha_hat",
    bootstrap_cv_column="bootstrap_cv",
    miscoverage=0.10,
    alpha_bins=4,
    stability_bins=3,
    min_group_size=30,
):
    if not 0 < miscoverage < 1:
        raise ValueError(
            "miscoverage must be between 0 and 1."
        )

    data = calibration_frame[
        [
            alpha_true_column,
            alpha_hat_column,
            bootstrap_cv_column,
        ]
    ].copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna()

    if len(data) == 0:
        raise ValueError(
            "No valid calibration observations."
        )

    scores = compute_nonconformity(
        data[
            alpha_hat_column
        ].to_numpy(
            dtype=float
        ),
        data[
            alpha_true_column
        ].to_numpy(
            dtype=float
        ),
    )

    alpha_group = make_tail_bins(
        data[
            alpha_hat_column
        ].to_numpy(
            dtype=float
        ),
        alpha_bins,
    )

    stability_group = make_stability_bins(
        data[
            bootstrap_cv_column
        ].to_numpy(
            dtype=float
        ),
        stability_bins,
    )

    groups = (
        alpha_group.astype(str)
        + "_"
        + stability_group.astype(str)
    )

    data = data.reset_index(
        drop=True
    )

    data["score"] = scores
    data["group"] = groups

    global_radius = float(
        np.quantile(
            scores,
            1.0 - miscoverage,
            method="higher",
        )
    )

    group_radii = {}

    for group, subset in data.groupby(
        "group"
    ):
        values = subset[
            "score"
        ].to_numpy(
            dtype=float
        )

        if len(values) < min_group_size:
            continue

        group_radii[group] = float(
            np.quantile(
                values,
                1.0 - miscoverage,
                method="higher",
            )
        )

    return {
        "global_radius": global_radius,
        "group_radii": group_radii,
        "alpha_bins": int(alpha_bins),
        "stability_bins": int(stability_bins),
        "miscoverage": float(miscoverage),
        "min_group_size": int(min_group_size),
    }


def predict_conditional_radius(
    model,
    alpha_hat,
    bootstrap_cv,
):
    alpha_hat = np.asarray(
        alpha_hat,
        dtype=float,
    )

    bootstrap_cv = np.asarray(
        bootstrap_cv,
        dtype=float,
    )

    if alpha_hat.shape != bootstrap_cv.shape:
        raise ValueError(
            "alpha_hat and bootstrap_cv must have identical shapes."
        )

    alpha_group = make_tail_bins(
        alpha_hat,
        model["alpha_bins"],
    )

    stability_group = make_stability_bins(
        bootstrap_cv,
        model["stability_bins"],
    )

    groups = (
        alpha_group.astype(str)
        + "_"
        + stability_group.astype(str)
    )

    radii = np.full(
        len(alpha_hat),
        model["global_radius"],
        dtype=float,
    )

    for i, group in enumerate(
        groups
    ):
        if group in model[
            "group_radii"
        ]:
            radii[i] = model[
                "group_radii"
            ][group]

    return radii


def build_conditional_interval(
    alpha_hat,
    radii,
):
    alpha_hat = np.asarray(
        alpha_hat,
        dtype=float,
    )

    radii = np.asarray(
        radii,
        dtype=float,
    )

    if alpha_hat.shape != radii.shape:
        raise ValueError(
            "alpha_hat and radii must have identical shapes."
        )

    lower = (
        alpha_hat
        / (1.0 + radii)
    )

    upper = np.where(
        radii < 1.0,
        alpha_hat
        / (1.0 - radii),
        np.inf,
    )

    return lower, upper


def conditional_certificate(
    alpha_hat,
    bootstrap_cv,
    model,
    max_relative_width=1.0,
):
    radii = predict_conditional_radius(
        model,
        alpha_hat,
        bootstrap_cv,
    )

    lower, upper = (
        build_conditional_interval(
            alpha_hat,
            radii,
        )
    )

    center = (
        lower + upper
    ) / 2.0

    width = (
        upper - lower
    )

    relative_width = np.divide(
        width,
        np.maximum(
            np.abs(center),
            1e-12,
        ),
    )

    certificate = np.exp(
        -relative_width
        / max(
            max_relative_width,
            1e-12,
        )
    )

    certificate[
        ~np.isfinite(certificate)
    ] = 0.0

    certificate = np.clip(
        certificate,
        0.0,
        1.0,
    )

    abstain = (
        ~np.isfinite(
            relative_width
        )
        | (
            relative_width
            > max_relative_width
        )
    )

    return pd.DataFrame(
        {
            "alpha_hat": alpha_hat,
            "conformal_radius": radii,
            "conformal_lower": lower,
            "conformal_upper": upper,
            "relative_interval_width": relative_width,
            "certificate_score": certificate,
            "abstain": abstain,
        }
    )


def evaluate_conditional_coverage(
    alpha_true,
    alpha_hat,
    bootstrap_cv,
    model,
):
    certificate = conditional_certificate(
        alpha_hat,
        bootstrap_cv,
        model,
        max_relative_width=np.inf,
    )

    alpha_true = np.asarray(
        alpha_true,
        dtype=float,
    )

    lower = certificate[
        "conformal_lower"
    ].to_numpy(
        dtype=float
    )

    upper = certificate[
        "conformal_upper"
    ].to_numpy(
        dtype=float
    )

    covered = (
        alpha_true >= lower
    ) & (
        alpha_true <= upper
    )

    valid = np.isfinite(
        alpha_true
    ) & np.isfinite(
        lower
    )

    return {
        "coverage": float(
            np.mean(
                covered[valid]
            )
        ),
        "n": int(
            valid.sum()
        ),
    }
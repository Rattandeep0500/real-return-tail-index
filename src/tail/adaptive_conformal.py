from __future__ import annotations

import numpy as np
import pandas as pd


def relative_nonconformity(
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


def make_quantile_edges(
    values,
    n_bins,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    valid = np.isfinite(
        values
    )

    if valid.sum() == 0:
        return np.array(
            [-np.inf, np.inf],
            dtype=float,
        )

    if valid.sum() < n_bins:
        return np.array(
            [-np.inf, np.inf],
            dtype=float,
        )

    edges = np.quantile(
        values[valid],
        np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        ),
    )

    edges = np.unique(
        edges
    )

    if len(edges) < 2:
        return np.array(
            [-np.inf, np.inf],
            dtype=float,
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def apply_quantile_bins(
    values,
    edges,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    groups = np.full(
        len(values),
        -1,
        dtype=int,
    )

    valid = np.isfinite(
        values
    )

    if len(edges) < 2:
        groups[valid] = 0
        return groups

    groups[valid] = (
        np.searchsorted(
            edges[1:-1],
            values[valid],
            side="right",
        )
    )

    return groups


def make_k_fraction(
    k,
    sample_size,
):
    k = np.asarray(
        k,
        dtype=float,
    )

    sample_size = np.asarray(
        sample_size,
        dtype=float,
    )

    return np.divide(
        k,
        sample_size,
        out=np.full_like(
            k,
            np.nan,
        ),
        where=sample_size > 0,
    )


def build_groups_from_edges(
    alpha_hat,
    bootstrap_cv,
    k,
    sample_size,
    alpha_edges,
    stability_edges,
    k_edges,
):
    alpha_group = apply_quantile_bins(
        alpha_hat,
        alpha_edges,
    )

    stability_group = apply_quantile_bins(
        bootstrap_cv,
        stability_edges,
    )

    k_fraction = make_k_fraction(
        k,
        sample_size,
    )

    k_group = apply_quantile_bins(
        k_fraction,
        k_edges,
    )

    groups_3d = (
        alpha_group.astype(str)
        + "_"
        + stability_group.astype(str)
        + "_"
        + k_group.astype(str)
    )

    groups_2d = (
        alpha_group.astype(str)
        + "_"
        + stability_group.astype(str)
    )

    return (
        groups_3d,
        groups_2d,
    )


def conformal_quantile(
    scores,
    miscoverage,
):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    scores = scores[
        np.isfinite(scores)
        & (scores >= 0)
    ]

    if len(scores) == 0:
        return np.nan

    if not 0 < miscoverage < 1:
        raise ValueError(
            "miscoverage must be between 0 and 1."
        )

    n = len(scores)

    rank = int(
        np.ceil(
            (n + 1)
            * (1.0 - miscoverage)
        )
    )

    rank = max(
        1,
        min(
            rank,
            n,
        ),
    )

    return float(
        np.sort(scores)[
            rank - 1
        ]
    )


def fit_adaptive_conformal(
    calibration_frame,
    alpha_true_column="true_alpha",
    alpha_hat_column="alpha_hat",
    bootstrap_cv_column="bootstrap_cv",
    k_column="k",
    sample_size_column="sample_size",
    miscoverage=0.10,
    alpha_bins=4,
    stability_bins=3,
    k_bins=4,
    min_group_size=40,
):
    required = [
        alpha_true_column,
        alpha_hat_column,
        bootstrap_cv_column,
        k_column,
        sample_size_column,
    ]

    missing = [
        column
        for column in required
        if column not in calibration_frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = calibration_frame[
        required
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

    scores = relative_nonconformity(
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

    alpha_edges = make_quantile_edges(
        data[
            alpha_hat_column
        ].to_numpy(
            dtype=float
        ),
        alpha_bins,
    )

    stability_edges = make_quantile_edges(
        data[
            bootstrap_cv_column
        ].to_numpy(
            dtype=float
        ),
        stability_bins,
    )

    k_fraction = make_k_fraction(
        data[
            k_column
        ].to_numpy(
            dtype=float
        ),
        data[
            sample_size_column
        ].to_numpy(
            dtype=float
        ),
    )

    k_edges = make_quantile_edges(
        k_fraction,
        k_bins,
    )

    groups_3d, groups_2d = (
        build_groups_from_edges(
            data[
                alpha_hat_column
            ].to_numpy(
                dtype=float
            ),
            data[
                bootstrap_cv_column
            ].to_numpy(
                dtype=float
            ),
            data[
                k_column
            ].to_numpy(
                dtype=float
            ),
            data[
                sample_size_column
            ].to_numpy(
                dtype=float
            ),
            alpha_edges,
            stability_edges,
            k_edges,
        )
    )

    data = data.reset_index(
        drop=True
    )

    data["score"] = scores
    data["group_3d"] = groups_3d
    data["group_2d"] = groups_2d

    global_radius = conformal_quantile(
        scores,
        miscoverage,
    )

    radii_3d = {}
    radii_2d = {}
    group_sizes_3d = {}
    group_sizes_2d = {}

    for group, subset in data.groupby(
        "group_3d"
    ):
        values = subset[
            "score"
        ].to_numpy(
            dtype=float
        )

        group_sizes_3d[group] = len(
            values
        )

        if len(values) >= min_group_size:
            radii_3d[group] = (
                conformal_quantile(
                    values,
                    miscoverage,
                )
            )

    for group, subset in data.groupby(
        "group_2d"
    ):
        values = subset[
            "score"
        ].to_numpy(
            dtype=float
        )

        group_sizes_2d[group] = len(
            values
        )

        if len(values) >= min_group_size:
            radii_2d[group] = (
                conformal_quantile(
                    values,
                    miscoverage,
                )
            )

    return {
        "global_radius": float(
            global_radius
        ),
        "radii_3d": radii_3d,
        "radii_2d": radii_2d,
        "group_sizes_3d": group_sizes_3d,
        "group_sizes_2d": group_sizes_2d,
        "alpha_edges": alpha_edges,
        "stability_edges": stability_edges,
        "k_edges": k_edges,
        "alpha_bins": int(
            alpha_bins
        ),
        "stability_bins": int(
            stability_bins
        ),
        "k_bins": int(
            k_bins
        ),
        "miscoverage": float(
            miscoverage
        ),
        "min_group_size": int(
            min_group_size
        ),
    }


def predict_adaptive_radius(
    model,
    alpha_hat,
    bootstrap_cv,
    k,
    sample_size,
):
    alpha_hat = np.asarray(
        alpha_hat,
        dtype=float,
    )

    bootstrap_cv = np.asarray(
        bootstrap_cv,
        dtype=float,
    )

    k = np.asarray(
        k,
        dtype=float,
    )

    sample_size = np.asarray(
        sample_size,
        dtype=float,
    )

    if not (
        alpha_hat.shape
        == bootstrap_cv.shape
        == k.shape
        == sample_size.shape
    ):
        raise ValueError(
            "All prediction arrays must have identical shapes."
        )

    groups_3d, groups_2d = (
        build_groups_from_edges(
            alpha_hat,
            bootstrap_cv,
            k,
            sample_size,
            model["alpha_edges"],
            model["stability_edges"],
            model["k_edges"],
        )
    )

    radii = np.full(
        len(alpha_hat),
        model["global_radius"],
        dtype=float,
    )

    source = np.full(
        len(alpha_hat),
        "global",
        dtype=object,
    )

    for i in range(
        len(alpha_hat)
    ):
        group_3d = groups_3d[i]
        group_2d = groups_2d[i]

        if group_3d in model[
            "radii_3d"
        ]:
            radii[i] = model[
                "radii_3d"
            ][group_3d]

            source[i] = "3d"

        elif group_2d in model[
            "radii_2d"
        ]:
            radii[i] = model[
                "radii_2d"
            ][group_2d]

            source[i] = "2d"

    return radii, source


def build_adaptive_interval(
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

    lower = np.divide(
        alpha_hat,
        1.0 + radii,
    )

    upper = np.where(
        radii < 1.0,
        alpha_hat
        / (1.0 - radii),
        np.inf,
    )

    return lower, upper


def build_adaptive_certificate(
    alpha_hat,
    bootstrap_cv,
    k,
    sample_size,
    model,
    max_relative_width=1.0,
):
    radii, source = (
        predict_adaptive_radius(
            model,
            alpha_hat,
            bootstrap_cv,
            k,
            sample_size,
        )
    )

    lower, upper = (
        build_adaptive_interval(
            alpha_hat,
            radii,
        )
    )

    width = (
        upper - lower
    )

    center = (
        lower + upper
    ) / 2.0

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
        ~np.isfinite(
            certificate
        )
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
            "radius_source": source,
            "conformal_lower": lower,
            "conformal_upper": upper,
            "relative_interval_width": relative_width,
            "certificate_score": certificate,
            "abstain": abstain,
        }
    )


def evaluate_coverage(
    alpha_true,
    certificate,
):
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

    valid = (
        np.isfinite(alpha_true)
        & np.isfinite(lower)
        & np.isfinite(upper)
    )

    if not np.any(valid):
        return {
            "coverage": np.nan,
            "n": 0,
        }

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
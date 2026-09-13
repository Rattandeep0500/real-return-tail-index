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


def conformal_radius(
    scores,
    miscoverage=0.10,
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
        raise ValueError(
            "No valid calibration scores."
        )

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
        min(rank, n),
    )

    sorted_scores = np.sort(
        scores
    )

    return float(
        sorted_scores[rank - 1]
    )


def calibrate_relative_conformal(
    alpha_hat,
    alpha_true,
    miscoverage=0.10,
):
    scores = relative_nonconformity(
        alpha_hat,
        alpha_true,
    )

    radius = conformal_radius(
        scores,
        miscoverage,
    )

    return {
        "radius": radius,
        "scores": scores,
        "n_calibration": len(scores),
        "coverage_target": 1.0 - miscoverage,
    }


def relative_prediction_interval(
    alpha_hat,
    radius,
):
    alpha_hat = np.asarray(
        alpha_hat,
        dtype=float,
    )

    if radius < 0:
        raise ValueError(
            "radius must be non-negative."
        )

    lower = (
        alpha_hat
        / (1.0 + radius)
    )

    if radius < 1.0:
        upper = (
            alpha_hat
            / (1.0 - radius)
        )
    else:
        upper = np.full_like(
            alpha_hat,
            np.inf,
        )

    return lower, upper


def interval_relative_width(
    lower,
    upper,
):
    lower = np.asarray(
        lower,
        dtype=float,
    )

    upper = np.asarray(
        upper,
        dtype=float,
    )

    center = (
        (lower + upper)
        / 2.0
    )

    width = (
        upper - lower
    )

    return np.divide(
        width,
        np.maximum(
            np.abs(center),
            1e-12,
        ),
    )


def certificate_score(
    lower,
    upper,
    max_relative_width=1.0,
):
    width = interval_relative_width(
        lower,
        upper,
    )

    score = np.exp(
        -width
        / max(
            max_relative_width,
            1e-12,
        )
    )

    score[
        ~np.isfinite(score)
    ] = 0.0

    return np.clip(
        score,
        0.0,
        1.0,
    )


def abstention_mask(
    lower,
    upper,
    max_relative_width=1.0,
):
    width = interval_relative_width(
        lower,
        upper,
    )

    return (
        ~np.isfinite(width)
        | (
            width
            > max_relative_width
        )
    )


def build_conformal_certificate(
    alpha_hat,
    radius,
    max_relative_width=1.0,
):
    alpha_hat = np.asarray(
        alpha_hat,
        dtype=float,
    )

    lower, upper = (
        relative_prediction_interval(
            alpha_hat,
            radius,
        )
    )

    width = interval_relative_width(
        lower,
        upper,
    )

    score = certificate_score(
        lower,
        upper,
        max_relative_width,
    )

    abstain = abstention_mask(
        lower,
        upper,
        max_relative_width,
    )

    return pd.DataFrame(
        {
            "alpha_hat": alpha_hat,
            "conformal_lower": lower,
            "conformal_upper": upper,
            "relative_interval_width": width,
            "certificate_score": score,
            "abstain": abstain,
        }
    )


def evaluate_coverage(
    alpha_true,
    lower,
    upper,
):
    alpha_true = np.asarray(
        alpha_true,
        dtype=float,
    )

    lower = np.asarray(
        lower,
        dtype=float,
    )

    upper = np.asarray(
        upper,
        dtype=float,
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

    covered = (
        alpha_true[valid]
        >= lower[valid]
    ) & (
        alpha_true[valid]
        <= upper[valid]
    )

    return {
        "coverage": float(
            np.mean(covered)
        ),
        "n": int(
            np.sum(valid)
        ),
    }
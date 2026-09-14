from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    ROOT
    / "tables"
    / "temporal_conformal_reliability.csv"
)

SUMMARY_FILE = (
    ROOT
    / "tables"
    / "temporal_conformal_reliability_summary.csv"
)

SEED = 20260914
REPLICATES = 60
N_TOTAL = 700
WINDOW_SIZE = 100
K = 20
CALIBRATION_SIZE = 300
TARGET_COVERAGE = 0.90
BLOCK_SIZE = 6
ADAPTIVE_BINS = 3
MIN_GROUP_SIZE = 20


def hill_estimate(sample, k):
    sample = np.asarray(
        sample,
        dtype=float,
    )

    sample = sample[
        np.isfinite(sample)
        & (sample > 0)
    ]

    if len(sample) <= k:
        return np.nan

    sample = np.sort(
        sample
    )[::-1]

    threshold = sample[k]

    if threshold <= 0:
        return np.nan

    values = sample[:k]

    logs = np.log(
        values / threshold
    )

    denominator = np.sum(
        logs
    )

    if denominator <= 0:
        return np.nan

    return float(
        k / denominator
    )


def conformal_quantile(
    scores,
    coverage,
):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    scores = scores[
        np.isfinite(scores)
    ]

    n = len(scores)

    if n == 0:
        return np.nan

    scores = np.sort(
        scores
    )

    rank = int(
        np.ceil(
            (n + 1)
            * coverage
        )
    )

    rank = min(
        max(
            rank,
            1,
        ),
        n,
    )

    return float(
        scores[
            rank - 1
        ]
    )


def make_quantile_edges(
    values,
    bins,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < bins:
        return None

    edges = np.quantile(
        values,
        np.linspace(
            0.0,
            1.0,
            bins + 1,
        ),
    )

    edges = np.unique(
        edges
    )

    if len(edges) < 2:
        return None

    return edges


def assign_bin(
    value,
    edges,
):
    if edges is None:
        return 0

    if not np.isfinite(value):
        return 0

    index = (
        np.searchsorted(
            edges,
            value,
            side="right",
        )
        - 1
    )

    return int(
        np.clip(
            index,
            0,
            len(edges) - 2,
        )
    )


def generate_alpha_iid(
    rng,
    n,
):
    return np.full(
        n,
        3.0,
    )


def generate_alpha_persistent(
    rng,
    n,
):
    alpha = np.empty(
        n
    )

    alpha[0] = 3.0

    for t in range(
        1,
        n,
    ):
        alpha[t] = (
            0.94
            * alpha[t - 1]
            + 0.18
            + rng.normal(
                0.0,
                0.18,
            )
        )

    return np.clip(
        alpha,
        1.5,
        5.5,
    )


def generate_alpha_regime_switch(
    rng,
    n,
):
    alpha = np.empty(
        n
    )

    state = 0

    values = [
        2.2,
        4.2,
    ]

    persistence = 0.97

    for t in range(
        n
    ):
        if t > 0:
            if rng.uniform() > persistence:
                state = 1 - state

        alpha[t] = values[
            state
        ]

    return alpha


def generate_volatility(
    rng,
    n,
    clustered,
):
    if not clustered:
        return np.ones(
            n
        )

    log_sigma = np.empty(
        n
    )

    log_sigma[0] = 0.0

    for t in range(
        1,
        n,
    ):
        log_sigma[t] = (
            0.93
            * log_sigma[t - 1]
            + rng.normal(
                0.0,
                0.14,
            )
        )

    return np.exp(
        log_sigma
    )


def generate_series(
    rng,
    regime,
    n,
):
    if regime == "IID-Pareto":
        alpha = generate_alpha_iid(
            rng,
            n,
        )

        volatility = generate_volatility(
            rng,
            n,
            False,
        )

    elif regime == "Persistent-Tail":
        alpha = generate_alpha_persistent(
            rng,
            n,
        )

        volatility = generate_volatility(
            rng,
            n,
            False,
        )

    elif regime == "Regime-Switch":
        alpha = generate_alpha_regime_switch(
            rng,
            n,
        )

        volatility = generate_volatility(
            rng,
            n,
            False,
        )

    elif regime == "Persistent-Tail-Volatility":
        alpha = generate_alpha_persistent(
            rng,
            n,
        )

        volatility = generate_volatility(
            rng,
            n,
            True,
        )

    else:
        raise ValueError(
            f"Unknown regime: {regime}"
        )

    u = rng.uniform(
        0.0,
        1.0,
        n,
    )

    magnitudes = (
        (1.0 - u)
        ** (
            -1.0
            / alpha
        )
    )

    magnitudes = (
        magnitudes
        * volatility
    )

    return (
        magnitudes,
        alpha,
    )


def build_forecasts(
    magnitudes,
    alpha_true,
):
    forecasts = []
    targets = []
    times = []

    start = WINDOW_SIZE

    stop = len(
        magnitudes
    ) - 1

    for t in range(
        start,
        stop,
    ):
        window = magnitudes[
            t - WINDOW_SIZE:t
        ]

        estimate = hill_estimate(
            window,
            K,
        )

        target = alpha_true[
            t + 1
        ]

        if not np.isfinite(
            estimate
        ):
            continue

        if not np.isfinite(
            target
        ):
            continue

        forecasts.append(
            estimate
        )

        targets.append(
            target
        )

        times.append(
            t
        )

    return (
        np.asarray(
            forecasts,
            dtype=float,
        ),
        np.asarray(
            targets,
            dtype=float,
        ),
        np.asarray(
            times,
            dtype=int,
        ),
    )


def get_scores(
    forecasts,
    targets,
):
    return np.abs(
        targets
        - forecasts
    )


def global_radius(
    forecasts,
    targets,
):
    scores = get_scores(
        forecasts,
        targets,
    )

    return conformal_quantile(
        scores,
        TARGET_COVERAGE,
    )


def fit_adaptive_calibration(
    forecasts,
    targets,
):
    scores = get_scores(
        forecasts,
        targets,
    )

    edges = make_quantile_edges(
        forecasts,
        ADAPTIVE_BINS,
    )

    global_radius_value = conformal_quantile(
        scores,
        TARGET_COVERAGE,
    )

    radii = {}

    if edges is not None:
        groups = np.array(
            [
                assign_bin(
                    value,
                    edges,
                )
                for value in forecasts
            ]
        )

        for group_id in range(
            len(edges) - 1
        ):
            group_scores = scores[
                groups == group_id
            ]

            if len(
                group_scores
            ) >= MIN_GROUP_SIZE:
                radii[group_id] = (
                    conformal_quantile(
                        group_scores,
                        TARGET_COVERAGE,
                    )
                )

    return {
        "global_radius": global_radius_value,
        "edges": edges,
        "radii": radii,
    }


def adaptive_radii_for_test(
    forecasts,
    fitted,
):
    global_radius_value = fitted[
        "global_radius"
    ]

    edges = fitted[
        "edges"
    ]

    radii = []

    for forecast in forecasts:
        group_id = assign_bin(
            forecast,
            edges,
        )

        radius = fitted[
            "radii"
        ].get(
            group_id,
            global_radius_value,
        )

        radii.append(
            radius
        )

    return np.asarray(
        radii,
        dtype=float,
    )


def block_radius(
    forecasts,
    targets,
):
    scores = get_scores(
        forecasts,
        targets,
    )

    usable = (
        len(scores)
        // BLOCK_SIZE
    ) * BLOCK_SIZE

    if usable < BLOCK_SIZE:
        return np.nan

    blocks = scores[
        :usable
    ].reshape(
        -1,
        BLOCK_SIZE,
    )

    block_maxima = np.max(
        blocks,
        axis=1,
    )

    return conformal_quantile(
        block_maxima,
        TARGET_COVERAGE,
    )


def evaluate_interval(
    forecasts,
    targets,
    radii,
):
    radii = np.asarray(
        radii,
        dtype=float,
    )

    lower = (
        forecasts
        - radii
    )

    upper = (
        forecasts
        + radii
    )

    covered = (
        (targets >= lower)
        & (targets <= upper)
    )

    return {
        "coverage": float(
            np.mean(
                covered
            )
        ),
        "mean_width": float(
            np.mean(
                2.0 * radii
            )
        ),
        "median_width": float(
            np.median(
                2.0 * radii
            )
        ),
        "mean_radius": float(
            np.mean(
                radii
            )
        ),
        "observations": int(
            len(targets)
        ),
    }


def evaluate_methods(
    forecasts,
    targets,
    calibration_end,
):
    calibration_forecasts = forecasts[
        :calibration_end
    ]

    calibration_targets = targets[
        :calibration_end
    ]

    test_forecasts = forecasts[
        calibration_end:
    ]

    test_targets = targets[
        calibration_end:
    ]

    global_radius_value = global_radius(
        calibration_forecasts,
        calibration_targets,
    )

    global_result = evaluate_interval(
        test_forecasts,
        test_targets,
        np.full(
            len(test_targets),
            global_radius_value,
        ),
    )

    adaptive_fit = fit_adaptive_calibration(
        calibration_forecasts,
        calibration_targets,
    )

    adaptive_radii = adaptive_radii_for_test(
        test_forecasts,
        adaptive_fit,
    )

    adaptive_result = evaluate_interval(
        test_forecasts,
        test_targets,
        adaptive_radii,
    )

    block_radius_value = block_radius(
        calibration_forecasts,
        calibration_targets,
    )

    block_result = evaluate_interval(
        test_forecasts,
        test_targets,
        np.full(
            len(test_targets),
            block_radius_value,
        ),
    )

    global_result[
        "radius"
    ] = global_radius_value

    adaptive_result[
        "radius"
    ] = float(
        np.mean(
            adaptive_radii
        )
    )

    block_result[
        "radius"
    ] = block_radius_value

    return {
        "Global": global_result,
        "Adaptive": adaptive_result,
        "Block": block_result,
    }


def main():
    rng = np.random.default_rng(
        SEED
    )

    regimes = [
        "IID-Pareto",
        "Persistent-Tail",
        "Regime-Switch",
        "Persistent-Tail-Volatility",
    ]

    rows = []

    for regime in regimes:
        for replicate in range(
            REPLICATES
        ):
            magnitudes, alpha_true = generate_series(
                rng,
                regime,
                N_TOTAL,
            )

            forecasts, targets, times = build_forecasts(
                magnitudes,
                alpha_true,
            )

            if len(forecasts) <= (
                CALIBRATION_SIZE
                + 20
            ):
                continue

            results = evaluate_methods(
                forecasts,
                targets,
                CALIBRATION_SIZE,
            )

            test_times = times[
                CALIBRATION_SIZE:
            ]

            test_true_alpha = alpha_true[
                test_times + 1
            ]

            test_forecasts = forecasts[
                CALIBRATION_SIZE:
            ]

            nonstationary_mask = (
                np.abs(
                    test_true_alpha
                    - 3.0
                )
                > 0.25
            )

            for method, result in results.items():
                radius = result[
                    "radius"
                ]

                lower = (
                    test_forecasts
                    - radius
                )

                upper = (
                    test_forecasts
                    + radius
                )

                subset_coverage = np.nan

                if np.sum(
                    nonstationary_mask
                ) > 0:
                    subset_covered = (
                        (
                            test_true_alpha[
                                nonstationary_mask
                            ]
                            >= lower[
                                nonstationary_mask
                            ]
                        )
                        & (
                            test_true_alpha[
                                nonstationary_mask
                            ]
                            <= upper[
                                nonstationary_mask
                            ]
                        )
                    )

                    subset_coverage = float(
                        np.mean(
                            subset_covered
                        )
                    )

                rows.append(
                    {
                        "regime": regime,
                        "replicate": replicate,
                        "method": method,
                        "coverage": result[
                            "coverage"
                        ],
                        "coverage_gap": (
                            result[
                                "coverage"
                            ]
                            - TARGET_COVERAGE
                        ),
                        "mean_width": result[
                            "mean_width"
                        ],
                        "median_width": result[
                            "median_width"
                        ],
                        "mean_radius": result[
                            "mean_radius"
                        ],
                        "test_observations": result[
                            "observations"
                        ],
                        "nonstationary_subset_coverage": subset_coverage,
                    }
                )

    result = pd.DataFrame(
        rows
    )

    if len(result) == 0:
        raise RuntimeError(
            "No validation results were produced."
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = (
        result
        .groupby(
            [
                "regime",
                "method",
            ]
        )
        .agg(
            observations=(
                "coverage",
                "size",
            ),
            mean_coverage=(
                "coverage",
                "mean",
            ),
            std_coverage=(
                "coverage",
                "std",
            ),
            mean_coverage_gap=(
                "coverage_gap",
                "mean",
            ),
            mean_width=(
                "mean_width",
                "mean",
            ),
            median_width=(
                "median_width",
                "mean",
            ),
            mean_radius=(
                "mean_radius",
                "mean",
            ),
            mean_nonstationary_coverage=(
                "nonstationary_subset_coverage",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    method_coverage = (
        result
        .groupby(
            "method"
        )[
            "coverage"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    method_abs_gap = (
        result
        .groupby(
            "method"
        )[
            "coverage_gap"
        ]
        .apply(
            lambda values: np.mean(
                np.abs(
                    values
                )
            )
        )
        .sort_values()
    )

    print(
        f"Replicates per regime: {REPLICATES}"
    )

    print(
        f"Total result rows: {len(result)}"
    )

    print()

    print(
        "Temporal conformal reliability summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Mean coverage by method:"
    )

    print(
        method_coverage.to_string()
    )

    print()

    print(
        "Mean absolute coverage gap by method:"
    )

    print(
        method_abs_gap.to_string()
    )

    print()

    print(
        f"Detailed results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
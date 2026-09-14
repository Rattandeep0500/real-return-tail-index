from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "sp500_real_returns.csv"
STABILITY_FILE = ROOT / "tables" / "tail_stability_surface_summary.csv"
OUTPUT_FILE = ROOT / "tables" / "tail_model_competition.csv"
SUMMARY_FILE = ROOT / "tables" / "tail_model_competition_summary.csv"

WINDOW_SIZE = 120
MIN_K = 20
MAX_K = 60
K_STEP = 5
AIC_MARGIN_THRESHOLD = 2.0


def clean_values(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x) & (x > 0)]


def hill_alpha(tail, threshold):
    tail = clean_values(tail)

    if len(tail) < 5 or threshold <= 0:
        return np.nan

    denominator = np.sum(np.log(tail / threshold))

    if denominator <= 0:
        return np.nan

    return float(len(tail) / denominator)


def empirical_ks(tail, cdf):
    tail = np.sort(clean_values(tail))
    n = len(tail)

    if n < 5:
        return np.nan

    fitted = cdf(tail)

    if not np.all(np.isfinite(fitted)):
        return np.nan

    upper = np.arange(1, n + 1, dtype=float) / n
    lower = np.arange(0, n, dtype=float) / n

    return float(
        max(
            np.max(upper - fitted),
            np.max(fitted - lower),
        )
    )


def fit_pareto(tail, threshold):
    tail = clean_values(tail)
    alpha = hill_alpha(tail, threshold)

    if not np.isfinite(alpha):
        return None

    log_likelihood = (
        len(tail) * np.log(alpha)
        - len(tail) * np.log(threshold)
        - (alpha + 1.0) * np.sum(np.log(tail / threshold))
    )

    aic = 2.0 - 2.0 * log_likelihood

    return {
        "parameters": 1,
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "alpha": float(alpha),
    }


def pareto_cdf(x, threshold, alpha):
    x = np.asarray(x, dtype=float)

    return np.clip(
        1.0 - (x / threshold) ** (-alpha),
        0.0,
        1.0,
    )


def fit_lognormal(tail, threshold):
    tail = clean_values(tail)
    z = np.log(tail)
    lower = np.log(threshold)

    def objective(params):
        mu = params[0]
        sigma = np.exp(params[1])

        if sigma <= 0:
            return np.inf

        log_pdf = stats.norm.logpdf(
            z,
            loc=mu,
            scale=sigma,
        )

        log_survival = stats.norm.logsf(
            lower,
            loc=mu,
            scale=sigma,
        )

        if not np.isfinite(log_survival):
            return np.inf

        return -float(
            np.sum(log_pdf)
            - len(z) * log_survival
        )

    initial = np.array(
        [
            np.mean(z),
            np.log(max(np.std(z, ddof=1), 1e-3)),
        ]
    )

    result = optimize.minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 3000},
    )

    if not result.success:
        return None

    mu = float(result.x[0])
    sigma = float(np.exp(result.x[1]))

    if sigma <= 0 or not np.isfinite(sigma):
        return None

    log_likelihood = -float(result.fun)
    aic = 4.0 - 2.0 * log_likelihood

    return {
        "parameters": 2,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "mu": mu,
        "sigma": sigma,
    }


def lognormal_cdf(x, threshold, mu, sigma):
    x = np.asarray(x, dtype=float)

    lower = np.log(threshold)
    z = np.log(x)

    denominator = stats.norm.sf(
        lower,
        loc=mu,
        scale=sigma,
    )

    if denominator <= 0:
        return np.full(len(x), np.nan)

    numerator = (
        stats.norm.cdf(
            z,
            loc=mu,
            scale=sigma,
        )
        - stats.norm.cdf(
            lower,
            loc=mu,
            scale=sigma,
        )
    )

    return np.clip(
        numerator / denominator,
        0.0,
        1.0,
    )


def fit_student_t(tail, threshold):
    tail = clean_values(tail)

    def objective(params):
        df = np.exp(params[0])
        scale = np.exp(params[1])

        if df <= 1.0 or scale <= 0:
            return np.inf

        log_pdf = stats.t.logpdf(
            tail,
            df=df,
            loc=0.0,
            scale=scale,
        )

        log_survival = stats.t.logsf(
            threshold,
            df=df,
            loc=0.0,
            scale=scale,
        )

        if not np.isfinite(log_survival):
            return np.inf

        return -float(
            np.sum(log_pdf)
            - len(tail) * log_survival
        )

    initial = np.array(
        [
            np.log(4.0),
            np.log(max(np.std(tail, ddof=1), 1e-3)),
        ]
    )

    result = optimize.minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 3000},
    )

    if not result.success:
        return None

    df = float(np.exp(result.x[0]))
    scale = float(np.exp(result.x[1]))

    if df <= 1.0 or scale <= 0:
        return None

    log_likelihood = -float(result.fun)
    aic = 4.0 - 2.0 * log_likelihood

    return {
        "parameters": 2,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "df": df,
        "scale": scale,
    }


def student_t_cdf(x, threshold, df, scale):
    x = np.asarray(x, dtype=float)

    denominator = stats.t.sf(
        threshold,
        df=df,
        loc=0.0,
        scale=scale,
    )

    if denominator <= 0:
        return np.full(len(x), np.nan)

    numerator = (
        stats.t.cdf(
            x,
            df=df,
            loc=0.0,
            scale=scale,
        )
        - stats.t.cdf(
            threshold,
            df=df,
            loc=0.0,
            scale=scale,
        )
    )

    return np.clip(
        numerator / denominator,
        0.0,
        1.0,
    )


def fit_truncated_pareto(tail, threshold):
    tail = clean_values(tail)
    upper = float(np.max(tail))

    if upper <= threshold:
        return None

    upper_ratio = threshold / upper

    def objective(params):
        alpha = np.exp(params[0])

        if alpha <= 0:
            return np.inf

        normalization = 1.0 - upper_ratio ** alpha

        if normalization <= 0:
            return np.inf

        log_density = (
            np.log(alpha)
            - np.log(threshold)
            - (alpha + 1.0) * np.log(tail / threshold)
            - np.log(normalization)
        )

        return -float(np.sum(log_density))

    initial_alpha = hill_alpha(
        tail,
        threshold,
    )

    if not np.isfinite(initial_alpha):
        initial_alpha = 2.0

    initial = np.array(
        [
            np.log(max(initial_alpha, 0.1)),
        ]
    )

    result = optimize.minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 3000},
    )

    if not result.success:
        return None

    alpha = float(np.exp(result.x[0]))

    if alpha <= 0:
        return None

    log_likelihood = -float(result.fun)

    aic = 4.0 - 2.0 * log_likelihood

    return {
        "parameters": 2,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "alpha": alpha,
        "upper": upper,
    }


def truncated_pareto_cdf(
    x,
    threshold,
    alpha,
    upper,
):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, threshold, upper)

    numerator = (
        1.0
        - (x / threshold) ** (-alpha)
    )

    denominator = (
        1.0
        - (upper / threshold) ** (-alpha)
    )

    if denominator <= 0:
        return np.full(len(x), np.nan)

    return np.clip(
        numerator / denominator,
        0.0,
        1.0,
    )


def fit_all_models(tail, threshold):
    fits = {}

    pareto = fit_pareto(
        tail,
        threshold,
    )

    if pareto is not None:
        pareto["ks"] = empirical_ks(
            tail,
            lambda x: pareto_cdf(
                x,
                threshold,
                pareto["alpha"],
            ),
        )
        fits["Pareto"] = pareto

    lognormal = fit_lognormal(
        tail,
        threshold,
    )

    if lognormal is not None:
        lognormal["ks"] = empirical_ks(
            tail,
            lambda x: lognormal_cdf(
                x,
                threshold,
                lognormal["mu"],
                lognormal["sigma"],
            ),
        )
        fits["Lognormal"] = lognormal

    student = fit_student_t(
        tail,
        threshold,
    )

    if student is not None:
        student["ks"] = empirical_ks(
            tail,
            lambda x: student_t_cdf(
                x,
                threshold,
                student["df"],
                student["scale"],
            ),
        )
        fits["Student-t"] = student

    truncated = fit_truncated_pareto(
        tail,
        threshold,
    )

    if truncated is not None:
        truncated["ks"] = empirical_ks(
            tail,
            lambda x: truncated_pareto_cdf(
                x,
                threshold,
                truncated["alpha"],
                truncated["upper"],
            ),
        )
        fits["Truncated-Pareto"] = truncated

    return fits


def select_best_model(fits):
    rows = []

    for model, values in fits.items():
        if (
            np.isfinite(values["aic"])
            and np.isfinite(values["ks"])
        ):
            rows.append(
                {
                    "model": model,
                    "aic": values["aic"],
                    "ks": values["ks"],
                    "log_likelihood": values["log_likelihood"],
                    "parameters": values["parameters"],
                }
            )

    if not rows:
        return None

    table = pd.DataFrame(rows).sort_values(
        "aic"
    ).reset_index(drop=True)

    best = table.iloc[0]

    if len(table) >= 2:
        second = table.iloc[1]

        aic_gap = float(
            second["aic"] - best["aic"]
        )
    else:
        aic_gap = np.nan

    return {
        "selected_model": str(best["model"]),
        "selected_aic": float(best["aic"]),
        "selected_ks": float(best["ks"]),
        "aic_gap": aic_gap,
        "candidate_count": int(len(table)),
        "comparison": table,
    }


def load_stability_reference():
    if not STABILITY_FILE.exists():
        return {}

    stability = pd.read_csv(
        STABILITY_FILE
    )

    if (
        "date" not in stability.columns
        or "stable_k_center" not in stability.columns
    ):
        return {}

    stability["date"] = pd.to_datetime(
        stability["date"]
    )

    return dict(
        zip(
            stability["date"],
            stability["stable_k_center"],
        )
    )


def choose_k(
    sample,
    preferred_k,
):
    n = len(
        clean_values(
            sample
        )
    )

    if n < MIN_K:
        return None

    maximum = min(
        MAX_K,
        n - 1,
    )

    if np.isfinite(preferred_k):
        k = int(
            round(
                preferred_k / K_STEP
            )
            * K_STEP
        )

        if MIN_K <= k <= maximum:
            return k

    candidates = list(
        range(
            MIN_K,
            maximum + 1,
            K_STEP,
        )
    )

    if not candidates:
        return None

    return candidates[
        len(candidates) // 2
    ]


def analyze_side(
    magnitudes,
    preferred_k,
):
    magnitudes = clean_values(
        magnitudes
    )

    if len(magnitudes) < MIN_K:
        return None

    magnitudes = np.sort(
        magnitudes
    )[::-1]

    k = choose_k(
        magnitudes,
        preferred_k,
    )

    if k is None:
        return None

    threshold = float(
        magnitudes[k]
    )

    tail = magnitudes[:k]

    fits = fit_all_models(
        tail,
        threshold,
    )

    selection = select_best_model(
        fits
    )

    if selection is None:
        return None

    return {
        "k": int(k),
        "threshold": threshold,
        "tail_size": int(len(tail)),
        "selected_model": selection[
            "selected_model"
        ],
        "selected_aic": selection[
            "selected_aic"
        ],
        "selected_ks": selection[
            "selected_ks"
        ],
        "aic_gap": selection[
            "aic_gap"
        ],
        "candidate_count": selection[
            "candidate_count"
        ],
        "fits": selection[
            "comparison"
        ],
    }


def build_summary(result):
    rows = []

    for side in ["left", "right"]:
        group = result[
            result["side"] == side
        ].copy()

        if len(group) == 0:
            continue

        selected = (
            group[
                [
                    "date",
                    "selected_model",
                    "aic_gap",
                ]
            ]
            .drop_duplicates()
        )

        counts = (
            selected[
                "selected_model"
            ]
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        rows.append(
            {
                "side": side,
                "windows": len(selected),
                "mean_k": group[
                    "k"
                ].mean(),
                "median_k": group[
                    "k"
                ].median(),
                "mean_selected_ks": group[
                    "selected_ks"
                ].mean(),
                "mean_aic_gap": selected[
                    "aic_gap"
                ].mean(),
                "high_confidence_fraction": (
                    selected[
                        "aic_gap"
                    ]
                    >= AIC_MARGIN_THRESHOLD
                ).mean(),
                "pareto_fraction": counts.get(
                    "Pareto",
                    0.0,
                ),
                "lognormal_fraction": counts.get(
                    "Lognormal",
                    0.0,
                ),
                "student_t_fraction": counts.get(
                    "Student-t",
                    0.0,
                ),
                "truncated_pareto_fraction": counts.get(
                    "Truncated-Pareto",
                    0.0,
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    data = pd.read_csv(
        INPUT_FILE,
        index_col=0,
    )

    data = data.reset_index()

    data = data.rename(
        columns={
            data.columns[0]: "date"
        }
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    if "real_return" not in data.columns:
        raise ValueError(
            "real_return column not found."
        )

    dates = data[
        "date"
    ].to_numpy()

    returns = data[
        "real_return"
    ].to_numpy(
        dtype=float
    )

    stability_map = (
        load_stability_reference()
    )

    rows = []

    total_windows = (
        len(returns)
        - WINDOW_SIZE
        + 1
    )

    for start in range(
        total_windows
    ):
        end = start + WINDOW_SIZE

        window = returns[
            start:end
        ]

        date = pd.Timestamp(
            dates[end - 1]
        )

        preferred_k = stability_map.get(
            date,
            np.nan,
        )

        left = -window[
            window < 0
        ]

        right = window[
            window > 0
        ]

        left_result = analyze_side(
            left,
            preferred_k,
        )

        right_result = analyze_side(
            right,
            preferred_k,
        )

        if left_result is not None:
            for _, fit in left_result[
                "fits"
            ].iterrows():
                rows.append(
                    {
                        "date": date,
                        "side": "left",
                        "k": left_result["k"],
                        "threshold": left_result["threshold"],
                        "tail_size": left_result["tail_size"],
                        "model": fit["model"],
                        "parameters": fit["parameters"],
                        "aic": fit["aic"],
                        "ks": fit["ks"],
                        "log_likelihood": fit["log_likelihood"],
                        "selected_model": left_result["selected_model"],
                        "selected_aic": left_result["selected_aic"],
                        "selected_ks": left_result["selected_ks"],
                        "aic_gap": left_result["aic_gap"],
                        "candidate_count": left_result["candidate_count"],
                    }
                )

        if right_result is not None:
            for _, fit in right_result[
                "fits"
            ].iterrows():
                rows.append(
                    {
                        "date": date,
                        "side": "right",
                        "k": right_result["k"],
                        "threshold": right_result["threshold"],
                        "tail_size": right_result["tail_size"],
                        "model": fit["model"],
                        "parameters": fit["parameters"],
                        "aic": fit["aic"],
                        "ks": fit["ks"],
                        "log_likelihood": fit["log_likelihood"],
                        "selected_model": right_result["selected_model"],
                        "selected_aic": right_result["selected_aic"],
                        "selected_ks": right_result["selected_ks"],
                        "aic_gap": right_result["aic_gap"],
                        "candidate_count": right_result["candidate_count"],
                    }
                )

    result = pd.DataFrame(
        rows
    )

    if len(result) == 0:
        raise RuntimeError(
            "No valid model-comparison results were produced."
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = build_summary(
        result
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    selected = (
        result[
            [
                "date",
                "side",
                "selected_model",
            ]
        ]
        .drop_duplicates()
    )

    print(
        f"Rolling windows: {total_windows}"
    )

    print(
        f"Model-comparison rows: {len(result)}"
    )

    print()

    print(
        "Selected-model distribution:"
    )

    print(
        selected[
            [
                "side",
                "selected_model",
            ]
        ]
        .groupby(
            [
                "side",
                "selected_model",
            ]
        )
        .size()
        .to_string()
    )

    print()

    print(
        "Summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Parameter counts:"
    )

    print(
        result[
            [
                "model",
                "parameters",
            ]
        ]
        .drop_duplicates()
        .sort_values("model")
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
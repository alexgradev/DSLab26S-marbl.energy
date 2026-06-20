"""Diagnostic plots. Every function returns a matplotlib Figure."""
from __future__ import annotations
from typing import Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd
import pathlib
import numpy as np
from src.config import ZoneConfig, ModelConfig
from matplotlib.patches import Patch
from src import evaluation
from sklearn.metrics import ConfusionMatrixDisplay
from src import features
import contextlib, io



# Public column-name conventions used across the plotting module.
HOUR_COLS:   List[str] = [f"h{h:02d}"        for h in range(24)]
PRED_COLS:   List[str] = [f"pred_h{h:02d}"   for h in range(24)]
NAIVE1_COLS: List[str] = [f"naive1_h{h:02d}" for h in range(24)]
NAIVE2_COLS: List[str] = [f"naive2_h{h:02d}" for h in range(24)]



def plot_price_distribution(
    df: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> matplotlib.figure.Figure:
    """Histogram of hourly prices with p0.1 and p99.9 reference lines.

    Title uses zone_cfg.zone, so it is correct for any bidding zone.
    """
    p001 = np.percentile(df[zone_cfg.schema.price].dropna(), 0.1)
    p999 = np.percentile(df[zone_cfg.schema.price].dropna(), 99.9)

    high_outliers = df[df[zone_cfg.schema.price] > p999]
    low_outliers  = df[df[zone_cfg.schema.price] < p001]

    print(f"p0.1 threshold:  {p001:.2f} EUR/MWh")
    print(f"p99.9 threshold: {p999:.2f} EUR/MWh")
    print(f"High outliers (>{p999:.1f}): {len(high_outliers):,}")
    print(f"Low outliers  (<{p001:.1f}):  {len(low_outliers):,}")

    # if len(high_outliers) > 0:
    #     print("\nHigh outliers (first 20):")
    #     print(high_outliers[[zone_cfg.schema.price]].head(20).to_string())
    # if len(low_outliers) > 0:
    #     print("\nLow outliers (first 20):")
    #     print(low_outliers[[zone_cfg.schema.price]].head(20).to_string())

    neg_hours   = (df[zone_cfg.schema.price] < 0).sum()
    total_hours = df[zone_cfg.schema.price].notna().sum()

    print(f"\nNegative price hours: {neg_hours:,}  ({100 * neg_hours / total_hours:.2f}% of total)")


    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df[zone_cfg.schema.price].dropna(), bins=120, color="steelblue", alpha=0.7)
    ax.axvline(p001, color="darkorange", linewidth=1.5, linestyle="--", label=f"p0.1 = {p001:.1f}")
    ax.axvline(p999, color="crimson",    linewidth=1.5, linestyle="--", label=f"p99.9 = {p999:.1f}")
    ax.set_xlabel("Price (EUR/MWh)")
    ax.set_ylabel("Count")
    ax.set_title("DK1 hourly price distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/price_distribution.png", dpi=120)
    plt.show()
    print("Saved: results/price_distribution.png")




def plot_feature_summary(
    df_daily: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> matplotlib.figure.Figure:
    """2x3 grid: lagged price/std/TTF series, neg-streak histogram,
    spark-spread series, wind vs price scatter."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0, 0]
    ax.plot(df_daily.index, df_daily["price_mean_lag1"], linewidth=0.6, color="steelblue")
    ax.set_title("price_mean_lag1 over time"); ax.set_xlabel("Date"); ax.set_ylabel("EUR/MWh")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df_daily.index, df_daily["price_std_lag1"], linewidth=0.6, color="darkorange")
    ax.set_title("price_std_lag1 over time"); ax.set_xlabel("Date"); ax.set_ylabel("EUR/MWh")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    if "ttf_lag1" in df_daily.columns:
        ax.plot(df_daily.index, df_daily["ttf_lag1"], linewidth=0.6, color="forestgreen")
        ax.set_title("ttf_lag1 over time"); ax.set_xlabel("Date"); ax.set_ylabel("EUR/MWh")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "ttf_lag1 not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("ttf_lag1 (missing)")

    ax = axes[1, 0]
    streak_vals = df_daily["neg_streak_length"].value_counts().sort_index()
    ax.bar(streak_vals.index, streak_vals.values, color="steelblue", edgecolor="white")
    ax.set_title("neg_streak_length distribution")
    ax.set_xlabel("Streak length (days)"); ax.set_ylabel("Count"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if "spark_spread_proxy" in df_daily.columns:
        ax.plot(df_daily.index, df_daily["spark_spread_proxy"], linewidth=0.6, color="crimson")
        ax.set_title("spark_spread_proxy over time"); ax.set_xlabel("Date"); ax.set_ylabel("EUR/MWh")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "spark_spread_proxy not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("spark_spread_proxy (missing)")

    ax = axes[1, 2]
    if "wind_mean_t" in df_daily.columns:
        ax.scatter(df_daily["wind_mean_t"], df_daily["price_mean_lag1"],
                alpha=0.3, s=8, color="steelblue")
        ax.set_title("wind_mean_t vs price_mean_lag1")
        ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Price lag1 (EUR/MWh)"); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "wind_mean_t not available", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("DK1 daily feature summary", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/feature_summary.png", dpi=120)
    plt.show()
    print("Saved: results/feature_summary.png")



def plot_hourly_errors(
    df_predictions: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> matplotlib.figure.Figure:
    """Side-by-side MAE-by-hour bars and mean-signed-error-by-hour bars."""
    mae_by_hour  = np.zeros(24)
    bias_by_hour = np.zeros(24)

    for h in range(24):
        a = df_predictions[f"h{h:02d}"].values
        p = df_predictions[f"pred_h{h:02d}"].values
        mask = ~(np.isnan(a) | np.isnan(p))
        mae_by_hour[h]  = np.abs(a[mask] - p[mask]).mean()
        bias_by_hour[h] = (p[mask] - a[mask]).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.bar(range(24), mae_by_hour, color="steelblue", edgecolor="white")
    ax1.set_title("MAE by hour of day")
    ax1.set_xlabel("Hour (UTC)"); ax1.set_ylabel("MAE (EUR/MWh)")
    ax1.set_xticks(range(24)); ax1.grid(axis="y", alpha=0.3)

    colors = ["darkorange" if b > 0 else "steelblue" for b in bias_by_hour]
    ax2.bar(range(24), bias_by_hour, color=colors, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Mean signed error by hour (+ = over-prediction)")
    ax2.set_xlabel("Hour (UTC)"); ax2.set_ylabel("Mean signed error (EUR/MWh)")
    ax2.set_xticks(range(24)); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{zone_cfg.zone} forecast error by hour of day -- walk-forward test", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/hourly_errors_{zone_cfg.zone}.png", dpi=120)
    plt.show()
    print(f"Saved: results/hourly_errors_{zone_cfg.zone}.png")

    top3 = np.argsort(mae_by_hour)[-3:][::-1]
    print(f"Hours with highest MAE: {[f'h{h:02d} ({mae_by_hour[h]:.2f})' for h in top3]}")



def plot_timeseries_actual_vs_predicted(
    df_predictions: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> matplotlib.figure.Figure:
    """Daily-mean overlay (Actual / Model / Naive 1) with regime strip below."""

    df_predictions["actual_mean"] = df_predictions[HOUR_COLS].mean(axis=1)
    df_predictions["pred_mean"]   = df_predictions[PRED_COLS].mean(axis=1)
    df_predictions["naive1_mean"] = df_predictions[NAIVE1_COLS].mean(axis=1)

    dates = pd.to_datetime(df_predictions["date"])
    regime_colors = {0: "steelblue", 1: "crimson", 2: "forestgreen", 3: "darkorange"}
    regime_labels_plot = {0: "Leaf A", 1: "Leaf B", 2: "C0", 3: "C1"}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [6, 1]}, sharex=True
    )

    ax1.plot(dates, df_predictions["actual_mean"],
            color="black", linewidth=0.8, label="Actual", alpha=0.9)
    ax1.plot(dates, df_predictions["pred_mean"],
            color="steelblue", linewidth=0.8, label="Model", alpha=0.8)
    ax1.plot(dates, df_predictions["naive1_mean"],
            color="darkorange", linewidth=0.6, label="Naive 1 (yesterday)",
            alpha=0.5, linestyle="--")
    ax1.set_ylabel("Daily mean price (EUR/MWh)")
    ax1.set_title(f"Actual vs predicted daily mean price for {zone_cfg.zone} -- walk-forward test")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Colour each day in the strip by its dominant regime.
    prob_cols = ["p_A", "p_B", "p_C0", "p_C1"]
    df_predictions["dominant_regime"] = df_predictions[prob_cols].values.argmax(axis=1)
    for _, row in df_predictions.iterrows():
        d = pd.to_datetime(row["date"])
        ax2.axvspan(d, d + pd.Timedelta(days=1),
                color=regime_colors[row["dominant_regime"]], alpha=0.8)
    ax2.set_yticks([])
    ax2.set_ylabel("Regime", fontsize=8)
    ax2.set_xlabel("Date")
    legend_patches = [Patch(facecolor=regime_colors[l], label=regime_labels_plot[l])
                    for l in range(4)]
    ax2.legend(handles=legend_patches, loc="upper right", ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"results/timeseries_actual_vs_predicted_{zone_cfg.zone}.png", dpi=120)
    plt.show()
    print(f"Saved: results/timeseries_actual_vs_predicted_{zone_cfg.zone}.png")



def plot_regime_profiles(
    df_predictions: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> matplotlib.figure.Figure:
    """2x2 grid: mean 24h actual vs predicted curve per dominant regime."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    hours = list(range(24))

    prob_cols = ["p_A", "p_B", "p_C0", "p_C1"]
    regime_names = {0: "Leaf A (negative)", 1: "Leaf B (spike)", 2: "C0 (normal)", 3: "C1 (normal)"}
    df_predictions["dominant_regime"] = df_predictions[prob_cols].values.argmax(axis=1)

    for idx in range(4):
        ax   = axes[idx]
        name = regime_names[idx]
        mask = df_predictions["dominant_regime"] == idx
        grp  = df_predictions[mask]

        if len(grp) == 0:
            ax.text(0.5, 0.5, "No test days", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(name)
            continue

        act_mean = grp[HOUR_COLS].mean(axis=0).values
        prd_mean = grp[PRED_COLS].mean(axis=0).values
        # act_std  = grp[hour_cols].std(axis=0).values

        # ax.fill_between(hours, act_mean - act_std, act_mean + act_std,
        #                 color="black", alpha=0.08, label="Actual +/-1 std")
        ax.plot(hours, act_mean, color="black", linewidth=1.8, label="Actual mean")
        ax.plot(hours, prd_mean, color="steelblue", linewidth=1.8,
                linestyle="--", label="Predicted mean")
        ax.set_title(f"{name}  (n={len(grp)} days)")
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.legend(fontsize=8)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)

    fig.suptitle(f"Average 24h price profile: actual vs predicted by regime for {zone_cfg.zone}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/regime_profiles_actual_vs_predicted_{zone_cfg.zone}.png", dpi=120)
    plt.show()
    print(f"Saved: results/regime_profiles_actual_vs_predicted_{zone_cfg.zone}.png")



def plot_error_distribution(
    df_predictions: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> matplotlib.figure.Figure:
    """Side-by-side error histogram and rolling 30-day WAPE."""
    # Flatten all hourly prediction errors.


    
    errors_raw = df_predictions[PRED_COLS].values - df_predictions[HOUR_COLS].values
    errors = errors_raw.ravel()
    errors = errors[~np.isnan(errors)]

    # Compute WAPE per day, then smooth with a 30-day rolling window.
    daily_wape_vals = []
    for _, row in df_predictions.iterrows():
        a = row[HOUR_COLS].values.astype(float)
        p = row[PRED_COLS].values.astype(float)
        daily_wape_vals.append(evaluation.wape(a, p))

    daily_wape_series = pd.Series(
        daily_wape_vals,
        index=pd.to_datetime(df_predictions["date"])
    )
    rolling_wape = daily_wape_series.rolling(30, min_periods=5).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.hist(errors, bins=100, color="steelblue", alpha=0.75, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=1.2)
    ax1.axvline(errors.mean(), color="crimson", linewidth=1.5, linestyle="--",
                label=f"Mean bias = {errors.mean():.2f} EUR/MWh")
    ax1.set_xlabel("Prediction error (EUR/MWh)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Distribution of hourly prediction errors for {zone_cfg.zone}")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(daily_wape_series.index, rolling_wape,
            color="steelblue", linewidth=1.2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("WAPE (%)")
    ax2.set_title(f"Rolling 30-day WAPE over time for {zone_cfg.zone}")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/error_distribution_{zone_cfg.zone}.png", dpi=120)
    plt.show()
    print(f"Saved: results/error_distribution_{zone_cfg.zone}.png")

    print(f"Mean error (bias):      {errors.mean():.2f} EUR/MWh")
    print(f"Error std:              {errors.std():.2f} EUR/MWh")
    print(f"Median absolute error:  {np.median(np.abs(errors)):.2f} EUR/MWh")
    print(f"90th pct absolute error:{np.percentile(np.abs(errors), 90):.2f} EUR/MWh")



def plot_regime_probabilities(
    df_predictions: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> matplotlib.figure.Figure:
    """Stacked area chart of p_A / p_B / p_C0 / p_C1 over time."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.stackplot(
        pd.to_datetime(df_predictions["date"]),
        df_predictions["p_A"],
        df_predictions["p_B"],
        df_predictions["p_C0"],
        df_predictions["p_C1"],
        labels=["p_A (Leaf A -- negative)", "p_B (Leaf B -- spike)",
                "p_C0 (normal)", "p_C1 (normal)"],
        colors=["steelblue", "crimson", "forestgreen", "darkorange"],
        alpha=0.8
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_title(f"Classifier regime probabilities over the test period for {zone_cfg.zone}")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/regime_probabilities_{zone_cfg.zone}.png", dpi=120)
    plt.show()
    print(f"Saved: results/regime_probabilities_{zone_cfg.zone}.png")



def plot_confusion_matrix(
    cm: np.ndarray,
    display_labels: List[str],
    title: str,
    cmap: str = "Blues",
) -> matplotlib.figure.Figure:
    """Generic confusion-matrix figure used by both L1 and L2 displays.

    Args:
        cm: 2D array from sklearn.metrics.confusion_matrix.
        display_labels: Class labels in the order of cm rows/cols.
        title: Plot title.
        cmap: Matplotlib colormap name.

    Returns:
        Figure containing the confusion matrix display.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=cmap, ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_single_day_diagnostics(
    date: pd.Timestamp,
    df_predictions: pd.DataFrame,
    fold_models: List[Dict],
    df_hourly: pd.DataFrame,
    df_daily: pd.DataFrame,
    zone_cfg: ZoneConfig,
    model_cfg: ModelConfig,
) -> List[matplotlib.figure.Figure]:
    """Three-panel diagnostic for a single date.

    Locates the fold containing the date, rebuilds features with the fold's
    spike threshold, queries each expert regime curve, and returns three
    Figures: (1) actual vs weighted prediction, (2) actual vs all four
    regime expert curves, (3) regime probability bar chart.

    Args:
        date: Test date to inspect.
        df_predictions: Predictions from walk_forward.
        fold_models: List of fold-specific models from walk_forward.
        df_hourly: Full hourly DataFrame.
        zone_cfg: Zone configuration.
        model_cfg: Model configuration.

    Returns:
        List of three Figures in the order described above.
    """
    _date = pd.Timestamp(date)

    # Locate the prediction row
    _pred_row = df_predictions[pd.to_datetime(df_predictions["date"]) == _date]
    if len(_pred_row) == 0:
        raise ValueError(
            f"Date {_date.date()} not found in predictions. Available range: "
            f"{df_predictions['date'].min()} -> {df_predictions['date'].max()}"
        )
    _pred_row = _pred_row.iloc[0]

    # Identify which fold this date belongs to
    _fold_idx = next(
        (i for i, (_, _, te_s, te_e) in enumerate(zone_cfg.fold_definitions)
         if pd.Timestamp(te_s) <= _date <= pd.Timestamp(te_e)),
        None,
    )
    if _fold_idx is None:
        raise ValueError(f"Date {_date.date()} is not in any fold test window.")

    _models = fold_models[_fold_idx]
    _clf_cols = _models["clf_feature_cols"]

    # Rebuild features with the fold-specific spike threshold (silence prints)
    with contextlib.redirect_stdout(io.StringIO()):
        _df_spike = features.build_daily_features(
            df_hourly, zone_cfg, spike_threshold=_models["spike_threshold"]
        )
        _pivot = features.build_price_pivot_lag1(df_hourly, df_daily, zone_cfg)

    _df_spike_clf = pd.concat([_df_spike, _pivot], axis=1)
    _feat_row = _df_spike_clf.loc[[_date], _clf_cols].values  # (1, n_features)

    _hours  = list(range(24))
    _actual = np.array([_pred_row[f"h{h:02d}"]      for h in _hours])
    _pred_w = np.array([_pred_row[f"pred_h{h:02d}"] for h in _hours])
    _probs  = np.array([_pred_row["p_A"], _pred_row["p_B"],
                        _pred_row["p_C0"], _pred_row["p_C1"]])

    _r_colors = ["steelblue", "crimson", "forestgreen", "darkorange"]
    _r_labels = ["Leaf A (negative)", "Leaf B (spike)", "C0 (normal)", "C1 (normal)"]

    # Per-regime expert curves (not stored in df_predictions - re-queried here)
    _regime_preds = np.array([
        [_models["expert_models"][r][h].predict(_feat_row)[0] for h in _hours]
        for r in range(4)
    ])

    figs: List[matplotlib.figure.Figure] = []

    # Plot 1 - actual vs weighted prediction
    fig1, ax = plt.subplots(figsize=(12, 4))
    ax.plot(_hours, _actual, color="black", linewidth=2, label="Actual")
    ax.plot(_hours, _pred_w, color="steelblue", linewidth=2,
            linestyle="--", label="Model (probability-weighted)")
    ax.set_title(f"{zone_cfg.zone} - Actual vs weighted prediction - {_date.date()}")
    ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Price (EUR/MWh)")
    ax.set_xticks(_hours); ax.legend(); ax.grid(alpha=0.3)
    fig1.tight_layout()
    figs.append(fig1)

    # Plot 2 - per-regime expert curves
    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.plot(_hours, _actual, color="black", linewidth=2, label="Actual")
    for r in range(4):
        ax.plot(_hours, _regime_preds[r], color=_r_colors[r], linewidth=1.5,
                linestyle="--", alpha=0.85,
                label=f"{_r_labels[r]}  (p={_probs[r]:.3f})")
    ax.set_title(f"{zone_cfg.zone} - Actual vs per-regime experts - {_date.date()}")
    ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Price (EUR/MWh)")
    ax.set_xticks(_hours); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig2.tight_layout()
    figs.append(fig2)

    # Plot 3 - regime probability bars
    fig3, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(_r_labels, _probs, color=_r_colors, edgecolor="white", alpha=0.85)
    for bar, p in zip(bars, _probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{p:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Probability")
    ax.set_title(f"{zone_cfg.zone} - Regime probabilities - {_date.date()}")
    ax.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    figs.append(fig3)

    # Feature table (diagnostic side-output)
    print(f"\n=== All features for {_date.date()} ({zone_cfg.zone}) ===")
    for col, val in _df_spike_clf.loc[_date, _clf_cols].items():
        print(f"  {col:<28s} {val:.4f}")

    return figs


def make_all_diagnostics(
    df_predictions: pd.DataFrame,
    fold_models: List[Dict],
    df_hourly: pd.DataFrame,
    zone_cfg: ZoneConfig,
    model_cfg: ModelConfig,
    save_dir: Optional["pathlib.Path"] = None,
) -> Dict[str, matplotlib.figure.Figure]:
    """Convenience: call every plot_* function, optionally saving to disk.

    Args:
        df_predictions: Predictions with naive baselines attached.
        fold_models: List of fold-specific models.
        df_hourly: Full hourly DataFrame.
        zone_cfg: Zone configuration.
        model_cfg: Model configuration.
        save_dir: If given, writes each figure to {save_dir}/{key}.png.

    Returns:
        Dict mapping figure name to Figure.
    """
    ...
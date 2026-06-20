"""
This module generates visual reports to analyze signed bias and residual patterns in energy price forecasts. It calculates the difference between predicted and actual prices across the bidding zones.

The script has several functions that target specific error behaviors:

plot_hourly_bias() charts the mean residual by hour of the day to reveal intraday patterns;
plot_intraday_residual_by_bucket breaks these hourly errors down by specific price regimes such as negative prices or sudden spikes;
plot_spike_hit_rate() plots hit rates for price spikes to determine how often the model under-predicts extreme peaks;
plot_weekend_weekday() compares model performance on weekdays versus weekends. You can execute all these visualizations at once using the provided runner function, which saves the generated charts directly to a local directory.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from IPython.display import display

#  constants

BUCKET_ORDER = ["negative", "near_zero", "normal", "spike"]

ZONE_COLORS = {
    "DK1": "#2c6fad",   # blue  - wind
    "ES":  "#c0392b",   # red   - solar
    "NO2": "#27ae60",   # green - hydro
}

REPORT_STYLE = {
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linewidth":   0.6,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
}


#  helpers



def _assign_price_bucket(
    prices: pd.Series,
    spike_quantile: float = 0.95,
    near_zero_upper: float = 20.0,
) -> pd.Series:
    
    """
    Categorizes a series of prices into distinct market conditions. The conditions are negative, near_zero, normal, and spike.

    Inputs:
        prices: A Pandas Series of actual energy prices.
        spike_quantile: A float setting the statistical threshold for spike prices.
        near_zero_upper: A float defining the upper bound for the near-zero category.

    Output:
        A Pandas Series containing the assigned string label for each price.
    """
    
    spike_thresh = float(prices.quantile(spike_quantile))
    out = pd.Series("normal", index=prices.index, dtype=str)
    out[prices < 0]                                 = "negative"
    out[(prices >= 0) & (prices < near_zero_upper)] = "near_zero"
    out[prices > spike_thresh]                       = "spike"
    return out


def _prepare(
    df: pd.DataFrame,
    spike_quantile: float = 0.95,
    near_zero_upper: float = 20.0,
) -> pd.DataFrame:
   
    """
    Formats and validates the input data for the plotting functions. It checks for required columns, calculates the error residual, extracts the hour of the day, and calculates price buckets for each specific zone.

    Inputs:
        df: A Pandas DataFrame containing date_time, zone, actual, and predicted columns.
        spike_quantile: A float used to determine the spike threshold.
        near_zero_upper: A float setting the maximum value for near-zero prices.

    Output:
        A cleaned and expanded Pandas DataFrame ready for visualization.

    """
   
    required = {"date_time", "zone", "actual", "predicted"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing columns: {missing}\n"
            f"Expected columns: date_time, zone, actual, predicted, cluster"
        )

    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["residual"]  = df["predicted"] - df["actual"]
    df["hour"]      = df["date_time"].dt.hour

    # Build price_bucket per zone so spike threshold is zone-specific
    bucket_parts = []
    for _, sub in df.groupby("zone"):
        bucket_parts.append(
            _assign_price_bucket(sub["actual"], spike_quantile, near_zero_upper)
        )
    df["price_bucket"] = pd.concat(bucket_parts).reindex(df.index)
    df["price_bucket"] = pd.Categorical(
        df["price_bucket"], categories=BUCKET_ORDER, ordered=True
    )
    return df


def _spike_threshold_per_zone(
    df: pd.DataFrame,
    spike_quantile: float = 0.95,
) -> dict:
    
    """
    Calculates the specific price value that qualifies as a spike in each geographic zone.

    Inputs:
        df: The prepared Pandas DataFrame with actual prices.
        spike_quantile: A float indicating the statistical cutoff for a spike.

    Output:
        A dictionary mapping each zone name to its calculated spike threshold.
    """

    return (
        df.groupby("zone")["actual"]
        .quantile(spike_quantile)
        .round(1)
        .to_dict()
    )


#  Plot A: hourly bias with confidence band 

def plot_hourly_bias(
    df: pd.DataFrame,
    spike_quantile: float = 0.95,
    near_zero_upper: float = 20.0,
    save_path: str = "p2_hourly_bias.png",
) -> pd.DataFrame:
    
    """
    Generates a line chart showing the mean signed bias for each hour of the day. The chart highlights the time of peak absolute deviation.

    Inputs:
        df: The raw input Pandas DataFrame.
        spike_quantile: A float defining the upper price bucket.
        near_zero_upper: A float defining the near-zero bucket.
        save_path: A string specifying where to save the output image.

    Output:
        A Pandas DataFrame containing the aggregated hourly mean and standard deviation for each zone.
    """
 
    df    = _prepare(df, spike_quantile, near_zero_upper)
    zones = sorted(df["zone"].unique())

    summary = (
        df.groupby(["zone", "hour"])["residual"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )

    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(
            1, len(zones),
            figsize=(5.5 * len(zones), 4.5),
            sharey=False,
        )
        if len(zones) == 1:
            axes = [axes]

        for ax, zone in zip(axes, zones):
            sub   = summary[summary["zone"] == zone].sort_values("hour")
            hours = sub["hour"].values
            mean  = sub["mean"].values
            std   = sub["std"].values
            color = ZONE_COLORS.get(zone, "#555555")

            ax.axhline(0, color="black", linewidth=1.1, linestyle="--",
                       alpha=0.7, zorder=3, label="Zero (no bias)")
            # ax.fill_between(
            #     hours, mean - std, mean + std,
            #     alpha=0.18, color=color, label="±1 std",
            # )
            ax.plot(hours, mean, color=color, linewidth=2, zorder=4, label="Mean bias")
            # ax.fill_between(hours, mean, 0, where=(mean > 0),
            #                 alpha=0.08, color="tomato", zorder=1, label="Over-prediction")
            # ax.fill_between(hours, mean, 0, where=(mean < 0),
            #                 alpha=0.08, color="dodgerblue", zorder=1, label="Under-prediction")

            # Annotate peak absolute deviation
            # peak_idx = np.argmax(np.abs(mean))
            # peak_h   = hours[peak_idx]
            # peak_v   = mean[peak_idx]
            # offset   = std.mean() * 0.8 * np.sign(peak_v) if peak_v != 0 else 3
            # ax.annotate(
            #     f"h{int(peak_h):02d}: {peak_v:+.1f}",
            #     xy=(peak_h, peak_v),
            #     xytext=(peak_h + 1.5, peak_v + offset),
            #     fontsize=8, color=color,
            #     arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
            # )

            peak_idx = np.argmax(np.abs(mean))
            peak_h   = hours[peak_idx]
            peak_v   = mean[peak_idx]

            y_min, y_max = ax.get_ylim()
            y_range      = y_max - y_min

            # always place label toward the interior, not toward the edge
            text_y = peak_v + y_range * 0.15 if peak_v < (y_min + y_max) / 2 else peak_v - y_range * 0.15
            text_x = min(peak_h + 1.5, 21)   # don't run off the right edge

            ax.annotate(
                f"h{int(peak_h):02d}: {peak_v:+.1f}",
                xy=(peak_h, peak_v),
                xytext=(text_x, text_y),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
                annotation_clip=True,   # clip to axes boundary - never escapes the box
            )

            ax.set_title(zone, fontsize=14, fontweight="bold", color=color, fontname="Georgia" )
            ax.set_xlabel("Hour of day", fontsize = 12, fontname="Verdana")
            ax.set_ylabel("Mean residual (EUR/MWh)" if zone == zones[0] else "", fontsize = 12, fontname="Verdana")
            ax.set_xticks(range(0, 24, 3))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{int(x):02d}:00"
            ))
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.7)

        fig.suptitle(
            "Signed Bias by Hour of Day - Mean Residual (Predicted − Actual)",
            fontsize=12, fontweight="bold", y=1.02, fontname="Verdana"
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        print(f"Saved → {save_path}")

    return summary


#  Plot B: mean residual by hour, faceted by price bucket 

BUCKET_LABELS = {
    "negative":  "Negative prices",
    "near_zero": "Near-zero prices\n(0–20 EUR/MWh)",
    "normal":    "Normal prices",
    "spike":     "Spike prices\n(top 5%)",
}

ZONE_STYLES = {
    "DK1": {"color": "#2c6fad", "label": "DK1 (wind)",  "lw": 2.2},
    "ES":  {"color": "#c0392b", "label": "ES (solar)",   "lw": 2.2},
    "NO2": {"color": "#27ae60", "label": "NO2 (hydro)",  "lw": 2.2},
}


def plot_intraday_residual_by_bucket(
    df: pd.DataFrame,
    spike_quantile:  float = 0.95,
    near_zero_upper: float = 20.0,
    save_path: str = "plot_intraday_residual_by_bucket.png",
) -> pd.DataFrame:
    
    """
    Creates a multi-panel chart displaying the hourly mean residual broken down by price condition. It lets you see how the model behaves during normal hours versus extreme events like spikes or negative prices.

    Inputs:
        df: The raw input Pandas DataFrame.
        spike_quantile: A float setting the spike category cutoff.
        near_zero_upper: A float determining the near-zero upper limit.
        save_path: A string determining the file path for the saved plot.

    Output:
        A Pandas DataFrame containing the grouped statistical summary.
"""
    
    df      = _prepare(df, spike_quantile, near_zero_upper)
    zones   = [z for z in ["DK1", "ES", "NO2"] if z in df["zone"].unique()]
    buckets = [b for b in BUCKET_ORDER if b in df["price_bucket"].unique()]

    summary = (
        df.groupby(["zone", "price_bucket", "hour"])["residual"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    n_cols = len(buckets)

    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(4.2 * n_cols, 4.8),
            sharey=False,
        )
        if n_cols == 1:
            axes = [axes]

        # per-bucket y-limits so columns are self-scaled
        bucket_ylims = {}
        for bucket in buckets:
            sub = summary[summary["price_bucket"] == bucket]
            lo  = (sub["mean"] - sub["std"]).min()
            hi  = (sub["mean"] + sub["std"]).max()
            pad = (hi - lo) * 0.18 if hi != lo else 5
            bucket_ylims[bucket] = (lo - pad, hi + pad)

        for ax, bucket in zip(axes, buckets):
            is_first = (bucket == buckets[0])

            ax.axhline(0, color="black", linewidth=1.0,
                       linestyle="--", alpha=0.65, zorder=3)

            for zone in zones:
                style = ZONE_STYLES.get(zone, {})
                color = style.get("color", "#555")
                lw    = style.get("lw", 1.8)

                sub = summary[
                    (summary["zone"] == zone) &
                    (summary["price_bucket"] == bucket)
                ].sort_values("hour")

                if sub.empty:
                    continue

                hours = sub["hour"].values
                mean  = sub["mean"].values
                std   = sub["std"].fillna(0).values

                # ax.fill_between(
                #     hours, mean - std, mean + std,
                #     alpha=0.10, color=color, zorder=2,
                # )
                ax.plot(
                    hours, mean,
                    color=color, linewidth=lw, zorder=4,
                )
                # ax.fill_between(hours, mean, 0, where=(mean > 0),
                #                 alpha=0.06, color=color, zorder=1)
                # ax.fill_between(hours, mean, 0, where=(mean < 0),
                #                 alpha=0.06, color=color, zorder=1)

                # annotate peak absolute deviation
                # peak_idx = np.argmax(np.abs(mean))
                # peak_h   = hours[peak_idx]
                # peak_v   = mean[peak_idx]
                # ax.annotate(
                #     f"h{int(peak_h):02d}: {peak_v:+.1f}",
                #     xy=(peak_h, peak_v),
                #     xytext=(
                #         peak_h + (2 if peak_h < 20 else -5),
                #         peak_v + np.sign(peak_v) * std.mean() * 0.6,
                #     ),
                #     fontsize=7, color=color,
                #     arrowprops=dict(arrowstyle="->", color=color, lw=0.7),
                #     zorder=6,
                # )

            ax.set_title(BUCKET_LABELS[bucket], fontsize=14, fontweight="bold", pad=7, fontname="Georgia" )
            ax.set_xlabel("Hour of day", fontsize=12,fontname="Verdana")
            ax.set_ylabel("Mean residual (EUR/MWh)" if is_first else "", fontsize=12, fontname="Verdana")
            ax.set_xticks(range(0, 24, 3))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):02d}:00")
            )
            ax.tick_params(axis="x", rotation=25)
            ax.set_xlim(-0.5, 23.5)
            ax.set_ylim(*bucket_ylims[bucket])
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)

            if is_first:
                ax.text(
                    0.01, 0.01,
                    "Positive = over-prediction  |  Negative = under-prediction",
                    transform=ax.transAxes, fontsize=7,
                    color="grey", va="bottom",
                )

        legend_handles = [
            Line2D([0], [0],
                   color=ZONE_STYLES[z]["color"],
                   linewidth=2,
                   label=ZONE_STYLES[z]["label"])
            for z in zones
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center", ncol=len(zones),
            fontsize=9, framealpha=0.8,
            bbox_to_anchor=(0.5, -0.04),
        )
        fig.suptitle(
            "Mean Residual by Hour of Day - Faceted by Price Bucket",
            fontsize=12, fontweight="bold", y=1.02, fontname="Georgia",
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        print(f"Saved → {save_path}")

    return summary


#  Plot C: spike hit-rate table 

def spike_hit_rate_table(
    df: pd.DataFrame,
    spike_quantile: float = 0.95,
    near_zero_upper: float = 20.0,
    thresholds: list = [-10, -20, -50],
) -> pd.DataFrame:

    """
    Computes a summary table analyzing model performance during price spikes. It calculates the percentage of spike hours where the forecast under-predicted the actual price by specific amounts.

    Inputs:
        df: The raw input Pandas DataFrame.
        spike_quantile: A float used to identify spike hours.
        near_zero_upper: A float used during initial data preparation.
        thresholds: A list of negative floats representing the under-prediction margins.

    Output:
        A Pandas DataFrame displaying the hit-rate percentages and mean residuals for spikes in each zone.
    """

    df           = _prepare(df, spike_quantile, near_zero_upper)
    spike_thresh = _spike_threshold_per_zone(df, spike_quantile)
    spike_df     = df[df["price_bucket"] == "spike"]

    rows = []
    for zone, sub in spike_df.groupby("zone"):
        row = {
            "Spike threshold": f">{spike_thresh.get(zone, '?')} EUR/MWh",
            "N hours":         len(sub),
            "Mean residual":   round(sub["residual"].mean(), 2),
            "Median residual": round(sub["residual"].median(), 2),
        }
        for t in thresholds:
            pct = (sub["residual"] < t).mean() * 100
            row[f"% under {t} EUR/MWh"] = f"{pct:.1f}%"
        rows.append({"Zone": zone, **row})

    result = pd.DataFrame(rows).set_index("Zone")
    display(result)
    return result



#  Plot C2: spike hit-rate column chart 

def plot_spike_hit_rate(
    result: pd.DataFrame,
    thresholds: list = [-10, -20, -50],
    save_path: str = "p2_spike_hit_rate.png",
) -> None:
    
    """
    Generates a grouped bar chart from the spike hit-rate data. The visualization displays the frequency of severe under-predictions across different zones.

    Inputs:
        result: The Pandas DataFrame generated by the spike_hit_rate_table function.
        thresholds: A list of negative floats that match the columns in the input table.
        save_path: A string indicating the save destination for the PNG file.

    
    """

    zones = list(result.index)

    # Parse percentage strings back to floats
    data = {}
    for t in thresholds:
        col = f"% under {t} EUR/MWh"
        data[t] = [float(result.loc[z, col].replace("%", "")) for z in zones]

    # Build x-axis labels including the zone-specific spike threshold
    x_labels = []
    for z in zones:
        thresh = result.loc[z, "Spike threshold"]
        x_labels.append(f"{z}\n({thresh})")

    x      = np.arange(len(zones))
    n_bars = len(thresholds)
    width  = 0.22
    # Blues: light → dark matching threshold severity
    bar_colors = ["#5a8fc9", "#2c6fad", "#0c447c"]

    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        bar_groups = []
        for i, (t, color) in enumerate(zip(thresholds, bar_colors)):
            offset = (i - (n_bars - 1) / 2) * width
            bars   = ax.bar(
                x + offset,
                data[t],
                width,
                label=f"{abs(t)} EUR/MWh",
                color=color,
                zorder=3,
            )
            bar_groups.append(bars)

            # value labels above each bar
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.2,
                    f"{h:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8, color=color,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontname="Verdana" )
        ax.set_ylabel("% of spike hours under-predicted", fontsize=12, fontname="Verdana")
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(v)}%")
        )
        ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            title="Under-predicted by more than",
            title_fontsize=8,
            fontsize=9,
            framealpha=0.7,
            loc="upper right",
        )
        ax.set_title(
            "Spike bucket - under-prediction hit-rate by zone\n"
            "% of spike hours where residual < threshold",
            fontsize=14, fontweight="bold", fontname="Georgia"
        )

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        print(f"Saved → {save_path}")


#  Plot D: weekday vs weekend hourly bias 

def plot_weekend_weekday(
    df: pd.DataFrame,
    spike_quantile:  float = 0.95,
    near_zero_upper: float = 20.0,
    save_path: str = "p2_weekend_weekday.png",
) -> pd.DataFrame:
 
    """
    Creates a plot comparing the hourly residual patterns of regular weekdays against weekends. It includes the overall Mean Absolute Error for each day type in the chart title.

    Inputs:
        df: The raw input Pandas DataFrame.
        spike_quantile: A float required by the preparation helper.
        near_zero_upper: A float required by the preparation helper.
        save_path: A string specifying the output file location.

    Output:
        A Pandas DataFrame summarizing the hourly mean residual grouped by zone and day type.
    """

    df    = _prepare(df, spike_quantile, near_zero_upper)
    zones = sorted(df["zone"].unique())

    df["day_type"] = np.where(
        df["date_time"].dt.dayofweek < 5, "Weekday", "Weekend"
    )

    summary = (
        df.groupby(["zone", "day_type", "hour"])["residual"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    overall = (
        df.groupby(["zone", "day_type"])["residual"]
        .agg(
            mean_bias="mean",
            mae=lambda x: np.abs(x).mean(),
            n="count",
        )
        .reset_index()
    )

    DAY_STYLES = {
        "Weekday": {"ls": "-",  "lw": 2.0},
        "Weekend": {"ls": "--", "lw": 2.0},
    }

    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(
            1, len(zones),
            figsize=(5.5 * len(zones), 4.8),
            sharey=False,
        )
        if len(zones) == 1:
            axes = [axes]

        for ax, zone in zip(axes, zones):
            color = ZONE_COLORS.get(zone, "#555555")

            ax.axhline(0, color="black", linewidth=1.0,
                       linestyle="--", alpha=0.65, zorder=3)

            for day_type, style in DAY_STYLES.items():
                sub = summary[
                    (summary["zone"] == zone) &
                    (summary["day_type"] == day_type)
                ].sort_values("hour")

                if sub.empty:
                    continue

                hours = sub["hour"].values
                mean  = sub["mean"].values

                ax.plot(
                    hours, mean,
                    color=color,
                    linewidth=style["lw"],
                    linestyle=style["ls"],
                    zorder=4,
                    label=day_type,
                )

                # # annotate peak absolute deviation
                # peak_idx = np.argmax(np.abs(mean))
                # peak_h   = hours[peak_idx]
                # peak_v   = mean[peak_idx]
                # ax.annotate(
                #     f"h{int(peak_h):02d}: {peak_v:+.1f}",
                #     xy=(peak_h, peak_v),
                #     xytext=(
                #         peak_h + (2 if peak_h < 20 else -6),
                #         peak_v + (3 if peak_v >= 0 else -4),
                #     ),
                #     fontsize=7.5, color=color,
                #     arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                #     zorder=6,
                # )

            # MAE per day type in subtitle
            row_wd = overall[(overall["zone"] == zone) & (overall["day_type"] == "Weekday")]
            row_we = overall[(overall["zone"] == zone) & (overall["day_type"] == "Weekend")]
            mae_wd = f"{row_wd['mae'].values[0]:.1f}" if not row_wd.empty else "n/a"
            mae_we = f"{row_we['mae'].values[0]:.1f}" if not row_we.empty else "n/a"

            ax.set_title(
                f"{zone}\nMAE - weekday: {mae_wd}  |  weekend: {mae_we} EUR/MWh",
                fontsize=14, fontweight="bold", color=color, pad=7, fontname="Georgia", 
            )
            ax.set_xlabel("Hour of day", fontsize=12, fontname="Verdana" )
            ax.set_ylabel(
                "Mean residual (EUR/MWh)" if zone == zones[0] else "",
                fontsize=12, fontname="Verdana"
            )
            ax.set_xticks(range(0, 24, 3))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):02d}:00")
            )
            ax.legend(fontsize=9, framealpha=0.7, loc="upper left")
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)

        fig.suptitle(
            "Weekday vs Weekend - Mean Residual by Hour of Day\n"
            "Solid = weekday  |  Dashed = weekend",
            fontsize=12, fontweight="bold", y=1.03, fontname="Georgia", 
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        print(f"Saved → {save_path}")

    

    return summary


def run_part2_diagn(
    df: pd.DataFrame,
    spike_quantile: float = 0.95,
    near_zero_upper: float = 20.0,
    thresholds: list = [-10, -20, -50],
    save_dir: str = ".",
) -> dict:
    
    """
    Executes the entire suite of signed bias diagnostics. It generates all tables and plots sequentially and saves the visuals to a specified directory.

    Inputs:
        df: The raw input Pandas DataFrame with actuals and predictions.
        spike_quantile: A float setting the statistical boundary for spikes.
        near_zero_upper: A float setting the upper boundary for near-zero prices.
        thresholds: A list of negative floats used to evaluate extreme under-predictions.
        save_dir: A string specifying the folder where all images will be saved.

    Output:
        A dictionary containing the summary DataFrames generated by each individual diagnostic step.
    """

    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("SIGNED BIAS DIAGNOSTICS")
    print("=" * 60)

    print("\n▶ Plot A: Hourly bias with confidence band")
    hourly_summary = plot_hourly_bias(
        df,
        spike_quantile  = spike_quantile,
        near_zero_upper = near_zero_upper,
        save_path       = f"{save_dir}/p2_hourly_bias.png",
    )

    print("\n▶ Plot B: Hour × Price-Bucket")
    intraday_residual_by_bucket = plot_intraday_residual_by_bucket(
        df,
        spike_quantile  = spike_quantile,
        near_zero_upper = near_zero_upper,
        save_path       = f"{save_dir}/plot_intraday_residual_by_bucket.png",
    )

    print("\n▶ Table C: Spike hit-rate")
    hit_rate = spike_hit_rate_table(
        df,
        spike_quantile  = spike_quantile,
        near_zero_upper = near_zero_upper,
        thresholds      = thresholds,
    )

    print("\n▶ Plot C: Spike hit-rate column chart")
    plot_spike_hit_rate(
        hit_rate,
        thresholds = thresholds,
        save_path  = f"{save_dir}/p2_spike_hit_rate.png",
    )

    print("\n▶ Plot D: Weekday vs Weekend hourly bias")
    weekend_summary = plot_weekend_weekday(
        df,
        spike_quantile  = spike_quantile,
        near_zero_upper = near_zero_upper,
        save_path       = f"{save_dir}/p2_weekend_weekday.png",
    )

    return {
        "hourly_summary":          hourly_summary,
        "intraday_residual_by_bucket": intraday_residual_by_bucket,
        "hit_rate_table":          hit_rate,
        "weekend_summary":         weekend_summary,
    }


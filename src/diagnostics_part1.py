
"""
This module is a diagnostic tool for evaluating energy price forecasts. It calculates standard error metrics like Mean Absolute Error, Weighted Average Percentage Error, clamped WAPE, and mean signed bias. The code applies a price floor to prevent mathematically inflated errors when prices are near zero.

The main pipeline slices the data to analyze performance across different conditions. It groups the results by season, day type, price bucket, and model clusters. The module has a printing helper that outputs a formatted text summary. It also has a plotting function to generate visual breakdowns of the metrics. The script also has wrapper functions to process multiple zones at once and generate a high-level comparison table.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


# #############################################################################
# 1.  CORE METRIC FUNCTION
# #############################################################################

def diagnostic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_floor: float = 10.0,
) -> dict:


    """
    Calculates standard and clamped error metrics for a set of predictions.

    Inputs:
        y_true: Numpy array of actual prices.
        y_pred: Numpy array of predicted prices.
        price_floor: Minimum absolute value denominator for the clamped WAPE calculation.

    Output:
        A dictionary containing the sample size, Mean Absolute Error, Weighted Average Percentage Error, clamped WAPE, and mean signed bias.
    """


    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    n      = len(y_true)
    errors = y_pred - y_true          # signed residuals
    abs_e  = np.abs(errors)

    mae  = float(np.mean(abs_e))
    bias = float(np.mean(errors))

    denom_raw     = np.sum(np.abs(y_true))
    denom_clamped = np.sum(np.maximum(np.abs(y_true), price_floor))

    wape    = float(np.sum(abs_e) / denom_raw    * 100) if denom_raw    > 0 else np.nan
    wape_cl = float(np.sum(abs_e) / denom_clamped * 100) if denom_clamped > 0 else np.nan

    return dict(n=n, mae=mae, wape=wape, wape_cl=wape_cl, bias=bias)


# #############################################################################
# 2.  PRICE BUCKET ASSIGNMENT
# #############################################################################

# Ordered list used for display / sorting throughout
BUCKET_ORDER = ["negative", "near_zero", "normal", "spike"]



def assign_price_bucket(
    prices: pd.Series,
    spike_quantile: float = 0.95,
    near_zero_upper: float = 20.0,
) -> pd.Series:
    
    """
    Categorizes electricity prices into distinct regimes like negative, near_zero, normal, and spike.

    Inputs:
        prices: Pandas Series of price data.
        spike_quantile: Statistical threshold determining the spike bucket.
        near_zero_upper: Upper limit for the near-zero bucket.

    Output:
        A Pandas Series containing the string labels for each price bucket.
    """
    spike_thresh = float(prices.quantile(spike_quantile))
    out = pd.Series("normal", index=prices.index, dtype=str)

    out[prices < 0]                                      = "negative"
    out[(prices >= 0) & (prices < near_zero_upper)]      = "near_zero"
    out[prices > spike_thresh]                            = "spike"
    # "normal" is already the default; the three conditions above overwrite it.

    return out


# #############################################################################
# 3.  FULL DIAGNOSTIC RUNNER
# #############################################################################




def run_full_diagnostic(
    pred_df:     pd.DataFrame,
    cluster_df:  pd.DataFrame,
    zone_name:   str,
    price_floor: float = 10.0,
) -> pd.DataFrame:
    

    """
    Executes the complete diagnostic pipeline for a single pricing zone. It merges predictions with cluster labels, extracts calendar features, and computes metrics across various data slices.

    Inputs:
        pred_df: DataFrame of model predictions.
        cluster_df: DataFrame of cluster assignments.
        zone_name: String identifying the pricing zone.
        price_floor: Floor value applied to error calculations.

    Output:
        A Pandas DataFrame summarizing the metrics for all computed dimensions.
    """

    # Prepare working DataFrame 
    df = pred_df.copy()
    df["date_time"] = pd.to_datetime(df["date"], utc=True)
    df["date"]      = df["date_time"].dt.normalize().dt.strftime("%Y-%m-%d")

    # Merge cluster labels
    cl = cluster_df[["date", "cluster"]].copy()
    cl["date"] = cl["date"].astype(str)
    df = df.merge(cl, on="date", how="left")

    # Calendar features
    df["hour"]       = df["date_time"].dt.hour
    df["dayofweek"]  = df["date_time"].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"]      = df["date_time"].dt.month
    df["season"]     = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3:  "Spring", 4: "Spring", 5: "Spring",
        6:  "Summer", 7: "Summer", 8: "Summer",
        9:  "Autumn", 10:"Autumn", 11:"Autumn",
    })
    df["day_type"]   = np.where(df["dayofweek"] < 5, "Weekday", "Weekend")

    # Price buckets
    df["price_bucket"] = assign_price_bucket(df["price_eur_mwh"])

    total_hours = len(df)
    rows = []

    def _add_slice(dimension: str, label: str, mask: pd.Series) -> None:
        sub = df[mask].dropna(subset=["price_eur_mwh", "weighted_pred_price"])
        if len(sub) == 0:
            return
        m = diagnostic_metrics(
            sub["price_eur_mwh"].values,
            sub["weighted_pred_price"].values,
            price_floor=price_floor,
        )
        rows.append({
            "dimension":   dimension,
            "group":       label,
            "n_hours":     m["n"],
            "pct_of_test": round(m["n"] / total_hours * 100, 1),
            "mae":         round(m["mae"],     2),
            "wape":        round(m["wape"],    2),
            "wape_cl":     round(m["wape_cl"], 2),
            "bias":        round(m["bias"],    2),
        })

    #  Overall 
    _add_slice("overall", "All", pd.Series(True, index=df.index))

    # By cluster 
    for cid in sorted(df["cluster"].dropna().unique()):
        _add_slice("cluster", f"Cluster {int(cid)}", df["cluster"] == cid)

    # By season
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        mask = df["season"] == season
        if mask.any():
            _add_slice("season", season, mask)

    # By day type 
    _add_slice("day_type", "Weekday", df["day_type"] == "Weekday")
    _add_slice("day_type", "Weekend", df["day_type"] == "Weekend")

    #  By price bucket 
    for bucket in BUCKET_ORDER:
        mask = df["price_bucket"] == bucket
        if mask.any():
            _add_slice("price_bucket", bucket, mask)

    summary = pd.DataFrame(rows)

    #   Print formatted table 
    _print_summary(summary, zone_name, price_floor)

    #  Plot 
    _plot_diagnostic(summary, zone_name)

    return summary


# #############################################################################
# 4.  PRINT HELPER
# #############################################################################



def _print_summary(df: pd.DataFrame, zone: str, price_floor: float) -> None:

    
    """
    Prints a formatted text table of the diagnostic results to the console. It flags areas where clamping significantly alters the error.

    Inputs:
        df: Summary DataFrame from the diagnostic run.
        zone: String name of the zone.
        price_floor: Floor value used in the run.
    """


    SEP = "=" * 78
    print(f"\n{SEP}")
    print(f"  DIAGNOSTIC SUMMARY - {zone}   (price_floor = {price_floor} EUR/MWh)")
    print(SEP)

    header = (
        f"  {'Group':<22} {'N hrs':>7}  {'% test':>6}  "
        f"{'MAE':>7}  {'WAPE':>7}  {'WAPE_CL':>8}  {'BIAS':>7}"
    )
    sub_hdr = (
        f"  {'':22} {'':>7}  {'':>6}  "
        f"{'EUR/MWh':>7}  {'%':>7}  {'% (fld)':>8}  {'EUR/MWh':>7}"
    )

    DIMENSIONS = [
        ("overall",      "Overall"),
        ("cluster",      "By Cluster"),
        ("season",       "By Season"),
        ("day_type",     "Weekday vs Weekend"),
        ("price_bucket", "By Price Bucket"),
    ]

    for dim_key, dim_label in DIMENSIONS:
        sub = df[df["dimension"] == dim_key]
        if sub.empty:
            continue
        print(f"\n  {'─'*74}")
        print(f"  {dim_label}")
        print(f"  {'─'*74}")
        print(header)
        print(sub_hdr)
        print(f"  {'─'*74}")

        # sort: clusters numerically, buckets by regime order, others alphabetically
        if dim_key == "cluster":
            sub = sub.sort_values("wape_cl", ascending=False)
        elif dim_key == "price_bucket":
            sub = sub.set_index("group").reindex(
                [b for b in BUCKET_ORDER if b in sub["group"].values]
            ).reset_index()
        else:
            sub = sub.sort_values("wape_cl", ascending=False)

        worst_wape = sub["wape_cl"].max()

        for _, row in sub.iterrows():
            flag = " ◄ worst" if row["wape_cl"] == worst_wape and len(sub) > 1 else ""
            # flag where clamping made a big difference
            inflate = ""
            if pd.notna(row["wape"]) and pd.notna(row["wape_cl"]):
                delta = row["wape"] - row["wape_cl"]
                if delta > 20:
                    inflate = f"  [raw WAPE inflated by {delta:.0f}pp - near-zero prices]"

            print(
                f"  {row['group']:<22} {int(row['n_hours']):>7,}  "
                f"{row['pct_of_test']:>5.1f}%  "
                f"{row['mae']:>7.2f}  "
                f"{row['wape']:>6.2f}%  "
                f"{row['wape_cl']:>7.2f}%  "
                f"{row['bias']:>+7.2f}"
                f"{flag}{inflate}"
            )

    print(f"\n{SEP}\n")


# #############################################################################
# 5.  PLOT HELPER
# #############################################################################



COLORS = {
    # raw WAPE bars (muted)
    "wape_raw":  "#b0c4de",
    # clamped WAPE bars (solid)
    "wape_cl":   "#2c6fad",
    # bias positive / negative
    "bias_pos":  "#e07b54",
    "bias_neg":  "#5aab61",
    # background stripe
    "stripe":    "#f5f5f5",
}


def _plot_diagnostic(summary: pd.DataFrame, zone: str) -> None:

    """
    Generates a multi-panel bar chart comparing raw and clamped WAPE. It overlays mean signed bias using a secondary axis.

    Inputs:
        summary: DataFrame of diagnostic results.
        zone: String name of the region.

    Output:
        None. The function saves a PNG file and displays the plot.
    """

   
    DIMS = [
        ("cluster",      "By Cluster"),
        ("season",       "By Season"),
        ("day_type",     "Weekday vs Weekend"),
        ("price_bucket", "By Price Bucket"),
    ]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"{zone} - Diagnostic Metric Breakdown\n"
        f"Bars: raw WAPE (light) vs clamped WAPE (dark)  |  "
        f"Dots: Mean Signed Bias (right axis)",
        fontsize=14, fontweight="bold", y=0.98, fontname = "Georgia"
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (dim_key, dim_label) in enumerate(DIMS):
        sub = summary[summary["dimension"] == dim_key].copy()
        if sub.empty:
            continue

        # Sort
        if dim_key == "cluster":
            sub = sub.sort_values("wape_cl", ascending=False)
        elif dim_key == "price_bucket":
            sub = sub.set_index("group").reindex(
                [b for b in BUCKET_ORDER if b in sub["group"].values]
            ).reset_index()
        else:
            sub = sub.sort_values("wape_cl", ascending=False)

        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax2 = ax.twinx()

        x      = np.arange(len(sub))
        width  = 0.35
        labels = sub["group"].tolist()

        # Background stripes for readability
        for i in range(len(sub)):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color=COLORS["stripe"], zorder=0)

        # WAPE bars
        bars_raw = ax.bar(
            x - width / 2, sub["wape"],    width,
            color=COLORS["wape_raw"], edgecolor="white", linewidth=0.5,
            label="WAPE (raw)", zorder=2,
        )
        bars_cl = ax.bar(
            x + width / 2, sub["wape_cl"], width,
            color=COLORS["wape_cl"],  edgecolor="white", linewidth=0.5,
            label="WAPE (clamped)", zorder=2,
        )

        # Bias dots on secondary axis
        bias_colors = [
            COLORS["bias_pos"] if b >= 0 else COLORS["bias_neg"]
            for b in sub["bias"]
        ]
        ax2.scatter(
            x, sub["bias"], color=bias_colors, s=70, zorder=5,
            edgecolors="white", linewidths=0.6,
        )
        ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=3)
        ax2.set_ylabel("Mean Signed Bias (EUR/MWh)", fontsize=12, color="grey", fontname="Verdana")
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=7)

        # Value labels on clamped WAPE bars
        for bar in bars_cl:
            h = bar.get_height()
            if pd.notna(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.5, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=7, color=COLORS["wape_cl"], fontname="Verdana",
                )

        ax.set_title(dim_label, fontsize=14, fontweight="bold", fontname="Georgia")
        ax.set_ylabel("WAPE (%)", fontsize=12, fontname="Verdana")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, rotation=20, ha="right", fontname="Verdana")
        ax.set_ylim(0, max(sub[["wape", "wape_cl"]].max().max() * 1.25, 10))
        ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

        # Legend only on first panel
        if idx == 0:
            legend_handles = [
                Patch(color=COLORS["wape_raw"],  label="WAPE (raw, no floor)"),
                Patch(color=COLORS["wape_cl"],   label="WAPE (clamped floor)"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["bias_pos"],
                           markersize=8, label="Bias > 0 (over-pred)"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["bias_neg"],
                           markersize=8, label="Bias < 0 (under-pred)"),
            ]
            ax.legend(handles=legend_handles, fontsize=7.5, loc="upper right")

    plt.savefig(f"diagnostic_{zone}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figure saved → diagnostic_{zone}.png")


# #############################################################################
# 6.  CONVENIENCE WRAPPER - run all zones in one call
# #############################################################################


def run_all_zones(
    pred_dfs:    dict,
    clusters_dfs: dict,
    price_floor: float = 10.0,
) -> dict:
    

    
    """
    Loops through multiple pricing zones and runs the full diagnostic pipeline on each one.

    Inputs:
        pred_dfs: Dictionary mapping zones to prediction DataFrames.
        clusters_dfs: Dictionary mapping zones to cluster DataFrames.
        price_floor: Minimum denominator value for clamped errors.

    Output:
        A dictionary mapping each zone to its corresponding summary DataFrame.
    """


    summaries = {}
    for zone, pred_df in pred_dfs.items():
        if zone not in clusters_dfs:
            print(f"[SKIP] {zone}: no cluster_df found in clusters_dfs.")
            continue
        print(f"\n{'▶'*3}  Running diagnostics for {zone}  {'◀'*3}")
        summaries[zone] = run_full_diagnostic(
            pred_df      = pred_df,
            cluster_df   = clusters_dfs[zone],
            zone_name    = zone,
            price_floor  = price_floor,
        )
    return summaries


# #############################################################################
# 7.  CROSS-ZONE COMPARISON TABLE  (bonus - one call after running all zones)
# #############################################################################



def cross_zone_comparison(summaries: dict) -> pd.DataFrame:

    """
    Extracts the overall performance metrics from each zone's summary and prints a high-level comparison table.

    Inputs:
        summaries: Dictionary of summary DataFrames from all tested zones.

    Output:
        A Pandas DataFrame containing the combined overall metrics for every zone.
    """

    rows = []
    for zone, df in summaries.items():
        overall = df[df["dimension"] == "overall"]
        if overall.empty:
            continue
        r = overall.iloc[0]
        rows.append({
            "zone":    zone,
            "mae":     r["mae"],
            "wape":    r["wape"],
            "wape_cl": r["wape_cl"],
            "bias":    r["bias"],
        })

    comp = pd.DataFrame(rows).set_index("zone")

    print("\n" + "=" * 60)
    print("  CROSS-ZONE OVERALL COMPARISON")
    print("=" * 60)
    print(f"  {'Zone':<8} {'MAE':>8}  {'WAPE':>8}  {'WAPE_CL':>9}  {'BIAS':>8}")
    print(f"  {'':8} {'EUR/MWh':>8}  {'%':>8}  {'% (fld)':>9}  {'EUR/MWh':>8}")
    print("  " + "-" * 56)
    for zone, row in comp.iterrows():
        print(
            f"  {zone:<8} {row['mae']:>8.2f}  {row['wape']:>7.2f}%  "
            f"{row['wape_cl']:>8.2f}%  {row['bias']:>+8.2f}"
        )
    print("=" * 60 + "\n")

    return comp

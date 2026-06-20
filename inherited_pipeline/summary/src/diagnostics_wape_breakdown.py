"""
Phase 2 — Step 1: Segmented WAPE Breakdown
===========================================
Computes WAPE broken down by:
  - Cluster
  - Season
  - Weekday vs. Weekend
  - Extreme price days (spikes + negatives)

INPUT DATAFRAME SCHEMA (one row per hour):
  - timestamp         : datetime (hourly)
  - zone              : str ('DK1', 'ES', 'NO2')
  - actual            : float  — actual price (EUR/MWh)
  - predicted         : float  — model predicted price (EUR/MWh)
  - cluster           : int    — assigned cluster label (from clean Phase 1 pipeline)

USAGE:
  from diagnostics_step1_wape_breakdown import run_diagnostics
  results = run_diagnostics(df)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Core WAPE function
# ---------------------------------------------------------------------------

def wape(actual: pd.Series, predicted: pd.Series) -> float:
    """
    WAPE = sum(|y - y_hat|) / sum(|y|) * 100
    Returns NaN if denominator is zero.
    """
    denom = actual.abs().sum()
    if denom == 0:
        return np.nan
    return (actual - predicted).abs().sum() / denom * 100


# ---------------------------------------------------------------------------
# Segment definitions
# ---------------------------------------------------------------------------

def assign_season(ts: pd.Series) -> pd.Series:
    """Map month -> season label."""
    month = ts.dt.month
    return pd.cut(
        month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=["Winter", "Spring", "Summer", "Autumn", "Winter"],
        ordered=False
    ).astype(str).where(month != 12, "Winter")
    # Simpler, explicit mapping:


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Autumn"


def is_extreme_price_day(df: pd.DataFrame,
                          spike_pct: float = 95,
                          negative_threshold: float = 0.0) -> pd.Series:
    """
    Mark a day as 'extreme' if its daily mean actual price is:
      - above the spike_pct percentile (price spike), OR
      - below negative_threshold (negative prices)

    Returns a boolean Series indexed like df.
    """
    daily_mean = (
        df.groupby(df["timestamp"].dt.date)["actual"]
        .mean()
        .rename("daily_mean")
    )
    df_copy = df.copy()
    df_copy["_date"] = df_copy["timestamp"].dt.date
    df_copy = df_copy.merge(daily_mean, left_on="_date", right_index=True)

    spike_cutoff = daily_mean.quantile(spike_pct / 100)
    is_extreme = (df_copy["daily_mean"] >= spike_cutoff) | \
                 (df_copy["daily_mean"] < negative_threshold)
    return is_extreme.values


# ---------------------------------------------------------------------------
# Breakdown functions
# ---------------------------------------------------------------------------

def _zone_totals(df: pd.DataFrame) -> dict:
    """Total hours per zone — used to compute share_pct in every breakdown."""
    return df.groupby("zone").size().to_dict()


def wape_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    totals = _zone_totals(df)
    rows = []
    for zone in df["zone"].unique():
        z = df[df["zone"] == zone]
        for cluster in sorted(z["cluster"].unique()):
            g = z[z["cluster"] == cluster]
            n_hours = len(g)
            rows.append({
                "zone": zone,
                "segment": "cluster",
                "group": f"Cluster {cluster}",
                "n_days": g["timestamp"].dt.date.nunique(),
                "n_hours": n_hours,
                "share_pct": round(n_hours / totals[zone] * 100, 1),
                "wape": wape(g["actual"], g["predicted"])
            })
    return pd.DataFrame(rows)


def wape_by_season(df: pd.DataFrame) -> pd.DataFrame:
    totals = _zone_totals(df)
    df = df.copy()
    df["season"] = df["timestamp"].dt.month.map(get_season)
    rows = []
    for zone in df["zone"].unique():
        z = df[df["zone"] == zone]
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            g = z[z["season"] == season]
            if len(g) == 0:
                continue
            n_hours = len(g)
            rows.append({
                "zone": zone,
                "segment": "season",
                "group": season,
                "n_days": g["timestamp"].dt.date.nunique(),
                "n_hours": n_hours,
                "share_pct": round(n_hours / totals[zone] * 100, 1),
                "wape": wape(g["actual"], g["predicted"])
            })
    return pd.DataFrame(rows)


def wape_by_weekday_weekend(df: pd.DataFrame) -> pd.DataFrame:
    totals = _zone_totals(df)
    df = df.copy()
    df["day_type"] = df["timestamp"].dt.dayofweek.apply(
        lambda d: "Weekend" if d >= 5 else "Weekday"
    )
    rows = []
    for zone in df["zone"].unique():
        z = df[df["zone"] == zone]
        for day_type in ["Weekday", "Weekend"]:
            g = z[z["day_type"] == day_type]
            n_hours = len(g)
            rows.append({
                "zone": zone,
                "segment": "day_type",
                "group": day_type,
                "n_days": g["timestamp"].dt.date.nunique(),
                "n_hours": n_hours,
                "share_pct": round(n_hours / totals[zone] * 100, 1),
                "wape": wape(g["actual"], g["predicted"])
            })
    return pd.DataFrame(rows)


def wape_by_extreme_days(df: pd.DataFrame,
                          spike_pct: float = 95) -> pd.DataFrame:
    totals = _zone_totals(df)
    df = df.copy()
    df["is_extreme"] = is_extreme_price_day(df, spike_pct=spike_pct)
    rows = []
    for zone in df["zone"].unique():
        z = df[df["zone"] == zone]
        for label, flag in [("Normal days", False), ("Extreme days", True)]:
            g = z[z["is_extreme"] == flag]
            if len(g) == 0:
                continue
            n_hours = len(g)
            rows.append({
                "zone": zone,
                "segment": "extreme",
                "group": label,
                "n_days": g["timestamp"].dt.date.nunique(),
                "n_hours": n_hours,
                "share_pct": round(n_hours / totals[zone] * 100, 1),
                "wape": wape(g["actual"], g["predicted"])
            })
    return pd.DataFrame(rows)


def wape_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zone in df["zone"].unique():
        z = df[df["zone"] == zone]
        n_hours = len(z)
        rows.append({
            "zone": zone,
            "segment": "overall",
            "group": "All",
            "n_days": z["timestamp"].dt.date.nunique(),
            "n_hours": n_hours,
            "share_pct": 100.0,
            "wape": wape(z["actual"], z["predicted"])
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostics(df: pd.DataFrame,
                    spike_pct: float = 95,
                    print_results: bool = True) -> pd.DataFrame:
    """
    Run all segmented WAPE breakdowns and return a combined results DataFrame.

    Parameters
    ----------
    df          : Input dataframe (see schema at top of file)
    spike_pct   : Percentile threshold for defining "extreme" spike days
    print_results : Whether to print a formatted summary to stdout

    Returns
    -------
    pd.DataFrame with columns: zone, segment, group, n_days, n_hours, wape
    """
    # Validate required columns
    required = {"timestamp", "zone", "actual", "predicted", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Run all breakdowns
    results = pd.concat([
        wape_overall(df),
        wape_by_cluster(df),
        wape_by_season(df),
        wape_by_weekday_weekend(df),
        wape_by_extreme_days(df, spike_pct=spike_pct),
    ], ignore_index=True)

    results["wape"] = results["wape"].round(2)
    results["share_pct"] = results["share_pct"].round(1)

    if print_results:
        _print_summary(results)

    return results


def _print_summary(results: pd.DataFrame):
    zones = results["zone"].unique()
    segments = ["overall", "cluster", "season", "day_type", "extreme"]
    segment_labels = {
        "overall": "Overall",
        "cluster": "By Cluster",
        "season": "By Season",
        "day_type": "Weekday vs. Weekend",
        "extreme": "Normal vs. Extreme Days",
    }

    for zone in sorted(zones):
        print(f"\n{'='*65}")
        print(f"  Zone: {zone}")
        print(f"{'='*65}")
        z = results[results["zone"] == zone]

        for seg in segments:
            subset = z[z["segment"] == seg].sort_values("wape", ascending=False)
            if subset.empty:
                continue
            print(f"\n  {segment_labels[seg]}:")
            print(f"  {'Group':<22} {'WAPE (%)':>9}  {'%of test':>6}  {'Days':>6}  {'Hours':>7}")
            print(f"  {'-'*22} {'-'*9}  {'-'*9}  {'-'*6}  {'-'*7}")
            worst_wape = subset["wape"].max()
            for _, row in subset.iterrows():
                flag = " ◄ worst" if (seg != "overall" and row["wape"] == worst_wape) else ""
                share = f"{row['share_pct']:.1f}%" if seg != "overall" else "  100%"
                print(f"  {row['group']:<22} {row['wape']:>8.2f}%  {share:>6}  "
                      f"{int(row['n_days']):>6}  {int(row['n_hours']):>7}{flag}")


# ---------------------------------------------------------------------------
# Export helper — saves results to CSV
# ---------------------------------------------------------------------------

def save_results(results: pd.DataFrame, path: str = "wape_breakdown.csv"):
    results.to_csv(path, index=False)
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Quick demo with synthetic data (run this file directly to test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    n = 24 * 365  # one year of hourly data

    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    rows = []
    for zone, n_clusters in [("DK1", 6), ("ES", 3), ("NO2", 5)]:
        actual = np.random.lognormal(mean=4.5, sigma=0.6, size=n)
        # Simulate model that's worse on spikes
        noise = np.where(actual > np.percentile(actual, 90),
                         actual * np.random.uniform(0.3, 0.7, size=n),
                         actual * np.random.uniform(0.05, 0.25, size=n))
        predicted = actual + noise * np.random.choice([-1, 1], size=n)
        clusters = np.random.randint(0, n_clusters, size=n)

        rows.append(pd.DataFrame({
            "timestamp": dates,
            "zone": zone,
            "actual": actual,
            "predicted": predicted,
            "cluster": clusters,
        }))

    df = pd.concat(rows, ignore_index=True)

    results = run_diagnostics(df, spike_pct=95, print_results=True)
    save_results(results, "wape_breakdown_demo.csv")
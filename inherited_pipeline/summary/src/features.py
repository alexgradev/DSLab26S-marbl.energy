"""
src/features.py — Masterset construction and feature engineering.

Covers:
  - ERA5 NetCDF → hourly weather DataFrame
  - ENTSO-E preprocessed CSV → UTC hourly price DataFrame
  - Masterset merge, validation, and export
  - Daily aggregation and lagged features for Layer 1
  - Rolling precipitation windows for Layer 2
  - Calendar dummies (weekend, season)
  - Cluster label attachment
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = [
    "process_era5_netcdf",
    "prepare_price_data",
    "build_masterset",
    "make_daily_values",
    "make_cluster_pred",
    "add_weekend_dummy",
    "add_season_dummies",
    "add_precipitation_last_x_days",
    "attach_cluster",
    "days_not_24_entries",
]


# ------------------------------------------------------------------
# ERA5 weather processing
# ------------------------------------------------------------------

def process_era5_netcdf(
    zone: str,
    years: list[int],
    raw_weather_dir: Path = Path("data/raw/weather"),
) -> pd.DataFrame:
    """Load ERA5 NetCDF files for a zone and return a UTC-indexed hourly weather DataFrame.

    Processes all (year, month) combinations found on disk. Missing files are skipped silently.

    Args:
        zone: Zone label matching the file naming convention (e.g. "DK1").
        years: List of years to include (e.g. [2023, 2024, 2025]).
        raw_weather_dir: Directory containing era5_{zone}_{year}_{month}_instant.nc files.

    Returns:
        DataFrame with UTC DatetimeIndex and columns
        [temperature_2m, wind_speed_10m, precipitation_mm, solar_radiation_W].
    """
    import xarray as xr  # local import — heavy dependency not needed just for import

    raw_weather_dir = Path(raw_weather_dir)
    all_monthly: list[pd.DataFrame] = []

    for year in years:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            path_instant = raw_weather_dir / f"era5_{zone}_{year}_{month_str}_instant.nc"
            path_accum   = raw_weather_dir / f"era5_{zone}_{year}_{month_str}_accum.nc"

            if not path_instant.exists() or not path_accum.exists():
                continue

            try:
                # --- Instantaneous variables (temperature, wind) ---
                ds_inst = xr.open_dataset(path_instant)
                if "valid_time" in ds_inst.coords:
                    ds_inst = ds_inst.rename({"valid_time": "time"})
                df_inst = ds_inst.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()

                if "si10" not in df_inst.columns:
                    if "u10" in df_inst.columns and "v10" in df_inst.columns:
                        df_inst["wind_speed_10m"] = np.sqrt(df_inst["u10"] ** 2 + df_inst["v10"] ** 2)
                    else:
                        df_inst["wind_speed_10m"] = np.nan
                else:
                    df_inst["wind_speed_10m"] = df_inst["si10"]

                if "t2m" in df_inst.columns:
                    df_inst["temperature_2m"] = df_inst["t2m"] - 273.15

                df_inst = df_inst[["time", "temperature_2m", "wind_speed_10m"]]

                # --- Accumulated variables (precipitation, solar radiation) ---
                ds_acc = xr.open_dataset(path_accum)
                if "valid_time" in ds_acc.coords:
                    ds_acc = ds_acc.rename({"valid_time": "time"})
                df_acc = ds_acc.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()

                df_acc["precipitation_mm"]  = df_acc["tp"]   * 1000  if "tp"   in df_acc.columns else 0.0
                df_acc["solar_radiation_W"] = df_acc["ssrd"] / 3600  if "ssrd" in df_acc.columns else 0.0
                df_acc = df_acc[["time", "precipitation_mm", "solar_radiation_W"]]

                df_month = pd.merge(df_inst, df_acc, on="time", how="inner")
                all_monthly.append(df_month)

                ds_inst.close()
                ds_acc.close()

            except Exception as exc:
                print(f"  Warning: could not read ERA5 {zone} {year}-{month_str}: {exc}")

    if not all_monthly:
        return pd.DataFrame()

    df_weather = pd.concat(all_monthly, ignore_index=True)
    df_weather["timestamp"] = pd.to_datetime(df_weather["time"])
    df_weather = df_weather.set_index("timestamp").sort_index().drop(columns=["time"])

    if df_weather.index.tz is None:
        df_weather.index = df_weather.index.tz_localize("UTC")
    else:
        df_weather.index = df_weather.index.tz_convert("UTC")

    return df_weather


# ------------------------------------------------------------------
# Price data preparation
# ------------------------------------------------------------------

def prepare_price_data(
    zone: str,
    price_clean_dir: Path = Path("data/clean"),
) -> pd.DataFrame:
    """Load the wide-format preprocessed price CSV and return a UTC-indexed hourly DataFrame.

    Handles Europe/Vienna → UTC conversion including DST spring-forward and fall-back.

    Args:
        zone: Zone label matching the CSV filename (e.g. "DK1").
        price_clean_dir: Directory containing {zone}_preprocessed.csv files.

    Returns:
        DataFrame with UTC DatetimeIndex and column [price_eur_mwh].
    """
    path = Path(price_clean_dir) / f"{zone}_preprocessed.csv"
    if not path.exists():
        print(f"  Price file not found: {path}")
        return pd.DataFrame()

    df_wide = pd.read_csv(path)
    df_long = df_wide.melt(id_vars=["date"], var_name="hour_str", value_name="price_eur_mwh")
    df_long["hour"] = df_long["hour_str"].str.replace("h", "").astype(int)
    df_long["timestamp_naive"] = pd.to_datetime(df_long["date"]) + pd.to_timedelta(df_long["hour"], unit="h")
    df_long = df_long.set_index("timestamp_naive").sort_index()
    df_long = df_long[~df_long.index.duplicated(keep="first")]

    before = len(df_long)
    try:
        df_long.index = df_long.index.tz_localize(
            "Europe/Vienna", ambiguous="infer", nonexistent="shift_forward"
        ).tz_convert("UTC")
    except Exception:
        temp_index = df_long.index.tz_localize("Europe/Vienna", ambiguous="NaT", nonexistent="NaT")
        df_long = df_long[~temp_index.isna()]
        df_long.index = temp_index[~temp_index.isna()].tz_convert("UTC")

    if before != len(df_long):
        print(f"  Note: dropped {before - len(df_long)} rows during DST conversion for {zone}")

    return df_long[["price_eur_mwh"]]


# ------------------------------------------------------------------
# Masterset construction
# ------------------------------------------------------------------

def build_masterset(
    zone: str,
    df_price: pd.DataFrame,
    df_weather: pd.DataFrame,
    output_dir: Path = Path("data/processed"),
) -> pd.DataFrame:
    """Merge price and weather data, validate, and write the masterset CSV.

    Args:
        zone: Zone label used in the output filename.
        df_price: UTC-indexed price DataFrame (from prepare_price_data).
        df_weather: UTC-indexed weather DataFrame (from process_era5_netcdf).
        output_dir: Directory for the output CSV file.

    Returns:
        Merged masterset DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_price   = df_price.copy()
    df_weather = df_weather.copy()

    # Align indices to UTC
    if df_price.index.tz is None:
        df_price.index = df_price.index.tz_localize("UTC")
    if df_weather.index.tz is None:
        df_weather.index = df_weather.index.tz_localize("UTC")

    masterset = pd.merge(
        df_price, df_weather,
        left_index=True, right_index=True,
        how="inner"
    )

    # Basic sanity assertions
    assert masterset["temperature_2m"].between(-50, 50).all(), \
        f"{zone}: temperature out of range [-50, 50]°C"
    assert (masterset["wind_speed_10m"] >= 0).all(), \
        f"{zone}: negative wind speed detected"
    assert (masterset["solar_radiation_W"] >= 0).all(), \
        f"{zone}: negative solar radiation detected"

    out_path = output_dir / f"{zone}_masterset.csv"
    masterset.to_csv(out_path)
    print(f"  Saved {zone} masterset → {out_path}  ({masterset.shape})")
    return masterset


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def days_not_24_entries(df: pd.DataFrame, name: str = "df") -> pd.Series:
    """Report and return days that do not have exactly 24 hourly entries.

    Args:
        df: DataFrame whose first column contains datetime strings (date + hour).
        name: Label printed in the diagnostic output.

    Returns:
        Series of (date → count) for days with ≠ 24 entries.
    """
    col0 = df.columns[0]
    day = df[col0].astype(str).str.slice(0, 10)
    counts = day.value_counts().sort_index()
    bad = counts[counts != 24]
    if bad.empty:
        print(f"{name}: all days have 24 entries.")
    else:
        print(f"\n{name}: days with != 24 entries\n{bad}")
    return bad


# ------------------------------------------------------------------
# Layer 1 feature engineering (daily aggregation + lags + dummies)
# ------------------------------------------------------------------

def make_daily_values(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an hourly masterset DataFrame to daily weather and price summaries.

    Args:
        df: Hourly DataFrame with a 'date_time' column and columns
            [price_eur_mwh, temperature_2m, wind_speed_10m,
             precipitation_mm, solar_radiation_W].

    Returns:
        Daily DataFrame with columns [date, avg_temperature, avg_wind_speed,
        sum_precipitation, sum_solar_radiation, avg_price].
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date_time"].astype(str).str.slice(0, 10), errors="coerce")

    daily = (
        df.groupby("date", as_index=False)
          .agg(
              avg_temperature=("temperature_2m", "mean"),
              avg_wind_speed=("wind_speed_10m", "mean"),
              sum_precipitation=("precipitation_mm", "sum"),
              sum_solar_radiation=("solar_radiation_W", "sum"),
              avg_price=("price_eur_mwh", "mean"),
          )
          .sort_values("date")
          .reset_index(drop=True)
    )
    return daily


def make_cluster_pred(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Add 3-day lagged price features and drop the original avg_price column.

    Args:
        df_daily: Daily DataFrame from make_daily_values.

    Returns:
        Daily DataFrame with avg_price replaced by avg_price_1/2/3 lags.
        The first 3 rows (NaN lags) are dropped.
    """
    df = df_daily.copy().sort_values("date").reset_index(drop=True)
    df["avg_price_1"] = df["avg_price"].shift(1)
    df["avg_price_2"] = df["avg_price"].shift(2)
    df["avg_price_3"] = df["avg_price"].shift(3)
    df = df.dropna().reset_index(drop=True)
    df = df.drop(columns=["avg_price"])
    return df


def add_weekend_dummy(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add a binary is_weekend column (1 = Saturday/Sunday, 0 = weekday).

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.

    Returns:
        Copy of df with an additional is_weekend column.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["is_weekend"] = (df[date_col].dt.dayofweek >= 5).astype(int)
    return df


def add_season_dummies(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add three binary season columns (spring, summer, autumn); winter is implicit.

    Season boundaries are based on astronomical seasons:
      Spring: Mar 21 – Jun 20
      Summer: Jun 21 – Sep 20
      Autumn: Sep 21 – Dec 20
      Winter: everything else (all three dummies = 0)

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.

    Returns:
        Copy of df with season_spring, season_summer, season_autumn columns added.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    doy = df[date_col].dt.dayofyear

    spring_start = pd.Timestamp("2001-03-21").dayofyear
    summer_start = pd.Timestamp("2001-06-21").dayofyear
    autumn_start = pd.Timestamp("2001-09-21").dayofyear
    winter_start = pd.Timestamp("2001-12-21").dayofyear

    df["season_spring"] = ((doy >= spring_start) & (doy < summer_start)).astype(int)
    df["season_summer"] = ((doy >= summer_start) & (doy < autumn_start)).astype(int)
    df["season_autumn"] = ((doy >= autumn_start) & (doy < winter_start)).astype(int)
    return df


# ------------------------------------------------------------------
# Layer 2 feature engineering (precipitation rolling windows)
# ------------------------------------------------------------------

def add_precipitation_last_x_days(df: pd.DataFrame, x: int) -> pd.DataFrame:
    """Add a column with the rolling sum of precipitation over the previous x days.

    The window excludes "today" (shift(1) before rolling) so there is no look-ahead.
    All 24 hourly rows for a given date receive the same daily rolling value.

    Args:
        df: Hourly DataFrame with a 'date' column and 'precipitation_mm'.
        x: Rolling window size in days (e.g. 3, 7, 14, 20).

    Returns:
        Copy of df with column precipitation_last_{x}_days added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    daily = (
        df.groupby("date", as_index=False)["precipitation_mm"]
          .sum()
          .sort_values("date")
    )

    out_col = f"precipitation_last_{x}_days"
    daily[out_col] = (
        daily["precipitation_mm"]
        .shift(1)
        .rolling(window=x, min_periods=x)
        .sum()
    )

    df = df.merge(daily[["date", out_col]], on="date", how="left")
    return df


# ------------------------------------------------------------------
# Cluster label attachment
# ------------------------------------------------------------------

def attach_cluster(df_pred: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """Join cluster label assignments onto a feature DataFrame by date.

    Performs an inner join on a 'YYYY-MM-DD' date key, so rows whose dates fall
    outside the clustering window are silently dropped.

    Args:
        df_pred: Daily feature DataFrame with a 'date' column.
        clusters: DataFrame with columns [date, cluster] (from *_date_cluster.csv).

    Returns:
        df_pred with an additional 'cluster' column; unmatched rows are dropped.
    """
    df_pred  = df_pred.copy()
    clusters = clusters.copy()

    df_pred["_date_key"]  = df_pred["date"].astype(str).str.slice(0, 10)
    clusters["_date_key"] = clusters["date"].astype(str).str.slice(0, 10)

    n_before = len(df_pred)
    out = df_pred.merge(clusters[["_date_key", "cluster"]], on="_date_key", how="inner")
    n_after = len(out)
    if n_before != n_after:
        print(f"attach_cluster: dropped {n_before - n_after} rows with no matching cluster date.")

    return out.drop(columns=["_date_key"])

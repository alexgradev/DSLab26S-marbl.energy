"""Loading, validation, and column-specific imputation for hourly mastersets."""
from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from src.config import ZoneConfig



def load_and_validate(zone_cfg: ZoneConfig) -> pd.DataFrame:
    """Load the hourly masterset CSV and return a UTC-indexed DataFrame.

    Reads zone_cfg.data_path, parses the timestamp column strictly to UTC
    (handles CET/CEST offset strings and tz-naive fallback), sets it as
    the index, asserts monotonic order, and reindexes to a continuous
    hourly UTC range. Duplicate timestamps are deduplicated keeping first.

    Args:
        zone_cfg: Zone configuration providing data_path and timestamp_col.

    Returns:
        Hourly DataFrame indexed by tz-aware UTC timestamps, gap-filled
        with NaN rows for any missing hours.

    Raises:
        ValueError: If the index is not monotonically increasing after
            parsing, or if the timestamp column is missing.
    """
    df = pd.read_csv(zone_cfg.data_path)

    # timestamp_naive contains timezone-aware ISO strings (CET/CEST offset); parse directly to UTC.
    dt_col = "timestamp_naive"
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    df = df.set_index(dt_col)
    df.index.name = "timestamp"

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
        print("Note: index was tz-naive, localized to UTC.")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not monotonically increasing.")

    print(f"Shape:        {df.shape}")
    print(f"Index min:    {df.index.min()}")
    print(f"Index max:    {df.index.max()}")
    print(f"Index dtype:  {df.index.dtype}")
    print(f"Monotonic:    {df.index.is_monotonic_increasing}")

    expected_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h", tz="UTC")

    missing_hours = expected_idx.difference(df.index)
    dup_mask  = df.index.duplicated()
    dup_hours = df.index[dup_mask]

    print(f"Expected hours:   {len(expected_idx):,}")
    print(f"Actual hours:     {len(df):,}")
    print(f"Missing hours:    {len(missing_hours):,}")
    print(f"Duplicate hours:  {len(dup_hours):,}")

    if len(dup_hours) > 0:
        print("\nDuplicate timestamps:")
        print(df[df.index.isin(dup_hours)].to_string())
        df = df[~df.index.duplicated(keep="first")]
        print(f"Kept first occurrence of each duplicate. Remaining rows: {len(df):,}")

    if len(missing_hours) > 0:
        gaps = []
        gap_start = missing_hours[0]
        prev = missing_hours[0]
        for ts in missing_hours[1:]:
            if (ts - prev).total_seconds() > 3600:
                gaps.append((gap_start, prev, int((prev - gap_start).total_seconds() / 3600) + 1))
                gap_start = ts
            prev = ts
        gaps.append((gap_start, prev, int((prev - gap_start).total_seconds() / 3600) + 1))
        print(f"\n{len(gaps)} gap block(s):")
        for g in gaps:
            flag = "  WARNING: manual inspection required" if g[2] > 3 else ""
            print(f"  {g[0]} -> {g[1]}  ({g[2]}h){flag}")


    df = df.reindex(expected_idx)
    print(f"\nAfter reindex: {df.shape}")
    return df




def report_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-column null counts and percentages, sorted descending.

    Pure inspection - does not mutate input. Intended for printing or
    logging before imputation runs.

    Args:
        df_hourly: Hourly DataFrame (post-validation).

    Returns:
        DataFrame with columns ['null_count', 'null_pct'], one row per
        input column, sorted by null_pct descending.
    """
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(3)
    miss_table  = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct}).sort_values("null_pct", ascending=False)
    print("Missingness table:")
    print(miss_table.to_string())


   


def impute(df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
    """Apply column-specific imputation policies.

    Dispatches per column class using zone_cfg.schema:
      - Price column: linear time-interpolation, limit=3h.
      - Weather columns: linear time-interpolation, limit=6h.
      - Commodity columns: forward-fill, limit=72h (weekend closures).
      - ENTSO-E forecast columns: 5-step cascade
          (i)   solar night-hours zero-fill,
          (ii)  short-gap interpolation (≤3h),
          (iii) lag-24h fill,
          (iv)  lag-168h fill,
          (v)   warn on residual nulls.

    After imputation, days where the price column is still NaN are dropped.

    Args:
        df_hourly: Validated hourly DataFrame from load_and_validate.
        zone_cfg: Provides schema slots that drive dispatch.

    Returns:
        Imputed hourly DataFrame, same shape minus dropped unresolvable
        price-NaN days.
    """
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(3)
    miss_table  = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct}).sort_values("null_pct", ascending=False)
    print("Missingness table:")
    print(miss_table.to_string())

    def _print_null_blocks(series, name):
        is_null = series.isnull()
        if not is_null.any():
            return
        null_idx = series.index[is_null]
        block_start = null_idx[0]
        prev = null_idx[0]
        for ts in null_idx[1:]:
            if (ts - prev).total_seconds() > 3600:
                length = int((prev - block_start).total_seconds() / 3600) + 1
                print(f"  {name}: {block_start} -> {prev}  ({length}h)")
                block_start = ts
            prev = ts
        length = int((prev - block_start).total_seconds() / 3600) + 1
        print(f"  {name}: {block_start} -> {prev}  ({length}h)")

    price_col = zone_cfg.schema.price


    _entso_candidates = [
        "load_forecast_Forecasted Load",
        "wind_solar_forecast_Solar",
        "wind_solar_forecast_Wind Offshore",
        "wind_solar_forecast_Wind Onshore",
        "generation_forecast",
    ]
    entso_cols     = [c for c in _entso_candidates if c in df.columns]
    weather_cols   = [c for c in df.columns if any(k in c.lower() for k in ["wind", "temp", "solar", "precip", "radiation"])
                    and c not in entso_cols]
    commodity_cols = [c for c in df.columns if c not in weather_cols and c != price_col and c not in entso_cols]

    # Price: linear interpolation for gaps <= 3h.
    price_null_before = df[price_col].isnull().sum()
    df[price_col] = df[price_col].interpolate(method="time", limit=3)
    print(f"\nPrice: interpolated {price_null_before - df[price_col].isnull().sum()} gaps (<= 3h).")

    # Weather: linear interpolation for gaps <= 6h.
    for col in weather_cols:
        before = df[col].isnull().sum()
        df[col] = df[col].interpolate(method="time", limit=6)
        after  = df[col].isnull().sum()
        if before > 0:
            print(f"Weather {col}: interpolated {before - after}, {after} remaining.")

    # Commodity: forward-fill up to 72h (weekend exchange closures).
    for col in commodity_cols:
        before = df[col].isnull().sum()
        df[col] = df[col].ffill(limit=72)
        after  = df[col].isnull().sum()
        if before > 0:
            print(f"Commodity {col}: forward-filled {before - after}, {after} remaining.")
            if after > 0:
                print(f"  WARNING: {after} remaining nulls in {col} -- gap exceeds 3 days.")

    # ---------------------------------------------------------------------------
    # ENTSO-E forecast columns imputation
    # Five-step cascade: nighttime zero-fill (solar) → short interpolation →
    # lag-24h → lag-168h → generation_forecast load-ratio proxy.
    # ---------------------------------------------------------------------------
    _solar_col      = "wind_solar_forecast_Solar"
    _load_col       = "load_forecast_Forecasted Load"
    _gen_col        = "generation_forecast"

    for col in entso_cols:
        # Step 1 - Solar nighttime zero-fill (solar column only).
        if col == _solar_col:
            night_mask = df[col].isnull() & ((df.index.hour < 5) | (df.index.hour >= 22))
            n_night = night_mask.sum()
            if n_night > 0:
                df.loc[night_mask, col] = 0.0
                print(f"ENTSO-E {col}: nighttime zero-fill filled {n_night}, {df[col].isnull().sum()} remaining.")

        # Step 2 - Short gap interpolation (≤ 3h).
        before = df[col].isnull().sum()
        if before > 0:
            df[col] = df[col].interpolate(method="time", limit=3)
            after = df[col].isnull().sum()
            if before > after:
                print(f"ENTSO-E {col}: interpolation filled {before - after}, {after} remaining.")

        # Step 3 - Medium gap fill via lag-24h.
        before = df[col].isnull().sum()
        if before > 0:
            lag24 = df[col].shift(24)
            mask  = df[col].isnull() & lag24.notna()
            df.loc[mask, col] = lag24[mask]
            after = df[col].isnull().sum()
            if before > after:
                print(f"ENTSO-E {col}: lag-24h filled {before - after}, {after} remaining.")

        # Step 4 - Long gap fill via lag-168h.
        before = df[col].isnull().sum()
        if before > 0:
            lag168 = df[col].shift(168)
            mask   = df[col].isnull() & lag168.notna()
            df.loc[mask, col] = lag168[mask]
            after = df[col].isnull().sum()
            if before > after:
                print(f"ENTSO-E {col}: lag-168h filled {before - after}, {after} remaining.")
            if after > 0:
                print(f"  WARNING: {after} remaining nulls in {col} after all lag fills.")
                _print_null_blocks(df[col], col)


    print("\nMissingness after imputation:")
    remaining = df.isnull().sum()
    print(remaining[remaining > 0] if remaining.any() else "  None.")

    # Drop days where price is still NaN after imputation.
    price_null_days = df[df[price_col].isnull()].index.normalize().unique()
    if len(price_null_days) > 0:
        print(f"\nDropped {len(price_null_days)} days due to unresolvable price gaps: {price_null_days.date.tolist()}")
        df = df[~df.index.normalize().isin(price_null_days)]
    else:
        print("No days dropped for unresolvable price gaps.")
    
    return df


def summarize_validated_data(df: pd.DataFrame, zone_cfg: ZoneConfig) -> dict:
    """Build a one-shot summary dict of the cleaned hourly dataset.

    Includes total hours, total days, date range, price moments, fraction
    of negative-price hours, and per-column mean/std. Intended for logging
    at the end of the data stage.

    Args:
        df_hourly: Imputed hourly DataFrame.
        zone_cfg: Provides the price column name.

    Returns:
        Dictionary of summary statistics; downstream callers can pretty-print.
    """
    
    df_hourly = df.copy()

    total_hours = len(df_hourly)
    total_days  = df_hourly.index.normalize().nunique()
    price_s     = df_hourly[zone_cfg.schema.price]
    neg_pct     = 100 * (price_s < 0).sum() / price_s.notna().sum()

    print("=== Validated Data Summary ===")
    print(f"Total hours:           {total_hours:,}")
    print(f"Total days:            {total_days:,}")
    print(f"Date range:            {df_hourly.index.min().date()} -> {df_hourly.index.max().date()}")
    print(f"Price mean:            {price_s.mean():.2f} EUR/MWh")
    print(f"Price std:             {price_s.std():.2f} EUR/MWh")
    print(f"Price min:             {price_s.min():.2f} EUR/MWh")
    print(f"Price max:             {price_s.max():.2f} EUR/MWh")
    print(f"Negative price hours:  {neg_pct:.2f}%")
    print()
    for col in df_hourly.columns:
        if col != zone_cfg.schema.price:
            s = df_hourly[col].dropna()
            print(f"  {col:<28s} mean={s.mean():.3f}  std={s.std():.3f}")

    print(f"\ndf_hourly ready: {df_hourly.shape}")

"""
src/ingestion.py — Functions for downloading raw price and weather data.

Design rules:
  - No network calls on import: every function that requires a client
    (EntsoePandasClient, cdsapi.Client) accepts it as a parameter.
  - ERA5_VARIABLES and ERA5_ZONE_COORDS are pure data constants (safe at module level).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

__all__ = [
    "fetch_entsoe_data",
    "fetch_entsoe_range",
    "download_era5_month",
    "_maybe_unzip_era5_file",
    "fetch_city_forecast",
    "process_and_save_forecasts",
    "ERA5_VARIABLES",
    "ERA5_ZONE_COORDS",
]

# ------------------------------------------------------------------
# Module-level constants (pure data — no network activity on import)
# ------------------------------------------------------------------

ERA5_VARIABLES: list[str] = [
    "2m_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]

ERA5_ZONE_COORDS: dict[str, dict[str, float]] = {
    "DK1": {"north": 57.846,  "west":  7.8714, "south": 54.7545, "east": 11.0739},
    "ES":  {"north": 43.733,  "west": -9.5816, "south": 36.0242, "east":  3.4922},
    "NO2": {"north": 59.361,  "west":  5.3171, "south": 57.9584, "east":  9.8517},
}


# ------------------------------------------------------------------
# ENTSO-E price ingestion (primary path: entsoe-py library)
# ------------------------------------------------------------------

def fetch_entsoe_data(country_code: str, start: str, end: str, client) -> pd.DataFrame:
    """Fetch day-ahead prices for one chunk (≤ 6 months) via entsoe-py.

    Args:
        country_code: ENTSO-E zone code (e.g. "10YDK-1--------W").
        start: Start date string "YYYYMMDD".
        end: End date string "YYYYMMDD" (exclusive — API convention).
        client: Initialised EntsoePandasClient instance.

    Returns:
        DataFrame indexed by date with columns h00…h23 (EUR/MWh, hourly mean).
    """
    start_ts = pd.Timestamp(start, tz="Europe/Brussels")
    end_ts   = pd.Timestamp(end,   tz="Europe/Brussels")

    if (end_ts - start_ts) > pd.Timedelta(days=190):
        raise ValueError("Time interval is too large — max ~6 months per chunk.")

    series = client.query_day_ahead_prices(country_code=country_code, start=start_ts, end=end_ts)
    df = series.to_frame(name="price")
    df["date"] = df.index.date
    df["time"] = df.index.strftime("%H:%M")

    df = df.pivot_table(index="date", columns="time", values="price")
    df = df.sort_index(axis=1)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    df.columns = pd.to_datetime(df.columns, format="%H:%M").time
    df.columns.name = "time"

    # Collapse to hourly (handles 15-min or 30-min resolution)
    df_hourly = df.groupby([t.hour for t in df.columns], axis=1).mean().round(2)
    df_hourly.columns = [f"h{h:02d}" for h in df_hourly.columns]
    df_hourly = df_hourly.sort_index(axis=1)

    # Drop trailing incomplete day (last row often has only the 00:00 value)
    if df_hourly.iloc[-1].isna().any():
        df_hourly = df_hourly.iloc[:-1]

    # DST spring-forward fix: fill the missing hour in March via forward-fill
    if (df_hourly.index.month == 3).any():
        march = df_hourly.index.month == 3
        if df_hourly.loc[march].isna().any().any():
            df_hourly.loc[march] = df_hourly.loc[march].ffill(axis=1)

    if df_hourly.isna().any().any():
        nan_positions = df_hourly[df_hourly.isna()].stack().index.tolist()
        raise ValueError(f"NaN values remain after DST fix: {nan_positions[:5]}")

    return df_hourly


def fetch_entsoe_range(
    zone: str,
    zone_code: str,
    start: str,
    end: str,
    client,
    out_dir: Path = Path("data/clean"),
) -> pd.DataFrame:
    """Fetch day-ahead prices for an arbitrary date range in 6-month chunks.

    Writes the result to ``out_dir/{zone}_preprocessed.csv`` and returns the DataFrame.

    Args:
        zone: Human-readable zone name used for the output filename (e.g. "DK1").
        zone_code: ENTSO-E API zone code (e.g. "10YDK-1--------W").
        start: Start date "YYYYMMDD".
        end: End date "YYYYMMDD".
        client: Initialised EntsoePandasClient instance.
        out_dir: Directory for the output CSV file.

    Returns:
        Combined wide-format DataFrame (date × h00…h23).
    """
    start_date = pd.to_datetime(start, format="%Y%m%d")
    end_date   = pd.to_datetime(end,   format="%Y%m%d")

    chunks = []
    cur = start_date
    while cur <= end_date:
        chunk_end = min(cur + pd.DateOffset(months=6) - pd.Timedelta(days=1), end_date)
        df_chunk = fetch_entsoe_data(
            country_code=zone_code,
            start=cur.strftime("%Y%m%d"),
            end=(chunk_end + pd.Timedelta(days=1)).strftime("%Y%m%d"),
            client=client,
        )
        chunks.append(df_chunk)
        cur = chunk_end + pd.Timedelta(days=1)

    full_df = pd.concat(chunks)
    full_df = full_df[~full_df.index.duplicated(keep="first")].sort_index()

    out_path = Path(out_dir) / f"{zone}_preprocessed.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path)
    print(f"Saved: {out_path}")
    return full_df


# ------------------------------------------------------------------
# ERA5 weather ingestion
# ------------------------------------------------------------------

def _maybe_unzip_era5_file(file_path: Path) -> Path:
    """If *file_path* is a ZIP archive, extract the NetCDF inside and return its path.

    Returns the original path if not a ZIP.
    """
    import zipfile

    with open(file_path, "rb") as fh:
        signature = fh.read(4)

    # ZIP magic bytes: PK (0x50 0x4B)
    if signature[:2] != b"PK":
        return file_path

    with zipfile.ZipFile(file_path, "r") as zf:
        nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_names:
            raise ValueError(f"No .nc file found inside ZIP: {file_path}")
        zf.extractall(file_path.parent)

    # Rename to preserve the expected stem pattern
    extracted = file_path.parent / nc_names[0]
    stem = file_path.stem
    if "instant" in stem or "accum" in stem:
        target = file_path.with_suffix(".nc")
    else:
        target = extracted
    if extracted != target:
        extracted.rename(target)

    file_path.unlink(missing_ok=True)  # remove ZIP
    return target


def download_era5_month(
    year: str,
    month: str,
    zone: str,
    coords: dict[str, float],
    client,
    variables: list[str] | None = None,
    output_dir: Path = Path("data/raw/weather"),
) -> str:
    """Download one (zone, year, month) ERA5 file pair via the CDS API.

    Skips the download if both instant and accum files already exist.

    Args:
        year: 4-digit year string.
        month: 2-digit month string (e.g. "03").
        zone: Zone label used in the filename (e.g. "DK1").
        coords: Bounding box dict with keys north/west/south/east.
        client: Initialised cdsapi.Client instance.
        variables: List of ERA5 variable names (defaults to ERA5_VARIABLES).
        output_dir: Where to write the .nc files.

    Returns:
        Status string: "downloaded", "skipped", or "failed".
    """
    if variables is None:
        variables = ERA5_VARIABLES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"era5_{zone}_{year}_{month}"
    path_instant = output_dir / f"{stem}_instant.nc"
    path_accum   = output_dir / f"{stem}_accum.nc"

    if path_instant.exists() and path_accum.exists():
        return "skipped"

    # Instantaneous variables (temp, wind)
    instant_vars = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]
    # Accumulated variables (precipitation, solar radiation)
    accum_vars = [
        "total_precipitation",
        "surface_solar_radiation_downwards",
    ]

    try:
        for var_group, suffix in [(instant_vars, "instant"), (accum_vars, "accum")]:
            target = output_dir / f"{stem}_{suffix}.nc"
            if target.exists():
                continue
            request = {
                "product_type": "reanalysis",
                "variable": var_group,
                "year": year,
                "month": month,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [
                    coords["north"], coords["west"],
                    coords["south"], coords["east"],
                ],
                "format": "netcdf",
            }
            client.retrieve("reanalysis-era5-single-levels", request, str(target))
            _maybe_unzip_era5_file(target)

        return "downloaded"

    except Exception as exc:
        print(f"  ERA5 download failed for {zone} {year}-{month}: {exc}")
        return "failed"


# ------------------------------------------------------------------
# Live weather forecast ingestion (WeatherAPI.com)
# ------------------------------------------------------------------

def fetch_city_forecast(
    city: str,
    api_key: str,
    base_url: str = "http://api.weatherapi.com/v1/forecast.json",
    days: int = 3,
) -> pd.DataFrame:
    """Fetch a 3-day hourly weather forecast for one city from WeatherAPI.com.

    Args:
        city: City name (e.g. "Madrid").
        api_key: WeatherAPI.com API key.
        base_url: API endpoint URL.
        days: Number of forecast days (1–3).

    Returns:
        DataFrame with columns [time_local, temperature_2m, precipitation_mm,
        wind_speed_10m, solar_radiation_W].
    """
    import requests

    params = {"key": api_key, "q": city, "days": days, "aqi": "no", "alerts": "no"}
    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for day in data["forecast"]["forecastday"]:
        for hour in day["hour"]:
            solar = hour.get("uv", 0) * 25.0  # UV index fallback
            if "solar_rad" in hour:
                solar = hour["solar_rad"]
            rows.append(
                {
                    "time_local": hour["time"],
                    "temperature_2m": hour["temp_c"],
                    "precipitation_mm": hour["precip_mm"],
                    "wind_kph": hour["wind_kph"],
                    "solar_radiation_W": solar,
                }
            )

    df = pd.DataFrame(rows)
    df["wind_speed_10m"] = df["wind_kph"] / 3.6  # km/h → m/s
    df = df.drop(columns=["wind_kph"])
    return df


def process_and_save_forecasts(
    zone_locations: dict[str, list[str]],
    api_key: str,
    base_url: str = "http://api.weatherapi.com/v1/forecast.json",
    output_dir: Path = Path("data/live"),
) -> None:
    """Fetch forecasts for all cities per zone, aggregate spatially, and write CSVs.

    Args:
        zone_locations: Dict mapping zone name → list of representative cities.
        api_key: WeatherAPI.com API key.
        base_url: API endpoint URL.
        output_dir: Directory where {ZONE}_forecast.csv files are written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for zone, cities in zone_locations.items():
        city_dfs = []
        for city in cities:
            try:
                df_city = fetch_city_forecast(city, api_key, base_url)
                city_dfs.append(df_city)
            except Exception as exc:
                print(f"  Warning: could not fetch {city} ({exc})")

        if not city_dfs:
            print(f"  No data fetched for {zone} — skipping.")
            continue

        # Spatial mean across cities
        df_zone = (
            pd.concat(city_dfs)
            .groupby("time_local", as_index=False)
            [["temperature_2m", "precipitation_mm", "wind_speed_10m", "solar_radiation_W"]]
            .mean()
        )
        out_path = output_dir / f"{zone}_forecast.csv"
        df_zone.to_csv(out_path, index=False)
        print(f"  Saved {zone} forecast -> {out_path}")

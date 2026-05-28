"""Daily feature engineering — every feature uses only T-1 information."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import holidays
from src.config import ZoneConfig


def build_daily_features(
    df_hourly: pd.DataFrame,
    zone_cfg: ZoneConfig,
    spike_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Construct the daily feature matrix from an hourly input.

    Every feature for day T is derived exclusively from information
    observable before noon on day T-1. The function is the single source
    of truth for feature definitions and is called identically by the
    training pipeline, the walk-forward loop, and any future inference path.

    Steps (each guarded by schema availability):
      A. Daily price aggregates (mean, std, min, max, neg_frac).
      B. Lag-1 price stats, intraday range, peak/trough hours with
         cyclic sin/cos encoding.
      C. Rolling-3/-7 momentum on lagged price mean.
      D. Persistence streaks (neg_streak; spike_streak if threshold given).
      E. Commodity features (TTF, CO2, Brent) with returns and spread proxies.
      F. ERA5 weather aggregates + transforms (temp_sq, wind_power_proxy,
         wind_chill).
      F.5/F.6. ENTSO-E daily-mean and intraday-block forecasts, plus
         derived shape features (ramps, residual load, wind diurnal amp).
      F.7. Compound spike-risk interactions (TTF/CO2 momentum × evening
         residual load).
      G. Calendar features (weekend, holiday via zone_cfg.holiday_factory,
         month/dow cyclic).
      H. Append zone-specific features via zone_cfg.extra_feature_builders.
      I. Drop intermediate raw columns; dropna for rolling startup.

    Args:
        df_hourly: UTC-indexed hourly DataFrame, post-imputation.
        zone_cfg: Drives column dispatch, holiday provider, block timezone,
            and extra feature builders.
        spike_threshold: Hourly-price spike threshold (EUR/MWh). When None,
            spike_streak_length is omitted (bootstrap case before the fold
            threshold is computed).

    Returns:
        Daily DataFrame indexed by tz-naive calendar dates, one row per day,
        with all engineered features and no remaining NaN rows.
    """

    cols = df_hourly.columns.tolist()
    h = df_hourly.copy()
    h["date"] = pd.to_datetime(h.index.normalize().date)

    pc = zone_cfg.schema.price

    # Step A: daily price statistics -- become lag-1 features after shifting.
    daily_price = h.groupby("date")[pc].agg(
        price_mean="mean",
        price_std="std",
        price_min="min",
        price_max="max",
        price_neg_frac=lambda x: (x < 0).mean(),
    )
    daily_price["was_negative"] = (daily_price["price_neg_frac"] > 0).astype(int)
    d = daily_price.copy()

    # Step B: lag price statistics by 1 day.
    for col in ["price_mean", "price_std", "price_min", "price_max", "price_neg_frac", "was_negative"]:
        d[col + "_lag1"] = d[col].shift(1)

    # Step B1: price range and volatility
    # price_range_lag1 — intraday price spread, more spike-sensitive than std
    d["price_range_lag1"] = d["price_max_lag1"] - d["price_min_lag1"]

    # Step B2
    d["price_peak_hour"]   = h.groupby("date")[pc].idxmax().dt.hour
    d["price_trough_hour"] = h.groupby("date")[pc].idxmin().dt.hour

    d["pivot_hour_peak_lag1"]   = d["price_peak_hour"].shift(1)
    d["pivot_hour_trough_lag1"] = d["price_trough_hour"].shift(1)


    # Cyclic encodings: 2π / 24-hour period.
    d["pivot_peak_sin"]   = np.sin(2 * np.pi * d["pivot_hour_peak_lag1"]   / 24)
    d["pivot_peak_cos"]   = np.cos(2 * np.pi * d["pivot_hour_peak_lag1"]   / 24)
    d["pivot_trough_sin"] = np.sin(2 * np.pi * d["pivot_hour_trough_lag1"] / 24)
    d["pivot_trough_cos"] = np.cos(2 * np.pi * d["pivot_hour_trough_lag1"] / 24)

    # Step C: rolling momentum (shift first so day T is excluded from the window).
    shifted_mean = d["price_mean"].shift(1)
    d["price_roll3_mean"] = shifted_mean.rolling(3, min_periods=1).mean()
    d["price_roll7_mean"] = shifted_mean.rolling(7, min_periods=1).mean()

    # Step D: persistence features.
    had_neg = d["was_negative"].values
    streak  = np.zeros(len(had_neg), dtype=int)
    for i in range(1, len(had_neg)):
        streak[i] = streak[i - 1] + 1 if had_neg[i - 1] else 0
    d["neg_streak_length"] = streak

    if spike_threshold is not None:
        was_spike = (d["price_max"] > spike_threshold).astype(int).values
        s_streak  = np.zeros(len(was_spike), dtype=int)
        for i in range(1, len(was_spike)):
            s_streak[i] = s_streak[i - 1] + 1 if was_spike[i - 1] else 0
        d["spike_streak_length"] = s_streak
    else:
        print("Note: spike_threshold is None -- spike_streak_length omitted.")

    # Step E: commodity features using actual column names.
    missing_comm = []
    has_ttf   = "ttf_gas_eur_mwh"   in cols
    has_co2   = "co2_eua_eur_tonne"  in cols
    has_brent = "brent_usd_bbl"      in cols

    if has_ttf:
        d["ttf_raw"]       = h.groupby("date")["ttf_gas_eur_mwh"].first()
        d["ttf_lag1"]      = d["ttf_raw"].shift(1)
        d["ttf_return_1d"] = d["ttf_raw"].shift(1) / d["ttf_raw"].shift(2) - 1
        d["ttf_return_3d"] = d["ttf_raw"].shift(1) / d["ttf_raw"].shift(4) - 1
        d["ttf_roll7_std"] = d["ttf_raw"].shift(1).rolling(7, min_periods=2).std()
    else:
        missing_comm.append("ttf_gas_eur_mwh")

    if has_co2:
        d["co2_raw"]       = h.groupby("date")["co2_eua_eur_tonne"].first()
        d["co2_lag1"]      = d["co2_raw"].shift(1)
        d["co2_return_1d"] = d["co2_raw"].shift(1) / d["co2_raw"].shift(2) - 1
    else:
        missing_comm.append("co2_eua_eur_tonne")

    if has_brent:
        d["brent_raw"]       = h.groupby("date")["brent_usd_bbl"].first()
        d["brent_return_3d"] = d["brent_raw"].shift(1) / d["brent_raw"].shift(4) - 1
    else:
        missing_comm.append("brent_usd_bbl")

    if has_ttf and has_co2:
        d["spark_spread_proxy"] = (
            d["price_mean_lag1"] - (d["ttf_lag1"] * 2.0) - (d["co2_lag1"] * 0.4)
        )

    # co2_gas_ratio — fuel switching pressure signal
    if has_ttf and has_co2:
        d["co2_gas_ratio"] = d["co2_lag1"] / (d["ttf_lag1"] + 1e-6)
        

    if missing_comm:
        print(f"Warning: commodity columns not found and skipped: {missing_comm}")




    # Step F: weather features for day T (ERA5 actuals as NWP proxy).
    weather_map = {
        "wind_speed_10m":   ("wind_mean_t",  "mean"),
        "temperature_2m":   ("temp_mean_t",  "mean"),
        "solar_radiation_W":("solar_mean_t", "mean"),
        "precipitation_mm": ("precip_sum_t", "sum"),
    }
    for src_col, (out_col, agg) in weather_map.items():
        if src_col in cols:
            if agg == "mean":
                d[out_col] = h.groupby("date")[src_col].mean()
            else:
                d[out_col] = h.groupby("date")[src_col].sum()


    #######################################################

    # Step F2: weather-derived transformations
    # temp_sq — nonlinear heating/cooling demand response
    if "temp_mean_t" in d.columns:
        d["temp_sq"] = d["temp_mean_t"] ** 2

    # wind_power_proxy — cubic wind-to-power curve
    if "wind_mean_t" in d.columns:
        d["wind_power_proxy"] = d["wind_mean_t"] ** 3

    # wind_chill — heating demand beyond raw temperature
    if "temp_mean_t" in d.columns and "wind_mean_t" in d.columns:
        d["wind_chill"] = d["temp_mean_t"] - (0.33 * d["wind_mean_t"])

    # # precip_roll7/30 — hydro reservoir accumulation proxy (NO2) - ADD AS ZONE-CONDITIONAL FEATURE E.G., ZONE = "NO2"
    # if "precip_sum_t" in d.columns:
    #     d["precip_roll7"]  = d["precip_sum_t"].shift(1).rolling(7,  min_periods=3).sum()
    #     d["precip_roll30"] = d["precip_sum_t"].shift(1).rolling(30, min_periods=10).sum()


    #######################################################


    # ---------------------------------------------------------------------------
    # Step F.5: ENTSO-E forecast features for day T.
    # These are same-day forecasts published before noon on day T-1 and are
    # therefore observable at feature-construction time without leakage.
    # Aggregated as daily mean (MW semantics). No lag or shift applied.
    # Degrades gracefully: missing source columns cause only their dependent
    # features to be skipped; a single consolidated warning is printed.
    # ---------------------------------------------------------------------------
    _load_col     = "load_forecast_Forecasted Load"
    _solar_col    = "wind_solar_forecast_Solar"
    _offshore_col = "wind_solar_forecast_Wind Offshore"
    _onshore_col  = "wind_solar_forecast_Wind Onshore"
    _gen_col      = "generation_forecast"

    _entso_src = {
        "load":     _load_col,
        "solar":    _solar_col,
        "offshore": _offshore_col,
        "onshore":  _onshore_col,
        "gen":      _gen_col,
    }

    # Aggregate each available source column to daily mean and stage in d for later dropping.
    _ea = {}
    for _key, _src in _entso_src.items():
        if _src in cols:
            _ea[_key] = h.groupby("date")[_src].mean()
            d[_src] = _ea[_key]

    # Intermediate wind_total (offshore + onshore). Not added to d as a feature.
    _has_wind = "offshore" in _ea and "onshore" in _ea
    _wind_total = _ea["offshore"] + _ea["onshore"] if _has_wind else None

    _entso_skipped = []



    # supply_margin = generation / load
    if all(k in _ea for k in ("load", "gen")):
        d["supply_margin"] = _ea["gen"] / _ea["load"]
    else:
        _entso_skipped.append("supply_margin")


    # conventional_gen = generation - (wind_total + solar)
    if all(k in _ea for k in ("gen", "offshore", "onshore", "solar")):
        d["conventional_gen"] = _ea["gen"] - (_wind_total + _ea["solar"])
    else:
        _entso_skipped.append("conventional_gen")

    
    # Step F3: cross-group interaction features
    # gas_x_residual — nonlinear spike mechanism: tight supply + expensive gas
    if "ttf_lag1" in d.columns and "residual_load" in d.columns:
        d["gas_x_residual"] = d["ttf_lag1"] * d["residual_load"]

    # co2_x_thermal — carbon cost exposure weighted by thermal generation share
    if "co2_lag1" in d.columns and "renewable_fraction" in d.columns:
        d["co2_x_thermal"] = d["co2_lag1"] * (1 - d["renewable_fraction"])

    if _entso_skipped:
        print(f"Warning: ENTSO-E source columns missing -- skipped features: {_entso_skipped}")



    # ---------------------------------------------------------------------------
    # Step F.6: ENTSO-E block aggregates for day T.
    # Breaks each hourly forecast series into four intraday blocks to inject
    # shape information into the daily feature matrix. Without this, the 24
    # hour-specific Stage-2 regressors receive identical exogenous inputs and
    # must recover intraday variation purely from hour-index specialization.
    #
    # Blocks (UTC, matching h.index timezone):
    #   overnight    : 00:00-06:00
    #   morning ramp : 06:00-10:00
    #   midday       : 10:00-16:00
    #   evening      : 16:00-21:00
    #   night        : 21:00-24:00
    
    #
    # NOTE on timezone: for DK1 the day-ahead trading day is CET/CEST. UTC blocks
    # are offset by 1-2h from local time. Acceptable for a DK1-only run; switch
    # to local-hour blocking when extending to ES (CET) and NO2 (CET).
    # ---------------------------------------------------------------------------
    _blocks = {
        "overnight":  range(0,  6),   # 00-05 : baseload, wind-dominated, flat pricing
        "morning":    range(6,  10),  # 06-09 : demand ramp (absorbs missing h09)
        "midday":     range(10, 16),  # 10-15 : max renewable penetration, duck curve  (absorbs missing h15)      
        "evening":    range(16, 21),  # 16-20 : solar off, demand peak, spike zone (absorbs missing h16 and h20)
        "night":      range(21, 24),  # 21-23 : demand decay, night wind  (absorbs missing h21-h23)
#        
    }

    _block_sources = {
        "load":     _load_col,
        "solar":    _solar_col,
        "wind_off": _offshore_col,
        "wind_on":  _onshore_col,
        "gen":      _gen_col,
    }

    _block_skipped = []
    _hour_idx = h.index.hour

    for _short, _src in _block_sources.items():
        if _src not in cols:
            _block_skipped.append(_short)
            continue
        for _block_name, _hr_range in _blocks.items():
            _mask = _hour_idx.isin(list(_hr_range))
            d[f"{_short}_{_block_name}"] = h.loc[_mask].groupby("date")[_src].mean()

    

    # Combined wind_total per block (offshore + onshore).
    _has_wind_blocks = all(k not in _block_skipped for k in ("wind_off", "wind_on"))
    if _has_wind_blocks:
        for _block_name in _blocks:
            d[f"wind_total_{_block_name}"] = (
                d[f"wind_off_{_block_name}"] + d[f"wind_on_{_block_name}"]
            )

    # Residual load per block: load - wind_total - solar.
    # Block-wise residual is more diagnostic than the daily mean because the
    # regime that sets price differs sharply across blocks:
    #   - midday residual_load drives negative-price (Leaf A) risk
    #   - evening residual_load drives spike (Leaf B) risk
    _has_residual_blocks = (
        _has_wind_blocks
        and "load"  not in _block_skipped
        and "solar" not in _block_skipped
    )
    if _has_residual_blocks:
        for _block_name in _blocks:
            d[f"residual_load_{_block_name}"] = (
                d[f"load_{_block_name}"]
                - d[f"wind_total_{_block_name}"]
                - d[f"solar_{_block_name}"]
            )

    # Structural shape features derived from the blocks.

    # evening_ramp_load: gross load ramp from midday to evening peak.
    if {"load_midday", "load_evening"}.issubset(d.columns):
        d["evening_ramp_load"] = d["load_evening"] - d["load_midday"]

    # morning_ramp_load: gross load ramp from overnight to morning.
    if {"load_overnight", "load_morning"}.issubset(d.columns):
        d["morning_ramp_load"] = d["load_morning"] - d["load_overnight"]

    # evening_ramp_residual: ramp the dispatchable fleet must cover, net of
    # renewables. More direct than gross load ramp for evening-peak stress.
    if {"residual_load_midday", "residual_load_evening"}.issubset(d.columns):
        d["evening_ramp_residual"] = (
            d["residual_load_evening"] - d["residual_load_midday"]
        )


    if {"residual_load_evening", "residual_load_night"}.issubset(d.columns):
        d["night_decay_residual"] = d["residual_load_night"] - d["residual_load_evening"]

    # solar_concentration: peakedness of the solar profile. High values flag
    # sharp midday peaks -> stronger duck-curve dip risk. Marginal for DK1,
    # central for ES.
    if "solar_midday" in d.columns and _solar_col in cols:
        _daily_solar = h.groupby("date")[_solar_col].mean()
        d["solar_concentration"] = d["solar_midday"] / (_daily_solar + 1e-6)

    # wind_diurnal_amp: spread between strongest and weakest wind blocks. High
    # amplitude flags wind dropout before the evening peak -> compound spike risk.
    _wind_block_cols = [
        f"wind_total_{b}" for b in _blocks
        if f"wind_total_{b}" in d.columns
    ]
    if len(_wind_block_cols) >= 2:
        d["wind_diurnal_amp"] = (
            d[_wind_block_cols].max(axis=1) - d[_wind_block_cols].min(axis=1)
        )



    if _block_skipped:
        print(f"Warning: ENTSO-E block sources missing -- skipped series: {_block_skipped}")

    

    # ---------------------------------------------------------------------------
    # Step F.7: compound spike-risk interactions.
    # Bridges PC8 (commodity momentum) and PC7 (evening supply stress), which
    # load on orthogonal axes in the PCA with no feature currently connecting
    # them. These interactions encode the compound Leaf B trigger: gas costs
    # surging at the same time the evening block is specifically tight.
    # Inputs are lag-safe: ttf_return_* use shift(1)/shift(2); residual_load_evening
    # is a day-T forecast slot (Step F.6).
    # ---------------------------------------------------------------------------
    _compound_skipped = []

    if "ttf_return_1d" in d.columns and "residual_load_evening" in d.columns:
        d["ttf_momentum_x_rl_evening"] = d["ttf_return_1d"] * d["residual_load_evening"]
    else:
        _compound_skipped.append("ttf_momentum_x_rl_evening")

    if "ttf_return_3d" in d.columns and "residual_load_evening" in d.columns:
        d["ttf_momentum3d_x_rl_evening"] = d["ttf_return_3d"] * d["residual_load_evening"]
    else:
        _compound_skipped.append("ttf_momentum3d_x_rl_evening")

    if "co2_return_1d" in d.columns and "residual_load_evening" in d.columns:
        d["co2_momentum_x_rl_evening"] = d["co2_return_1d"] * d["residual_load_evening"]
    else:
        _compound_skipped.append("co2_momentum_x_rl_evening")

    if "ttf_return_1d" in d.columns and "co2_return_1d" in d.columns and "residual_load_evening" in d.columns:
        d["commodity_momentum_composite"] = (
            (d["ttf_return_1d"] + d["co2_return_1d"]) / 2.0
        ) * d["residual_load_evening"]
    else:
        _compound_skipped.append("commodity_momentum_composite")

    if _compound_skipped:
        print(f"Warning: compound spike features skipped (missing inputs): {_compound_skipped}")



    # Step G: calendar features for day T.
    idx = d.index
    dk_holidays       = holidays.Denmark()
    d["is_weekend"]   = (idx.dayofweek >= 5).astype(int)
    d["is_holiday"]   = idx.map(lambda dt: int(dt.date() in dk_holidays))
    d["month_sin"]    = np.sin(2 * np.pi * idx.month / 12)
    d["month_cos"]    = np.cos(2 * np.pi * idx.month / 12)
    d["dow_sin"]      = np.sin(2 * np.pi * idx.dayofweek / 7)
    d["dow_cos"]      = np.cos(2 * np.pi * idx.dayofweek / 7)

    # Drop raw intermediate columns before returning.
    drop_raw = ["price_mean", "price_std", "price_min", "price_max",
                "price_neg_frac", "was_negative", "price_peak_hour", "price_trough_hour", 
                "ttf_raw", "co2_raw", "brent_raw","wind_on", "wind_off", 
                _load_col, _solar_col, _offshore_col, _onshore_col, _gen_col]
    d = d.drop(columns=[c for c in drop_raw if c in d.columns])

    # Step H: drop NaN rows from rolling startup and remaining gaps.
    before = len(d)
    d = d.dropna()
    dropped = before - len(d)
    print(f"build_daily_features: dropped {dropped} rows. "
          f"Remaining: {len(d)} ({d.index.min().date()} -> {d.index.max().date()})")
    return d
    


def build_price_pivot_lag1(
    df_hourly: pd.DataFrame,
    df_daily: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> pd.DataFrame:
    """Build the 24-hour lagged price pivot used inside PCA + classifier.

    Row T contains the 24 hourly prices that cleared on day T-1. Columns
    are named h00..h23. Used as the shape signal input to PCA before FCM
    clustering and concatenated to the classifier feature matrix.

    Args:
        df_hourly: UTC-indexed hourly DataFrame, post-imputation.
        zone_cfg: Provides the price column name.

    Returns:
        DataFrame indexed by calendar date with 24 columns (h00..h23).
        First row will be NaN (no T-1 reference); callers handle this.
    """
    h_pivot = df_hourly.copy()
    h_pivot["date"] = pd.to_datetime(h_pivot.index.normalize().date)
    h_pivot["hour"] = h_pivot.index.hour

    price_pivot = h_pivot.pivot_table(
        index="date", columns="hour", values=zone_cfg.schema.price , aggfunc="mean"
    )
    price_pivot.columns = [f"h{c:02d}" for c in price_pivot.columns]

    # Shift by 1 day so row T contains day T-1 price curve.
    price_pivot_lag1 = price_pivot.shift(1)

    # Align to df_daily index.
    price_pivot_lag1 = price_pivot_lag1.reindex(df_daily.index)

    print(f"price_pivot_lag1 shape: {price_pivot_lag1.shape}")
    print(f"NaN count:              {price_pivot_lag1.isnull().sum().sum()}")
    print("\nFirst 3 rows:")
    print(price_pivot_lag1.head(3).to_string())

    return price_pivot_lag1


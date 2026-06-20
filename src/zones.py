"""Registry of pre-configured ZoneConfig instances."""
from pathlib import Path
import holidays
from src.config import ZoneConfig, ColumnSchema
from src.features_extra import precip_rolling_features  # zone-specific feature
from src.config import DATA_DIR, ZoneConfig, ColumnSchema
from typing import Tuple

# add fold definitions, as Spain requires a separate definition due to lack of data
_DEFAULT_FOLDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2023-01-01", "2025-12-31", "2026-01-01", "2026-03-28"),
)

_ES_FOLDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2024-01-01", "2025-12-31", "2026-01-01", "2026-03-28"),
)

# Weather and price column names are uniform across the masterset; ENTSO-E
# and commodity slots vary by data availability per zone.
_SHARED_WEATHER = dict(
    temperature="temperature_2m",
    wind_speed="wind_speed_10m",
    solar_radiation="solar_radiation_W",
    precipitation="precipitation_mm",
)
_SHARED_COMMODITY = dict(
    ttf_gas="ttf_gas_eur_mwh",
    co2_eua="co2_eua_eur_tonne",
    brent="brent_usd_bbl",
)
_SHARED_ENTSO = dict(
    load_forecast="load_forecast_Forecasted Load",
    solar_forecast="wind_solar_forecast_Solar",
    wind_offshore_forecast="wind_solar_forecast_Wind Offshore",
    wind_onshore_forecast="wind_solar_forecast_Wind Onshore",
    generation_forecast="generation_forecast",
)

ZONES: dict[str, ZoneConfig] = {
    "DK1": ZoneConfig(
        zone="DK1",
        data_path = DATA_DIR/"DK1_masterset_enriched_all.csv",
        local_tz="Europe/Copenhagen",
        timestamp_col="timestamp_naive",
        schema=ColumnSchema(price="price_eur_mwh", **_SHARED_WEATHER,
                            **_SHARED_COMMODITY, **_SHARED_ENTSO),
        holiday_factory=holidays.Denmark,
        block_tz="UTC",  # preserves original notebook behavior,
        fold_definitions=_DEFAULT_FOLDS
    ),
    "ES": ZoneConfig(
        zone="ES",
        data_path = DATA_DIR/"ES_masterset_enriched_all.csv",
        local_tz="Europe/Madrid",
        timestamp_col="timestamp_naive",
        schema=ColumnSchema(price="price_eur_mwh", **_SHARED_WEATHER,
                            **_SHARED_COMMODITY, **_SHARED_ENTSO),
        holiday_factory=holidays.Spain,
        block_tz="UTC",
        extra_feature_builders=(precip_rolling_features,),  # hydro proxy
        fold_definitions=_ES_FOLDS
    ),
    "NO2": ZoneConfig(
        zone="NO2",
        data_path = DATA_DIR/"NO2_masterset_enriched_all.csv",
        local_tz="Europe/Oslo",
        timestamp_col="timestamp_naive",
        schema=ColumnSchema(price="price_eur_mwh", **_SHARED_WEATHER,
                            **_SHARED_COMMODITY, **_SHARED_ENTSO),
        holiday_factory=holidays.Norway,
        block_tz="UTC",
        extra_feature_builders=(precip_rolling_features,),
        fold_definitions=_DEFAULT_FOLDS  # hydro proxy
    ),


    "DE-LU": ZoneConfig(
        zone="DE-LU",
        data_path = DATA_DIR/"DE-LU_masterset_enriched_all.csv",
        local_tz="Europe/Berlin",
        timestamp_col="timestamp_naive",
        schema=ColumnSchema(price="price_eur_mwh", **_SHARED_WEATHER,
                            **_SHARED_COMMODITY, **_SHARED_ENTSO),
        holiday_factory=holidays.Germany,
        block_tz="UTC",
        fold_definitions=_DEFAULT_FOLDS
    ),

    "FR": ZoneConfig(
        zone="FR",
        data_path = DATA_DIR/"FR_masterset_enriched_all.csv",
        local_tz="Europe/Paris",
        timestamp_col="timestamp_naive",
        schema=ColumnSchema(price="price_eur_mwh", **_SHARED_WEATHER,
                            **_SHARED_COMMODITY, **_SHARED_ENTSO),
        holiday_factory=holidays.France,
        block_tz="UTC",
        extra_feature_builders=(precip_rolling_features,),  # hydro proxy
        fold_definitions=_DEFAULT_FOLDS
    ),
}
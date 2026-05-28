"""Frozen configuration objects for zones and model hyperparameters."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple, Optional, Any
import holidays
import pandas as pd


# sets up the working directory to be the project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data_new"

@dataclass(frozen=True)
class ColumnSchema:
    """Names of zone-dependent input columns in the hourly masterset.

    Slots set to None mean the column is absent for this zone; downstream
    feature blocks that depend on it are skipped gracefully (matching the
    existing `if col in cols:` pattern in build_daily_features).

    Attributes:
        price: Hourly day-ahead price column.
        temperature: ERA5 2m temperature column.
        wind_speed: ERA5 10m wind speed column.
        solar_radiation: ERA5 solar radiation column.
        precipitation: ERA5 precipitation column.
        ttf_gas: TTF natural gas settlement price column (or None).
        co2_eua: EUA CO2 price column (or None).
        brent: Brent crude price column (or None).
        load_forecast: ENTSO-E day-ahead load forecast column (or None).
        solar_forecast: ENTSO-E day-ahead solar forecast column (or None).
        wind_offshore_forecast: ENTSO-E offshore wind forecast (or None).
        wind_onshore_forecast: ENTSO-E onshore wind forecast (or None).
        generation_forecast: ENTSO-E total generation forecast (or None).
    """
    price: str
    temperature: Optional[str] = None
    wind_speed: Optional[str] = None
    solar_radiation: Optional[str] = None
    precipitation: Optional[str] = None
    ttf_gas: Optional[str] = None
    co2_eua: Optional[str] = None
    brent: Optional[str] = None
    load_forecast: Optional[str] = None
    solar_forecast: Optional[str] = None
    wind_offshore_forecast: Optional[str] = None
    wind_onshore_forecast: Optional[str] = None
    generation_forecast: Optional[str] = None


@dataclass(frozen=True)
class ZoneConfig:
    """All zone-dependent inputs the pipeline needs.

    Attributes:
        zone: Short bidding-zone code (e.g. "DK1", "ES", "NO2").
        data_path: Path to the enriched hourly masterset CSV for this zone.
        local_tz: IANA timezone string for the zone's day-ahead market
            (e.g. "Europe/Copenhagen"). Used for holiday boundaries and
            intraday block aggregation if block_tz == "local".
        block_tz: Timezone used for intraday block aggregation. "UTC"
            preserves the original notebook's behavior; "local" uses
            local_tz. Default "UTC" for backward compatibility.
        timestamp_col: Name of the raw timestamp column in the CSV.
        schema: Per-zone column-name registry.
        holiday_factory: Zero-arg callable returning a holidays.HolidayBase
            instance for the zone's country (e.g. holidays.Denmark).
        extra_feature_builders: Optional tuple of pure functions appended
            to build_daily_features. Signature:
                (df_daily, df_hourly, schema) -> df_daily.
            Used for zone-specific features like NO2 hydro proxies.
    """
    zone: str
    data_path: Path
    local_tz: str
    timestamp_col: str
    schema: ColumnSchema
    holiday_factory: Callable[[], "holidays.HolidayBase"]
    block_tz: str = "UTC"
    extra_feature_builders: Tuple[Callable[..., pd.DataFrame], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelConfig:
    """All model hyperparameters and walk-forward fold definitions.

    Attributes:
        spike_pct: Percentile (0-100) of hourly training prices used as
            the Leaf B spike threshold.
        leaf_a_neg_frac_threshold: Fraction of negative-price hours per
            day above which the day is labelled Leaf A.
        fkm_k_normal: Number of fuzzy clusters fit on normal days.
        fkm_m: Fuzziness exponent for FCM.
        fkm_error: FCM convergence tolerance.
        fkm_maxiter: FCM maximum iterations.
        pca_variance_target: Cumulative explained-variance ratio for
            PCA component selection.
        min_cluster_size: Below this size a regime falls back to RidgeCV
            experts instead of XGBoost.
        clf_params: Kwargs forwarded to XGBClassifier (L2 multinomial).
        reg_params: Kwargs forwarded to XGBRegressor (expert models).
        l1_C_grid: Inverse-regularization grid for L1 LogisticRegressionCV.
        l1_cv_folds: Number of CV folds for L1 LogisticRegressionCV.
        ridge_alpha_grid: Alphas for the RidgeCV fallback experts.
        random_seed: Single seed propagated to numpy, FCM, PCA, XGBoost,
            and LogReg.
        fold_definitions: List of (tr_start, tr_end, te_start, te_end)
            ISO-date strings.
        models_dir: Output directory for serialized models / arrays.
        results_dir: Output directory for CSVs and figures.
    """
    spike_pct: int = 95
    leaf_a_neg_frac_threshold: float = 0.15
    fkm_k_normal: int = 2
    fkm_m: float = 2.0
    fkm_error: float = 0.005
    fkm_maxiter: int = 1000
    pca_variance_target: float = 0.875
    min_cluster_size: int = 80
    clf_params: dict = field(default_factory=lambda: {
        "max_depth":        3,     # 8 leaves per tree — cannot memorise 355 samples
        "n_estimators":     300,
        "learning_rate":    0.03,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 8,     # no split on fewer than 8 samples
        "reg_lambda":       2.0,
        "reg_alpha":        0.1,
        })
    reg_params: dict = field(default_factory=lambda: {
        "max_depth":        3,
        "n_estimators":     300,
        "learning_rate":    0.03,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "reg_lambda":       2.0,
        "reg_alpha":        0.1,})
    l1_C_grid: Tuple[float, ...] = (0.001, 0.01, 0.1, 1, 10, 100)
    l1_cv_folds: int = 5
    ridge_alpha_grid: Tuple[float, ...] = (10, 50, 100, 500, 1000, 5000)
    random_seed: int = 42
    fold_definitions: Tuple[Tuple[str, str, str, str], ...] = (
        ("2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
        ("2023-01-01", "2025-12-31", "2026-01-01", "2026-04-30"),
    )
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")
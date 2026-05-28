# Day-Ahead Pricing Pipeline — Architecture Overview

A regime-switching, hour-by-hour electricity price forecasting pipeline, refactored
from a single monolithic notebook into a modular `src/` package. The notebook is now
a thin orchestration layer; all logic lives in importable modules.

The pipeline is **zone-agnostic**: the same code runs for any bidding zone (DK1, ES,
NO2, DE-LU, FR). Zone-specific details (column names, holidays, data paths) are
injected via a `ZoneConfig` object rather than hardcoded.

---

## Package Structure

```
DSLab26S-marbl.energy/
├── data_new/                       # enriched hourly masterset CSVs (one per zone)
├── results/                        # output figures, prediction CSVs, model cache
├── tests/golden/                   # parity-test reference outputs
├── pipeline_AM_new.ipynb           # orchestration notebook (thin)
└── src/
    ├── __init__.py                 # re-exports ZoneConfig, ColumnSchema, ModelConfig, ZONES
    ├── config.py                   # frozen dataclasses + project paths
    ├── zones.py                    # ZONES registry (one ZoneConfig per bidding zone)
    ├── data.py                     # load, validate, impute hourly data
    ├── features.py                 # daily feature engineering
    ├── features_extra.py           # zone-specific feature builders
    ├── model.py                    # regimes, classifiers, experts, walk-forward
    ├── evaluation.py               # metrics + stratified breakdowns
    └── plotting.py                 # all diagnostic figures
```

---

## The Two Config Objects

Everything zone- or model-dependent flows through two frozen dataclasses defined in
`config.py`. Nothing else in the codebase holds hardcoded zone names or hyperparameters.

**`ZoneConfig`** — one per bidding zone, all registered in `zones.py` under `ZONES`.
Holds the data path, local timezone, holiday provider, and a `ColumnSchema` mapping
abstract roles (price, wind, solar, gas, load forecast, …) to the actual column names
in that zone's CSV. Swapping zones is a one-line change: `zone_cfg = ZONES["ES"]`.

**`ModelConfig`** — a single shared instance holding every hyperparameter: spike
percentile, FCM cluster count, PCA variance target, XGBoost params, Ridge fallback
grid, random seed, and the walk-forward fold definitions.

Because both are `frozen=True`, they cannot be mutated after construction — config is
read-only state threaded explicitly into every function.

---

## Module Responsibilities

### `data.py` — raw data → clean hourly frame
- `load_and_validate` — read CSV, parse to UTC, check hourly frequency, dedupe, reindex.
- `report_missingness` — per-column null inspection.
- `impute` — column-specific imputation (linear for price/weather, forward-fill for
  commodities, 5-step lag cascade for ENTSO-E forecasts), then drop unresolvable days.
- `summarize_validated_data` — one-shot summary of the cleaned dataset.

### `features.py` — clean hourly frame → daily feature matrix
- `build_daily_features` — the single source of truth for feature definitions. Every
  feature for day *T* uses only information observable before noon on day *T−1* (lag-1
  price stats, momentum, commodities, weather proxies, ENTSO-E intraday blocks,
  compound interactions, calendar encodings). Appends any zone-specific features
  registered in `ZoneConfig.extra_feature_builders`.
- `build_price_pivot_lag1` — the 24-hour lagged price curve (one row per day), used as
  the shape signal for PCA and the classifiers.

### `features_extra.py` — zone-specific add-ons
- `precip_rolling_features` — rolling precipitation windows (hydro-reservoir proxy),
  attached only for hydro-influenced zones (NO2, FR) via their `ZoneConfig`.

### `model.py` — features → trained models → predictions
The four-regime architecture: **Leaf A** (negative-price days), **Leaf B** (spike days),
**C0/C1** (normal days, found by Fuzzy C-Means on PCA-reduced features).

- `compute_spike_threshold` — fold-specific Leaf B threshold from training hours only.
- `build_leaf_masks` — assign each training day to A / B / Normal from actual prices.
- `fit_pca_and_fcm` — PCA + Fuzzy C-Means on normal days.
- `assemble_regime_labels` — combine masks + FCM labels into one label series.
- `train_classifier_l1` — binary LogisticRegressionCV gate (Leaf A vs not-A).
- `train_classifier_l2` — multinomial XGBoost gate (B vs C0 vs C1).
- `train_expert_models` — 96 hour-specific regressors (4 regimes × 24 hours), with a
  RidgeCV fallback for under-sized regimes.
- `predict_fold` — hierarchical soft-blend cascade producing the 24h forecast.
- `run_fold` — glues all the above into one per-fold pipeline.
- `walk_forward` — runs `run_fold` across all expanding-window folds; the **single
  public entry point** for training.
- `load_or_train` — cache wrapper: load saved predictions/models, or train if absent.
- `extract_pca_loadings` — post-hoc PCA loadings table for feature audits.

`PCA_EXCLUDE_COLS` (module constant) is the one source of truth for which columns are
excluded from PCA, shared by `fit_pca_and_fcm` and `extract_pca_loadings`.

### `evaluation.py` — predictions → metrics
- `wape`, `mae` — NaN-safe error metrics.
- `attach_naive_baselines` — append naive-1 (yesterday) and naive-2 (7-day mean) curves.
- `reconstruct_actual_regimes` — recover ground-truth regime labels per test day.
- `confusion_matrix_l1`, `confusion_matrix_l2` — gate-level classification metrics.
- `overall_metrics`, `regime_metrics`, `hourly_metrics`, `fold_metrics` — stratified WAPE/MAE.
- `worst_best_days_wape`, `worst_best_days_mae` — best/worst predicted days.
- Module constants `HOUR_COLS`, `PRED_COLS`, `NAIVE1_COLS`, `NAIVE2_COLS` define the
  column-name conventions reused across evaluation and plotting.

### `plotting.py` — predictions → figures
Each function returns a matplotlib `Figure`. Titles use `zone_cfg.zone` so they are
correct for any zone: `plot_price_distribution`, `plot_feature_summary`,
`plot_hourly_errors`, `plot_timeseries_actual_vs_predicted`, `plot_regime_profiles`,
`plot_error_distribution`, `plot_regime_probabilities`, `plot_confusion_matrix`,
`plot_single_day_diagnostics`, and `make_all_diagnostics` (a convenience wrapper).

---

## Pipeline Data Flow

```
ZONES["DK1"] ─┐
              ▼
   data.load_and_validate ──► data.impute ──► df_hourly
                                                  │
                  ┌───────────────────────────────┤
                  ▼                                ▼
   features.build_daily_features        features.build_price_pivot_lag1
                  │                                │
                  └──────────────┬─────────────────┘
                                 ▼
                       model.walk_forward
              (per fold: spike threshold → features → leaf masks →
               PCA+FCM → regime labels → L1+L2 classifiers →
               96 experts → soft-blend prediction)
                                 │
                                 ▼
                        df_predictions, all_fold_models
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                                          ▼
   evaluation.* (WAPE/MAE,                     plotting.* (diagnostics,
   regime/hourly/fold, confusion)              per-day inspection)
```

---

## The Orchestration Notebook

The notebook (`pipeline_AM_new.ipynb`) contains no business logic. A full single-zone
run reduces to:

```python
from src.zones import ZONES
from src.config import ModelConfig
from src import data, features, model, evaluation, plotting

zone_cfg  = ZONES["DK1"]          # swap to "ES", "NO2", "DE-LU", "FR"
model_cfg = ModelConfig()

# Data
df_hourly = data.impute(data.load_and_validate(zone_cfg), zone_cfg)

# Features
df_daily  = features.build_daily_features(df_hourly, zone_cfg)
pivot_lag = features.build_price_pivot_lag1(df_hourly, zone_cfg)

# Model (single entry point; trains all folds or loads cache)
df_preds, all_fold_models = model.load_or_train(
    zone_cfg, model_cfg, df_daily, pivot_lag, df_hourly
)

# Evaluation
df_preds = evaluation.attach_naive_baselines(df_preds, df_hourly, zone_cfg)
df_eval  = evaluation.reconstruct_actual_regimes(
    df_preds, df_hourly, all_fold_models, model_cfg, zone_cfg
)

# Plots
plotting.make_all_diagnostics(df_preds, all_fold_models, df_hourly, zone_cfg, model_cfg)
```

The helper functions inside `model.py` (`compute_spike_threshold`, `build_leaf_masks`,
etc.) are never called directly from the notebook — they are implementation details of
`walk_forward`.

---

## Running Across Multiple Zones

Zone-dependence is configuration that flows *in*, not iteration inside the functions.
To run all five zones, loop at the orchestration level:

```python
for zone_name in ["DK1", "ES", "NO2", "DE-LU", "FR"]:
    zone_cfg  = ZONES[zone_name]
    df_hourly = data.impute(data.load_and_validate(zone_cfg), zone_cfg)
    df_daily  = features.build_daily_features(df_hourly, zone_cfg)
    pivot_lag = features.build_price_pivot_lag1(df_hourly, zone_cfg)
    df_preds, models = model.load_or_train(
        zone_cfg, model_cfg, df_daily, pivot_lag, df_hourly
    )
    # ... evaluate / store per zone
```

Each zone trains an independent model — DK1 never sees ES data. Adding a new zone means
adding one `ZoneConfig` entry to `ZONES`; no pipeline code changes.

---

## Design Principles

- **No global state.** Every function takes `zone_cfg` / `model_cfg` explicitly; the
  frozen dataclasses cannot be mutated.
- **One source of truth.** Column conventions (`HOUR_COLS`, …), PCA exclusions
  (`PCA_EXCLUDE_COLS`), and feature definitions (`build_daily_features`) are each defined
  in exactly one place.
- **Pure functions, side effects at the edges.** Core functions return objects; file I/O
  and plotting display happen in the notebook, not inside the pipeline.
- **Parity-tested.** The refactor reproduces the original DK1 notebook outputs exactly
  (`tests/golden/`), so behavior is provably unchanged before any zone extension.

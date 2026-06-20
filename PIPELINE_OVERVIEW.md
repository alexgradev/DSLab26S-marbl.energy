# Day-Ahead Pricing Pipeline - Architecture Overview

Most of the logic of the pipeline is organised in functions which are grouped in importable .py modules based on their respective context.

There are 5 Master notebooks: one notebook for each bidding zone. These notebooks are called pipeline_{zone}.ipynb. 
The whole modeling pipeline for a bidding zone is executed in its respective Master notebook, where the functions
are called from the .py source files in ./src.



The pipeline is **zone-agnostic**: the same code runs for any bidding zone (DK1, ES,
NO2, DE-LU, FR). Zone-specific details (column names, holidays, data paths) are
injected via a `ZoneConfig` object rather than hardcoded.

---

## Structure

```
DSLab26S-marbl.energy/
├── data_new/                       # hourly masterset CSVs (one per zone)
├── results/                        # output figures, prediction CSVs, model cache
├── tests/golden/                   # parity-test reference outputs
├── pipeline_{country}.ipynb           # master notebook
└── src/
    ├── __init__.py                 # re-exports ZoneConfig, ColumnSchema, ModelConfig, ZONES
    ├── config.py                   # frozen dataclasses + project paths
    ├── zones.py                    # ZONES registry (one ZoneConfig per bidding zone)
    ├── data.py                     # load, validate, impute hourly data
    ├── features.py                 # daily feature engineering
    ├── model.py                    # regimes, classifiers, experts, walk-forward
    ├── evaluation.py               # metrics + stratified breakdowns
    └── plotting.py                 # all diagnostic figures
```

---

## The 3 classes in config.py

Everything zone- or model-dependent flows through 3 frozen dataclasses defined in
`config.py.

**`ZoneConfig`** - one per bidding zone, all registered in `zones.py` under `ZONES`.
Holds the data path, local timezone, holiday provider, walk-forward fold definitions
and a **`ColumnSchema`** class which maps abstract roles (price, wind, solar, gas, load forecast, …) 
to the actual column names in that zone's CSV. 
Swapping zones is a one-line change, e.g., `zone_cfg = ZONES["ES"]`.

**`ModelConfig`** - a single shared instance holding every hyperparameter: spike
percentile, FCM cluster count, PCA variance target, XGBoost params, Ridge fallback
grid, random seed, and the walk-forward fold definitions.

Because the classes are `frozen=True` they cannot be mutated after construction.

---

## Module Responsibilities

### `data.py` - raw data -> clean hourly frame
- `load_and_validate` - read CSV, parse to UTC, check hourly frequency, dedupe, reindex.
- `report_missingness` - per-column null inspection.
- `impute` - column-specific imputation (linear for price/weather, forward-fill for
  commodities, 5-step lag cascade for ENTSO-E forecasts), then drop unresolvable days.
- `summarize_validated_data` - one-shot summary of the cleaned dataset.

### `features.py` - clean hourly frame -> daily feature matrix
- `build_daily_features` - the single source of truth for feature definitions. Every
  feature for day *T* uses only information observable before noon on day *T−1* (lag-1
  price stats, momentum, commodities, weather proxies, ENTSO-E intraday blocks,
  compound interactions, calendar encodings). Appends any zone-specific features
  registered in `ZoneConfig.extra_feature_builders`.
- `build_price_pivot_lag1` - the 24-hour lagged price curve (one row per day), used as
  the shape signal for PCA and the classifiers.


### `model.py` - features -> trained models -> predictions
The four-regime architecture: **Leaf A** (negative-price days), **Leaf B** (spike days),
**C0/C1** (normal days, found by Fuzzy C-Means on PCA-reduced features).

- `compute_spike_threshold` - fold-specific Leaf B threshold from training hours only.
- `build_leaf_masks` - assign each training day to A / B / Normal from actual prices.
- `fit_pca_and_fcm` - PCA + Fuzzy C-Means on normal days.
- `assemble_regime_labels` - combine masks + FCM labels into one label series.
- `train_classifier_l1` - binary LogisticRegressionCV gate (Leaf A vs not-A).
- `train_classifier_l2` - multinomial XGBoost gate (B vs C0 vs C1).
- `train_expert_models` - 96 hour-specific regressors (4 regimes × 24 hours), with a
  RidgeCV fallback for under-sized regimes.
- `predict_fold` - hierarchical soft-blend cascade producing the 24h forecast.
- `run_fold` - glues all the above into one per-fold pipeline.
- `walk_forward` - runs `run_fold` across all expanding-window folds; the **single
  public entry point** for training.
- `load_or_train` - cache wrapper: load saved predictions/models, or train if absent.
- `extract_pca_loadings` - post-hoc PCA loadings table for feature audits.

`PCA_EXCLUDE_COLS` (module constant) is the one source of truth for which columns are
excluded from PCA, shared by `fit_pca_and_fcm` and `extract_pca_loadings`.

### `evaluation.py` - predictions -> metrics
- `wape`, `mae` - NaN-safe error metrics.
- `attach_naive_baselines` - append naive-1 (yesterday) and naive-2 (7-day mean) curves.
- `reconstruct_actual_regimes` - recover ground-truth regime labels per test day.
- `confusion_matrix_l1`, `confusion_matrix_l2` - gate-level classification metrics.
- `overall_metrics`, `regime_metrics`, `hourly_metrics`, `fold_metrics` - stratified WAPE/MAE.
- `worst_best_days_wape`, `worst_best_days_mae` - best/worst predicted days.
- Module constants `HOUR_COLS`, `PRED_COLS`, `NAIVE1_COLS`, `NAIVE2_COLS` define the
  column-name conventions reused across evaluation and plotting.

### `plotting.py` - predictions -> figures
Each function returns a matplotlib `Figure`. Titles use `zone_cfg.zone` so they are
correct for any zone: `plot_price_distribution`, `plot_feature_summary`,
`plot_hourly_errors`, `plot_timeseries_actual_vs_predicted`, `plot_regime_profiles`,
`plot_error_distribution`, `plot_regime_probabilities`, `plot_confusion_matrix`,
`plot_single_day_diagnostics`, and `make_all_diagnostics` (a convenience wrapper).

### `diagnostics.py` - new pipeline -> old model diagnostics bridge
- `to_long_predictions` - reshapes wide daily predictions into a continuous hourly long format.
- `build_part1_inputs` / `build_part3_input` - adapts the long format to meet the input
  contracts of the old diagnostic modules.
- `save_diagnostic_inputs` / `load_diagnostic_inputs` - handles I/O for multi-zone
  notebook execution.
- `run_multizone_diagnostics` / `run_comparison_diagnostics` - convenience wrappers to
  execute both old diagnostic modules in one call.
- `wape_inflation_report` - slices out groups where raw WAPE is heavily distorted by
  near-zero prices.

### `diagnostics_part1.py` - metrics and breakdowns
- `diagnostic_metrics` - computes MAE, raw WAPE, clamped WAPE, and mean signed bias.
- `assign_price_bucket` - classifies hours into negative, near-zero, normal, or spike regimes.
- `run_full_diagnostic` / `run_all_zones` - runs metrics sliced by cluster, season, day
  type, and price bucket, outputting summary tables and charts.
- `cross_zone_comparison` - builds a compact side-by-side table of overall metrics across
  all bidding zones.

### `diagnostics_part2.py` - signed bias analysis
- `plot_hourly_bias` / `plot_weekend_weekday` - visualizes mean residual (Predicted − Actual)
  by hour of day, optionally split by day type.
- `plot_intraday_residual_by_bucket` - facets the hourly bias line plots across the four price buckets.
- `spike_hit_rate_table` / `plot_spike_hit_rate` - evaluates the share of spike hours
  under-predicted by varying threshold severities.
- `run_part2_diagn` - loops through all Part 2 plots into a single run.

---

## Pipeline Data Flow

```
ZONES[country] ─┐
              ▼
   data.load_and_validate ──► data.impute ──► df_hourly
                                                  │
                  ┌───────────────────────────────┤
                  ▼                                ▼
   features.build_daily_features        features.build_price_pivot_lag1
                  │                                │
                  └──────────────┬─────────────────┘
                                 ▼
                        model.load_or_train
              (per fold: spike threshold -> features -> leaf masks ->
               PCA+FCM -> regime labels -> L1+L2 classifiers ->
               96 experts -> soft-blend prediction)
                                 │
                                 ▼
                        df_predictions, all_fold_models
                                 │
            ┌────────────────────┼──────────────────────────────┐
            ▼                    ▼                              ▼
   evaluation.* (WAPE/   plotting.* (daily            diagnostics.* (comparison
   MAE, regime/hourly,   inspection, distributions,   of new model results to
   fold, confusion)      feature summaries)           old model using the same
                                                      diagnostics and test sets)


```

---

## blueprint of the Master notebooks

```python

from src.zones import ZONES
from src.config import ModelConfig
from src import data, features, model, evaluation, plotting, diagnostics

zone_cfg  = ZONES["DK1"]          # swap to "ES", "NO2", "DE-LU", "FR"
model_cfg = ModelConfig()

# Data
df_hourly = data.impute(data.load_and_validate(zone_cfg), zone_cfg)

# Features
df_daily  = features.build_daily_features(df_hourly, zone_cfg)
pivot_lag = features.build_price_pivot_lag1(df_hourly, df_daily, zone_cfg)

# Model (single entry point; trains all folds or loads cache)
df_preds, all_fold_models = model.load_or_train(
    zone_cfg, model_cfg, df_daily, pivot_lag, df_hourly
)

# Evaluation
df_preds = evaluation.attach_naive_baselines(df_preds, df_hourly, zone_cfg)
df_eval  = evaluation.reconstruct_actual_regimes(
    df_preds, df_hourly, all_fold_models, model_cfg, zone_cfg
)

# Plots (Standard Pipeline Diagnostics)
plotting.make_all_diagnostics(df_preds, all_fold_models, df_hourly, zone_cfg, model_cfg)

# Old model diagnostics
# Extract bounds of the final test fold for matched evaluation
_, _, te_start, te_end = zone_cfg.fold_definitions[-1]

diag_results = diagnostics.run_comparison_diagnostics(
    df_preds, 
    zone_cfg, 
    save_dir=f"results/diagnostics_p3/{zone_cfg.zone}",
    te_start=te_start, 
    te_end=te_end
)

# Detect WAPE inflation (distortions from near-zero prices)
inflated_slices = diagnostics.wape_inflation_report(diag_results["summary"])

# Export predictions for the multi-zone cross-comparison notebook
diagnostics.save_diagnostic_inputs(
    df_preds, 
    zone_cfg, 
    te_start, 
    te_end, 
    out_dir="results/diagnostics_multizone"
)
```

---

---


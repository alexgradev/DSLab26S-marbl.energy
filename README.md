# Marbl Energy - Regime-Switching Day-Ahead Electricity Price Forecasting

![Marbl Energy](marbl.png)

A regime-switching forecasting pipeline for **day-ahead electricity prices** across European bidding zones. The pipeline detects latent market regimes (negative-price days, spike days and normal-day sub-clusters) using rule-based system and PCA + K-means clustering, then trains specialised expert models per regime and blends their predictions based on predicted regime probabilities.

Built as part of the [WU Vienna SBWL Data Science Lab](https://www.wu.ac.at/dpkm/teaching/sbwl-data-science)  (Summer Semester 2026) in collaboration with [Marbl Energy](https://www.marbl.energy).

---

## Features

- **Zone-agnostic architecture** - a single codebase covers five European bidding zones (DK1, ES, NO2, DE-LU, FR) via a `ZoneConfig` registry; adding a new zone requires only a config entry
- **Four-regime detection** - Leaf A (negative-price days), Leaf B (spike days), C0/C1 (normal sub-clusters via PCA + K-Means)
- **Hierarchical soft-blend prediction** - binary LogisticRegression -> multinomial XGBoost gate -> 24 hour × 4 regime expert regressors (XGBoost with RidgeCV fallback for small regimes)
- **Walk-forward (expanding-window) cross-validation** - proper time-series evaluation with no future data leakage
- **Rich feature set** - weather (ERA5), commodity prices (TTF gas, EU ETS CO₂, Brent crude), ENTSO-E load/generation/wind/solar forecasts, lagged prices, momentum, calendar encodings and zone-specific hydro proxies
- **Diagnostics** - WAPE/MAE breakdowns by regime, season, hour, fold and price bucket; confusion matrices for both gating layers; spike hit-rate analysis; signed-bias plots
- **Naive baselines** - yesterday's price (naive 1) 7-day rolling mean (naive 2) for fair comparison
- **Parity testing** - regression test comparing new pipeline output against golden reference data

---

## Supported Bidding Zones

| Zone  | Country            | Market Characteristics     |
|-------|--------------------|----------------------------|
| DK1   | Denmark West       | Wind-dominated             |
| ES    | Spain              | Solar-dominated            |
| NO2   | Southern Norway    | Hydro-dominated            |
| DE-LU | Germany/Luxembourg | Mixed thermal + renewables |
| FR    | France             | Nuclear + hydro            |

---

## Prerequisites

- **Python 3.10+**
- **Jupyter Notebook / JupyterLab** 
- ~2 GB disk space for datasets and model artefacts

### API Keys (for data fetching only)

If re-fetching raw data from source (not required if using the provided CSVs):

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) API key (for `entsoe-py`)
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) API key (for `cdsapi`)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/alexgradev/DSLab26S-marbl.energy.git
cd DSLab26S-marbl.energy

# 2. Create a virtual environment
python -m venv marbl_env

# 3. Activate the environment
# Windows (PowerShell):
.\marbl_env\Scripts\Activate.ps1
# Windows (cmd):
marbl_env\Scripts\activate.bat
# macOS / Linux:
source marbl_env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Running the Full Pipeline

Each bidding zone has a dedicated master notebook that orchestrates the entire pipeline - from data loading through evaluation and diagnostics:

```
pipeline_DK1.ipynb    # Denmark West
pipeline_ES.ipynb     # Spain
pipeline_NO2.ipynb    # Southern Norway
pipeline_DE-LU.ipynb  # Germany/Luxembourg
pipeline_FR.ipynb     # France
```

Open any notebook in Jupyter and run all cells:

```bash
jupyter notebook pipeline_DK1.ipynb
```

The notebooks follow a standard sequence:

```python
from src.zones import ZONES
from src.config import ModelConfig
from src import data, features, model, evaluation, plotting, diagnostics

zone_cfg  = ZONES["DK1"]
model_cfg = ModelConfig()

# Data loading & imputation
df_hourly = data.impute(data.load_and_validate(zone_cfg), zone_cfg)

# Feature engineering
df_daily  = features.build_daily_features(df_hourly, zone_cfg)
pivot_lag = features.build_price_pivot_lag1(df_hourly, df_daily, zone_cfg)

# Train (or load cached) models across all walk-forward folds
df_preds, all_fold_models = model.load_or_train(
    zone_cfg, model_cfg, df_daily, pivot_lag, df_hourly
)

# Evaluation & diagnostics
df_preds = evaluation.attach_naive_baselines(df_preds, df_hourly, zone_cfg)
plotting.make_all_diagnostics(df_preds, all_fold_models, df_hourly, zone_cfg, model_cfg)
```

### Cross-Zone Diagnostics

The `diagnostics_comparison_to_old.ipynb` notebook runs multi-zone comparison diagnostics, evaluating the new pipeline against the inherited (previous semester) model.

### Parity Test

To verify that data-loading and feature-engineering changes haven't altered outputs:

```bash
python tests/parity.py
```

---

## Project Structure

```
DSLab26S-marbl.energy/
├── pipeline_{zone}.ipynb                # Master notebooks (one per bidding zone)
├── diagnostics_comparison_to_old.ipynb  # Cross-zone comparison with inherited model
├── requirements.txt                     # Pinned Python dependencies
├── PIPELINE_OVERVIEW.md                 # Detailed architecture documentation
│
├── src/                                 # Core pipeline modules
│   ├── __init__.py                      # Public API: ZoneConfig, ModelConfig, ZONES
│   ├── config.py                        # Frozen dataclasses (ZoneConfig, ColumnSchema, ModelConfig)
│   ├── zones.py                         # ZONES registry - one ZoneConfig per bidding zone
│   ├── data.py                          # Load, validate, impute hourly masterset
│   ├── features.py                      # Daily feature engineering (lag-1, momentum, calendar, etc.)
│   ├── features_extra.py                # Zone-specific feature builders (e.g. precipitation rolling)
│   ├── model.py                         # Regime detection, classifiers, experts, walk-forward CV
│   ├── diagnostics.py                   # Bridge between new pipeline and old diagnostic modules
│   ├── diagnostics_part1.py             # Metrics & breakdowns (MAE, WAPE, price buckets)
│   └── diagnostics_part2.py             # Signed-bias analysis & spike hit-rate evaluation
│
├── data_new/                            # Enriched hourly masterset CSVs (one per zone)
│   ├── DK1_masterset_enriched_all.csv
│   ├── ES_masterset_enriched_all.csv
│   ├── NO2_masterset_enriched_all.csv
│   ├── DE-LU_masterset_enriched_all.csv
│   └── FR_masterset_enriched_all.csv
│
├── models/                              # Cached model artefacts (scalers, centroids, thresholds)
├── results/                             # Output figures, prediction CSVs (gitignored)
├── tests/                               # Parity tests against golden reference data
│   └── parity.py
│
├── inherited_pipeline/                  # Previous semester's code (reference only)
│   └── summary/                         # Summarised version of the old pipeline
│
└── archive/                             # Superseded notebook versions
```

---

## Key Dependencies

| Package          | Purpose                                    |
|------------------|--------------------------------------------|
| `pandas`         | Data manipulation and time-series handling |
| `numpy` / `scipy`| Numerical computation                     |
| `scikit-learn`   | PCA, LogisticRegressionCV, RidgeCV         |
| `xgboost`        | Gradient-boosted classifiers and regressors|
| `scikit-fuzzy`   | Clusterin                    |
| `entsoe-py`      | ENTSO-E Transparency Platform API client   |
| `cdsapi`         | Copernicus Climate Data Store API client    |
| `matplotlib`     | Diagnostic plotting                        |
| `altair`         | Interactive visualisations                 |
| `holidays`       | Country-specific holiday calendars         |

See [requirements.txt](requirements.txt) for the full pinned dependency list.

---

## Data

The pipeline consumes **enriched hourly masterset CSVs** located in `data_new/`. Each CSV contains:

- **Day-ahead prices** (EUR/MWh) sourced from ENTSO-E
- **ERA5 weather variables** - temperature, wind speed, solar radiation, precipitation (from Copernicus)
- **Commodity prices** - TTF natural gas, EU ETS CO₂ allowances, Brent crude
- **ENTSO-E forecasts** - day-ahead load, solar generation, wind (onshore + offshore), total generation

Data spans 2023–2026 (ES starts from 2024 due to data availability).

---

## Walk-Forward Cross-Validation

The pipeline uses expanding-window walk-forward validation to respect temporal ordering:

| Fold | Training Window        | Test Window              |
|------|------------------------|--------------------------|
| 1    | 2023-01-01 -> 2023-12-31| 2024-01-01 -> 2024-12-31  |
| 2    | 2023-01-01 -> 2024-12-31| 2025-01-01 -> 2025-12-31  |
| 3    | 2023-01-01 -> 2025-12-31| 2026-01-01 -> 2026-03-28  |

Spain uses a reduced fold structure (folds 2–3 only) due to later data availability.

---

## Background

This project is a direct follow-up to a previous semester's work (Dieringer, Körbel, Gomez Valverde, Klaric). The prior group built an initial pipeline using Ward hierarchical clustering and a two-layer XGBoost architecture. This iteration addresses critical methodological issues - most notably **data leakage** in clustering and evaluation, **feature poverty** and the absence of proper **time-series cross-validation** - while extending coverage to five bidding zones and enriching the feature set with fundamental market drivers.

For a detailed architecture walkthrough, see [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md).

---

## Team

| Name | Responsibility |
|------|----------------|
| Pavle Cvijanovic | Data Sourcing and Preprocessing |
| Aleksandar Gradev | Research and Modelling |
| Aleksandar Milosavljevic | Research and Modelling |
| Vincent Skakala | Data Sourcing and Preprocessing |

---


## License

This project was developed for academic purposes as part of the WU Vienna Data Science Lab.


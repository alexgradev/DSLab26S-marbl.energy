"""
src/models.py — XGBoost model training, saving, and inference.

Contains:
  - Layer 1: XGBoost cluster classifier (train_xgb_classifier, save_cluster_model_bundle)
  - Layer 2: Per-cluster XGBoost price regressor (xgb_price_within_cluster)
  - Pipeline assembly: predict_cluster_probabilities, weighted_price_prediction
  - Naive baselines: naive_approach_one, train_naive_weather_model
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

__all__ = [
    "train_xgb_classifier",
    "save_cluster_model_bundle",
    "xgb_price_within_cluster",
    "predict_cluster_probabilities",
    "weighted_price_prediction",
    "naive_approach_one",
    "train_naive_weather_model",
]


# ------------------------------------------------------------------
# Layer 1 — Cluster classifier
# ------------------------------------------------------------------

def train_xgb_classifier(
    df: pd.DataFrame,
    name: str = "country",
    out_dir: Path = Path("data/metrics"),
    random_state: int = 42,
) -> tuple:
    """Train an XGBoost multi-class classifier to predict daily market regimes.

    Uses a chronological 70/15/15 split. Labels are 1-encoded internally via
    LabelEncoder so XGBoost receives 0-based classes.

    Args:
        df: Daily DataFrame with columns [date, cluster, *features].
        name: Short country label for logging and report filenames (e.g. "DK").
        out_dir: Directory where xgb_eval_{name}.csv is written.
        random_state: XGBoost random seed.

    Returns:
        (model, feature_cols, label_encoder, splits_dict, report_df)
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report, log_loss
    from sklearn.preprocessing import LabelEncoder

    df = df.copy().sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["cluster"])
    df["cluster"] = df["cluster"].astype(int)

    feature_cols = [c for c in df.columns if c not in ["date", "cluster"]]
    X = df[feature_cols]

    le = LabelEncoder()
    y = le.fit_transform(df["cluster"])  # 0-based labels

    n = len(df)
    n_train = int(np.floor(n * 0.70))
    n_val   = int(np.floor(n * 0.15))

    X_train, y_train = X.iloc[:n_train],             y[:n_train]
    X_val,   y_val   = X.iloc[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X.iloc[n_train+n_val:],        y[n_train+n_val:]

    print(f"\n[{name}] rows: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"[{name}] original clusters: {list(le.classes_)}")

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    proba_test = model.predict_proba(X_test)
    pred_enc   = proba_test.argmax(axis=1)
    test_acc   = accuracy_score(y_test, pred_enc)
    test_ll    = log_loss(y_test, proba_test, labels=list(range(len(le.classes_))))
    print(f"[{name}] test accuracy: {test_acc:.4f}")
    print(f"[{name}] test logloss:  {test_ll:.5f}")

    pred_labels = le.inverse_transform(pred_enc)
    true_labels = le.inverse_transform(y_test)
    print(classification_report(true_labels, pred_labels, zero_division=0))

    report_dict = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    report_df   = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "label"})
    report_df.insert(0, "country", name)
    report_df["n_train"]      = len(X_train)
    report_df["n_val"]        = len(X_val)
    report_df["n_test"]       = len(X_test)
    report_df["test_accuracy"] = test_acc
    report_df["test_logloss"]  = test_ll
    report_df["classes"]       = [",".join(map(str, le.classes_))] * len(report_df)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path / f"xgb_eval_{name}.csv", index=False)

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
    return model, feature_cols, le, splits, report_df


def save_cluster_model_bundle(
    model,
    feature_cols: list[str],
    le,
    out_path: Path,
) -> None:
    """Serialize a Layer 1 model bundle (model + feature order + label encoder) to disk.

    Args:
        model: Trained XGBClassifier.
        feature_cols: Ordered list of feature column names used during training.
        le: Fitted LabelEncoder (maps cluster int → encoded 0-based class).
        out_path: Full path for the .joblib file.
    """
    bundle = {"model": model, "feature_cols": feature_cols, "label_encoder": le}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"  Saved Layer-1 bundle → {out_path}")


# ------------------------------------------------------------------
# Layer 2 — Per-cluster price regressor
# ------------------------------------------------------------------

def xgb_price_within_cluster(
    df: pd.DataFrame,
    model_name: str,
    model_dir: Path = Path("data/models/within_cluster"),
) -> tuple[float, float, Path, list[str]]:
    """Train one XGBoost price regressor for a single cluster and save the bundle.

    Uses a chronological 70/15/15 row-based split (same as the original notebook).

    Args:
        df: Hourly DataFrame for one cluster. Target column: price_eur_mwh.
            Feature columns: everything except price_eur_mwh and cluster.
        model_name: Filename stem (e.g. "xgb_price_DK_p20_c1") — .joblib is appended.
        model_dir: Directory where the .joblib bundle is saved.

    Returns:
        (rmse, wape_pct, saved_path, feature_cols)
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import root_mean_squared_error

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = df.reset_index(drop=True)
    y = df["price_eur_mwh"]
    X = df.drop(columns=["price_eur_mwh", "cluster"], errors="ignore")
    feature_cols = X.columns.tolist()

    n         = len(df)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    X_train, y_train = X.iloc[:train_end],          y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end],   y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:],            y.iloc[val_end:]

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred   = model.predict(X_test)
    rmse     = float(root_mean_squared_error(y_test, y_pred))
    y_true   = y_test.to_numpy(dtype=float)
    abs_sum  = float(np.abs(y_pred - y_true).sum())
    denom    = float(np.abs(y_true).sum())
    wape_pct = float(abs_sum / denom * 100) if denom != 0 else float("nan")

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": {"rmse": rmse, "wape_pct": wape_pct},
        "splits": {"train_end": train_end, "val_end": val_end, "n": n},
    }
    save_path = model_dir / f"{model_name}.joblib"
    joblib.dump(artifact, save_path)
    print(f"  Saved {model_name}  RMSE={rmse:.2f}  WAPE={wape_pct:.2f}%")
    return rmse, wape_pct, save_path, feature_cols


# ------------------------------------------------------------------
# Pipeline inference — cluster probabilities and weighted prediction
# ------------------------------------------------------------------

def predict_cluster_probabilities(
    bundle_path: Path,
    df_new: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Load a Layer 1 model bundle and return per-cluster probabilities for each day.

    Args:
        bundle_path: Path to the .joblib bundle (model + feature_cols + label_encoder).
        df_new: Daily feature DataFrame (no target column, must contain all feature_cols).
        date_col: Name of the date column to carry through to the output.

    Returns:
        DataFrame with columns [date_col, proba_cluster_1, proba_cluster_2, ...].
    """
    bundle = joblib.load(bundle_path)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    le           = bundle["label_encoder"]

    X_new = df_new[feature_cols].copy()
    proba = model.predict_proba(X_new)

    proba_cols = [f"proba_cluster_{c}" for c in le.classes_]
    out = pd.DataFrame(proba, columns=proba_cols, index=df_new.index)
    out.insert(0, date_col, df_new[date_col].values)
    return out


def weighted_price_prediction(
    test_df: pd.DataFrame,
    proba_df: pd.DataFrame,
    model_files: list[str],
    cluster_ids: list[int],
    models_dir: Path,
    date_col: str = "date",
    actual_col: str = "price_eur_mwh",
) -> pd.DataFrame:
    """Combine per-cluster price predictions weighted by cluster probabilities.

    Works when proba_df is daily and test_df is hourly — merges by day.

    Args:
        test_df: Hourly DataFrame with weather/precipitation features.
        proba_df: Daily DataFrame with proba_cluster_* columns from predict_cluster_probabilities.
        model_files: List of .joblib filenames for each cluster's regressor.
        cluster_ids: List of cluster integer IDs matching model_files order.
        models_dir: Directory containing the Layer 2 .joblib bundles.
        date_col: Name of the date column in both DataFrames.
        actual_col: Name of the actual price column in test_df (included in output if present).

    Returns:
        DataFrame with columns [date_col, price_eur_mwh (if present),
        pred_price_c*, weighted_pred_price].
    """
    df = test_df.copy()
    all_proba_cols = [f"proba_cluster_{c}" for c in cluster_ids]

    # Merge daily probabilities onto hourly rows by day
    df["_day"] = pd.to_datetime(df[date_col]).dt.normalize()
    p = proba_df.copy()
    p["_day"] = pd.to_datetime(p[date_col]).dt.normalize()
    df = df.merge(p[["_day"] + all_proba_cols], on="_day", how="left").drop(columns=["_day"])

    models_path = Path(models_dir)
    preds       = []
    pred_cols   = []
    modelled_cluster_ids = []
    for c, fname in zip(cluster_ids, model_files):
        bundle       = joblib.load(models_path / fname)
        model        = bundle["model"]
        feature_cols = bundle["feature_cols"]
        pred = model.predict(df[feature_cols])
        preds.append(pred)
        pred_cols.append(f"pred_price_c{c}")
        modelled_cluster_ids.append(c)

    # If some clusters were skipped (too few rows), redistribute their probability
    # mass proportionally across the clusters that do have models.
    modelled_proba_cols = [f"proba_cluster_{c}" for c in modelled_cluster_ids]
    probas_mat = df[modelled_proba_cols].to_numpy()
    row_sums   = probas_mat.sum(axis=1, keepdims=True)
    # Avoid division by zero (shouldn't happen, but guard anyway)
    row_sums   = np.where(row_sums == 0, 1.0, row_sums)
    probas_mat = probas_mat / row_sums   # renormalize to sum to 1

    preds_mat = np.column_stack(preds)
    weighted  = (preds_mat * probas_mat).sum(axis=1)

    out = pd.DataFrame({date_col: df[date_col].values})
    if actual_col in df.columns:
        out[actual_col] = df[actual_col].values
    for i, col in enumerate(pred_cols):
        out[col] = preds_mat[:, i]
    out["weighted_pred_price"] = weighted
    return out


# ------------------------------------------------------------------
# Naive baselines
# ------------------------------------------------------------------

def naive_approach_one(
    df: pd.DataFrame,
    dt_col: str = "date_time",
    price_col: str = "price_eur_mwh",
) -> pd.DataFrame:
    """Generate Naive-1 predictions: yesterday's price at the same hour.

    Builds a lookup table shifted forward by 1 day and merges it back. The first
    24 rows will have NaN predictions; callers should drop these.

    Args:
        df: Hourly DataFrame with datetime and price columns.
        dt_col: Name of the datetime column.
        price_col: Name of the actual price column.

    Returns:
        Copy of df with an additional naive_one_pred_price column.
    """
    d = df.copy()
    d[dt_col] = pd.to_datetime(d[dt_col], utc=True)

    lookup = d[[dt_col, price_col]].rename(columns={price_col: "naive_one_pred_price"})
    lookup[dt_col] = lookup[dt_col] + pd.Timedelta(days=1)
    d = d.merge(lookup, on=dt_col, how="left")
    return d


def train_naive_weather_model(
    df: pd.DataFrame,
    model_name: str,
    label: str,
    model_dir: Path = Path("data/models/naive_two"),
    target_col: str = "price_eur_mwh",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Train the Naive-2 baseline: a single XGBoost regressor on weather features only.

    No clustering; trained on all hourly rows (no cluster split). Uses a
    chronological 70/15/15 row-based split.

    Args:
        df: Hourly DataFrame with weather + precipitation_last_x_days features.
            Must NOT contain cluster or date columns (drop before calling).
        model_name: Filename stem for the saved bundle (e.g. "naive_two_dk").
        label: Display label for printed metrics (e.g. "DK").
        model_dir: Directory for the .joblib bundle.
        target_col: Name of the price target column.
        train_frac: Fraction of rows for training.
        val_frac: Fraction of rows for validation (early stopping).
        random_state: XGBoost random seed.

    Returns:
        (model, saved_path, pred_df_test)  where pred_df_test has columns
        [price_real, price_predicted] for the test portion.
    """
    from xgboost import XGBRegressor
    from src.evaluation import mae_smape_wape

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    d = df.reset_index(drop=True).copy()
    y = d[target_col]
    X = d.drop(columns=[target_col], errors="ignore")
    feature_cols = X.columns.tolist()

    n       = len(d)
    n_train = int(np.floor(n * train_frac))
    n_val   = int(np.floor(n * val_frac))

    X_train, y_train = X.iloc[:n_train],             y.iloc[:n_train]
    X_val,   y_val   = X.iloc[n_train:n_train+n_val], y.iloc[n_train:n_train+n_val]
    X_test,  y_test  = X.iloc[n_train+n_val:],        y.iloc[n_train+n_val:]

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred  = model.predict(X_test).astype(float)
    y_true  = y_test.to_numpy(dtype=float)
    mae, smape, wape = mae_smape_wape(y_true, y_pred)

    print(f"{label} → MAE: {mae:.2f} EUR/MWh | sMAPE: {smape:.2f}% | WAPE: {wape:.2f}%")
    print(f"{label} → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    pred_df_test = pd.DataFrame(
        {"price_real": y_true, "price_predicted": y_pred},
        index=y_test.index,
    )

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": {
            "mae": float(mae),
            "smape_pct": float(smape) if not np.isnan(smape) else np.nan,
            "wape_pct":  float(wape)  if not np.isnan(wape)  else np.nan,
        },
        "splits": {
            "n_rows": int(n),
            "train_end": int(n_train),
            "val_end": int(n_train + n_val),
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
        },
    }
    save_path = model_dir / f"{model_name}.joblib"
    joblib.dump(bundle, save_path)
    print(f"  Saved → {save_path}")
    return model, save_path, pred_df_test

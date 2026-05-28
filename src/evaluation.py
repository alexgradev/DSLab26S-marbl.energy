"""WAPE / MAE, naive baselines, and stratified breakdowns."""
from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from src.config import ZoneConfig, ModelConfig
import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# Public column-name conventions used across the evaluation module.
HOUR_COLS:   List[str] = [f"h{h:02d}"        for h in range(24)]
PRED_COLS:   List[str] = [f"pred_h{h:02d}"   for h in range(24)]
NAIVE1_COLS: List[str] = [f"naive1_h{h:02d}" for h in range(24)]
NAIVE2_COLS: List[str] = [f"naive2_h{h:02d}" for h in range(24)]


from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


def reconstruct_actual_regimes(
    df_predictions: pd.DataFrame,
    df_hourly: pd.DataFrame,
    fold_models: List[Dict],
    model_cfg: ModelConfig,
    zone_cfg: ZoneConfig,
) -> pd.DataFrame:
    """Attach ground-truth regime labels to df_predictions.

    Uses each fold's own spike threshold (stored in fold_models[i]
    ["spike_threshold"]) and model_cfg.leaf_a_neg_frac_threshold to
    label each test day as 0=Leaf A, 1=Leaf B, 2=Normal.

    Returns df_predictions with two added columns:
        actual_regime:    0/1/2
        actual_negative:  1 if Leaf A else 0
    """
    pc = zone_cfg.schema.price
    h = df_hourly.copy()
    h["_d"] = pd.to_datetime(h.index.normalize().date)

    records = []
    for fold_idx, fold_def in enumerate(model_cfg.fold_definitions):
        _, _, te_start, te_end = fold_def
        spike_thr = fold_models[fold_idx]["spike_threshold"]

        te_mask = (
            (pd.to_datetime(df_predictions["date"]) >= te_start) &
            (pd.to_datetime(df_predictions["date"]) <= te_end)
        )
        fold_dates = pd.to_datetime(df_predictions.loc[te_mask, "date"])
        test_date_set = set(fold_dates.dt.date)

        agg = (
            h[h["_d"].dt.date.isin(test_date_set)]
            .groupby("_d")[pc]
            .agg(neg_frac=lambda x: (x < 0).mean(), day_max="max")
        )
        agg.index = pd.to_datetime(agg.index)

        for date, row in agg.iterrows():
            label = (
                0 if row["neg_frac"] > model_cfg.leaf_a_neg_frac_threshold
                else 1 if row["day_max"] > spike_thr
                else 2
            )
            records.append({"date": date, "actual_regime": label})

    actual_df = pd.DataFrame(records)
    actual_df["date"] = pd.to_datetime(actual_df["date"])

    df_out = df_predictions.merge(actual_df, on="date", how="left")
    df_out["actual_negative"] = (df_out["actual_regime"] == 0).astype(int)
    return df_out


def confusion_matrix_l1(df_eval: pd.DataFrame) -> Dict[str, Any]:
    """Build the L1 (Leaf A gate) confusion matrix and metrics.

    Expects df_eval to have 'actual_negative' and 'p_A' columns
    (produced by reconstruct_actual_regimes).
    """
    y_true = df_eval["actual_negative"].values
    y_prob = df_eval["p_A"].values
    y_pred = (y_prob >= 0.50).astype(int)

    return {
        "cm":             confusion_matrix(y_true, y_pred),
        "display_labels": ["Not Negative", "Negative"],
        "roc_auc":        roc_auc_score(y_true, y_prob),
        "pr_auc":         average_precision_score(y_true, y_prob),
        "brier":          brier_score_loss(y_true, y_prob),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Not Negative", "Negative"]
        ),
    }


def confusion_matrix_l2(df_eval: pd.DataFrame) -> Dict[str, Any]:
    """Build the L2 (B vs Normal) confusion matrix, restricted to non-A days."""
    df_sub = df_eval[df_eval["actual_negative"] == 0].copy()
    y_true = df_sub["actual_regime"].values

    p_B  = df_sub["p_B"].values
    p_C0 = df_sub["p_C0"].values
    p_C1 = df_sub["p_C1"].values
    y_pred = np.where((p_B > p_C0) & (p_B > p_C1), 1, 2)

    return {
        "cm":             confusion_matrix(y_true, y_pred, labels=[1, 2]),
        "display_labels": ["Leaf B (Spike)", "Normal (C0/C1)"],
        "classification_report": classification_report(
            y_true, y_pred, labels=[1, 2],
            target_names=["Leaf B (Spike)", "Normal (C0/C1)"]
        ),
    }


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error, NaN-safe.

    Returns NaN if the denominator |actual|.sum() is zero (e.g. all-zero
    or all-negative slice where absolute values cancel).

    Args:
        actual: Ground-truth values, any shape.
        predicted: Forecast values, same shape as actual.

    Returns:
        WAPE in percent (0-100+).
    """
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    denom = np.abs(a).sum()
    return 100 * np.abs(a - p).sum() / denom if denom > 0 else np.nan


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error, NaN-safe.

    Args:
        actual: Ground-truth values.
        predicted: Forecast values, same shape.

    Returns:
        MAE in EUR/MWh.
    """
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    return np.abs(a - p).mean() if len(a) > 0 else np.nan
    


def attach_naive_baselines(
    df_predictions: pd.DataFrame,
    df_hourly: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> pd.DataFrame:
    """Append naive_1 (yesterday's curve) and naive_2 (7-day mean) columns.

    For each test date d, naive_1 looks up the 24h price vector of d-1;
    naive_2 averages the curves of d-1 .. d-7 (where available).

    Args:
        df_predictions: Walk-forward prediction DataFrame.
        df_hourly: Full hourly DataFrame for actual price lookup.
        zone_cfg: Provides the price column name.

    Returns:
        df_predictions with 48 new columns: naive1_h00..h23, naive2_h00..h23.
    """
    ...


def overall_metrics(df_predictions: pd.DataFrame) -> Dict[str, float]:
    """Compute overall WAPE and MAE for model, naive_1, and naive_2.

    Args:
        df_predictions: Predictions with naive baselines attached.

    Returns:
        Flat dict with keys like "model_wape", "model_mae", "naive1_wape",
        "naive2_wape", etc.
    """
    ...


def regime_metrics(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-regime WAPE/MAE breakdown using dominant-regime assignment.

    Dominant regime is argmax over (p_A, p_B, p_C0, p_C1). WAPE is
    suppressed for Leaf A (denominators near zero); MAE is reported for
    all regimes.

    Args:
        df_predictions: Predictions with naive baselines attached.

    Returns:
        DataFrame indexed by regime label with columns: regime_name, n,
        share_pct, wape, mae, naive1_wape, naive2_wape.
    """
    prob_cols = ["p_A", "p_B", "p_C0", "p_C1"]
    regime_names = {0: "Leaf A (negative)", 1: "Leaf B (spike)", 2: "C0 (normal)", 3: "C1 (normal)"}
    
    # Supress SettingWithCopyWarning if df is a slice
    df_predictions = df_predictions.copy() 
    df_predictions["dominant_regime"] = df_predictions[prob_cols].values.argmax(axis=1)

    # print("=== Regime-Level Metrics ===")
    # print(f"{'Regime':<20} {'N':>5} {'Share%':>8} {'WAPE%':>10} {'MAE':>8} {'N1 WAPE%':>10} {'N2 WAPE%':>10}")
    # print("-" * 80)

    # Initialize a list to hold the row data
    results = []

    for label in range(4):
        mask = df_predictions["dominant_regime"] == label
        grp  = df_predictions[mask]
        
        # Handle empty regimes
        if len(grp) == 0:
            print(f"{regime_names[label]:<20} {'(empty)':>5}")
            results.append({
                "regime_name": regime_names[label],
                "n": 0,
                "share_pct": 0.0,
                "wape": np.nan,
                "mae": np.nan,
                "naive1_wape": np.nan,
                "naive2_wape": np.nan
            })
            continue

        # Extract and cast arrays to float to prevent type errors
        a  = grp[HOUR_COLS].values.ravel().astype(float)
        p  = grp[PRED_COLS].values.ravel().astype(float)
        n1 = grp[NAIVE1_COLS].values.ravel().astype(float)
        n2 = grp[NAIVE2_COLS].values.ravel().astype(float)
        
        n_count = len(grp)
        share = 100 * n_count / len(df_predictions)
        mae_v = mae(a, p)
        
        if label == 0:
            # print(f"{regime_names[label]:<20} {n_count:>5} {share:>8.1f} {'(MAE only)':>10} {mae_v:>8.2f} {'--':>10} {'--':>10}")
            print("  Note: WAPE suppressed for Regime A -- near-zero/negative price denominators.")
            
            # WAPE values recorded as np.nan for Leaf A
            results.append({
                "regime_name": regime_names[label],
                "n": n_count,
                "share_pct": share,
                "wape": np.nan,
                "mae": mae_v,
                "naive1_wape": np.nan,
                "naive2_wape": np.nan
            })
        else:
            w_p = wape(a, p)
            w_n1 = wape(a, n1)
            w_n2 = wape(a, n2)
            
            # print(f"{regime_names[label]:<20} {n_count:>5} {share:>8.1f} {w_p:>10.2f} {mae_v:>8.2f} {w_n1:>10.2f} {w_n2:>10.2f}")
            
            # Record standard regime results
            results.append({
                "regime_name": regime_names[label],
                "n": n_count,
                "share_pct": share,
                "wape": w_p,
                "mae": mae_v,
                "naive1_wape": w_n1,
                "naive2_wape": w_n2
            })

    # Construct the final DataFrame
    df_metrics = pd.DataFrame(results)
    
    # Set the index to be the regime label (0, 1, 2, 3) as requested in docstring
    df_metrics.index = pd.Index(range(4), name="regime_label")

    return df_metrics

def hourly_metrics(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-hour-of-day MAE and mean signed error.

    Args:
        df_predictions: Predictions with hour columns.

    Returns:
        DataFrame indexed by hour 0..23 with columns: mae, mean_signed_error.
    """
    


def fold_metrics(
    df_predictions: pd.DataFrame,
    model_cfg: ModelConfig,
    zone_cfg: ZoneConfig,
) -> pd.DataFrame:
    """Per-fold WAPE breakdown for model vs naive baselines.

    Tags each prediction with its fold number based on fold_definitions
    and computes WAPE per fold. Detects regression where model WAPE
    exceeds both naive baselines.

    Args:
        df_predictions: Predictions with naive baselines attached.
        model_cfg: Provides fold_definitions for tagging.

    Returns:
        DataFrame indexed by fold number with WAPE columns and a
        boolean 'model_worse_than_naives' flag.
    """
    fold_tags = np.zeros(len(df_predictions), dtype=int)
    for i, (_, _, te_start, te_end) in enumerate(model_cfg.fold_definitions):
        mask = (
            (pd.to_datetime(df_predictions["date"]) >= te_start) &
            (pd.to_datetime(df_predictions["date"]) <= te_end)
        )
        fold_tags[mask] = i + 1
    df_predictions["fold"] = fold_tags

    print("=== Fold-Level Metrics ===")
    print(f"{'Fold':<6} {'N':>5} {'Test period':<25} {'WAPE% model':>12} {'WAPE% N1':>10} {'WAPE% N2':>10}")
    print("-" * 75)

    for i, (_, _, te_start, te_end) in enumerate(model_cfg.fold_definitions):
        grp = df_predictions[df_predictions["fold"] == i + 1]
        if len(grp) == 0:
            continue
        a   = grp[HOUR_COLS].values.ravel()
        p   = grp[PRED_COLS].values.ravel()
        n1  = grp[NAIVE1_COLS].values.ravel()
        n2  = grp[NAIVE2_COLS].values.ravel()
        wm  = wape(a, p)
        wn1 = wape(a, n1)
        wn2 = wape(a, n2)
        period = f"{te_start} -> {te_end}"
        print(f"{i+1:<6} {len(grp):>5} {period:<25} {wm:>12.2f} {wn1:>10.2f} {wn2:>10.2f}")
        if wm > wn1 and wm > wn2:
            print(f"  WARNING: fold {i+1} model WAPE exceeds both naive baselines.")

    # Save final predictions with naive baselines appended.
    df_predictions.to_csv(f"results/walk_forward_predictions_final_{zone_cfg.zone}.csv", index=False)
    size_kb = os.path.getsize(f"results/walk_forward_predictions_final_{zone_cfg.zone}.csv") / 1024
    print(f"\nSaved: results/walk_forward_predictions_final_{zone_cfg.zone}.csv  ({size_kb:.1f} KB)")
    print(f"Columns: {df_predictions.shape[1]}  |  Rows: {len(df_predictions)}")



def worst_best_days_wape(
    df_predictions: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> Dict[str, pd.DataFrame]:
    """Return the top-n worst and best predicted days by daily WAPE or MAE.

    Adds daily_wape, daily_mae, daily_mape, actual_mean, pred_mean columns
    to the input (in-place is fine; doc the side effect).

    Args:
        df_predictions: Predictions from walk_forward.
        by: Sort key — 'wape', 'mae', or 'mape'.
        n: Number of days per side.
        zone_cfg: provides the bidding zone name for the title

    Returns:
        Dict with keys 'worst' and 'best', each a DataFrame with
        date, regime_name, daily_mae, daily_mape, daily_wape, actual_mean,
        pred_mean.
    """

        # Compute WAPE per day, then smooth with a 30-day rolling window.
    daily_wape_vals = []
    for _, row in df_predictions.iterrows():
        a = row[HOUR_COLS].values.astype(float)
        p = row[PRED_COLS].values.astype(float)
        daily_wape_vals.append(wape(a, p))
    df_predictions["daily_wape"] = daily_wape_vals

    # MAPE: mean over hours of |actual - predicted| / |actual|, skipping hours where actual == 0.
    daily_mape_vals = []
    for _, row in df_predictions.iterrows():
        a = row[HOUR_COLS].values.astype(float)
        p = row[PRED_COLS].values.astype(float)
        mask = ~(np.isnan(a) | np.isnan(p)) & (np.abs(a) > 0)
        if mask.sum() > 0:
            daily_mape_vals.append(100 * np.mean(np.abs(a[mask] - p[mask]) / np.abs(a[mask])))
        else:
            daily_mape_vals.append(np.nan)
    df_predictions["daily_mape"] = daily_mape_vals

    cols_show = ["date", "dominant_regime", "daily_mape", "daily_wape", "actual_mean", "pred_mean"]
    worst = df_predictions.nlargest(10, "daily_wape")[cols_show].copy()
    best  = df_predictions.nsmallest(10, "daily_wape")[cols_show].copy()
    
    regime_names = {0: "Leaf A (negative)", 1: "Leaf B (spike)", 2: "C0 (normal)", 3: "C1 (normal)"}

    worst["regime_name"] = worst["dominant_regime"].map(regime_names)
    best["regime_name"]  = best["dominant_regime"].map(regime_names)

    display_cols = ["date", "regime_name", "daily_mape", "daily_wape", "actual_mean", "pred_mean"]
    rename_map   = {"daily_mape": "MAPE%", "daily_wape": "WAPE%",
                    "actual_mean": "actual_mean_eur", "pred_mean": "pred_mean_eur"}

    print(f"=== 10 worst predicted days for {zone_cfg.zone} (by WAPE) ===")
    print(worst[display_cols].rename(columns=rename_map).round(2).to_string(index=False))

    print(f"=== 10 best predicted days for {zone_cfg.zone} (by WAPE) ===")
    print(best[display_cols].rename(columns=rename_map).round(2).to_string(index=False))

    print(f"=== Worst-day regime breakdown for {zone_cfg.zone} ===")
    print(worst["regime_name"].value_counts().to_string())



def worst_best_days_mae(
    df_predictions: pd.DataFrame,
    zone_cfg: ZoneConfig,
) -> Dict[str, pd.DataFrame]:
    """Return the top-n worst and best predicted days by daily WAPE or MAE.

    Adds daily_wape, daily_mae, daily_mape, actual_mean, pred_mean columns
    to the input (in-place is fine; doc the side effect).

    Args:
        df_predictions: Predictions from walk_forward.
        by: Sort key — 'wape', 'mae', or 'mape'.
        n: Number of days per side.
        zone_cfg: provides the bidding zone name for the title

    Returns:
        Dict with keys 'worst' and 'best', each a DataFrame with
        date, regime_name, daily_mae, daily_mape, daily_wape, actual_mean,
        pred_mean.
    """

    # MAE per day: mean absolute error across all 24 hours.
    daily_mae_vals = []
    for _, row in df_predictions.iterrows():
        a = row[HOUR_COLS].values.astype(float)
        p = row[PRED_COLS].values.astype(float)
        mask = ~(np.isnan(a) | np.isnan(p))
        daily_mae_vals.append(np.abs(a[mask] - p[mask]).mean() if mask.sum() > 0 else np.nan)
    df_predictions["daily_mae"] = daily_mae_vals

    cols_show = ["date", "dominant_regime", "daily_mae", "daily_mape", "daily_wape", "actual_mean", "pred_mean"]
    worst_mae = df_predictions.nlargest(10, "daily_mae")[cols_show].copy()
    best_mae  = df_predictions.nsmallest(10, "daily_mae")[cols_show].copy()

    regime_names = {0: "Leaf A (negative)", 1: "Leaf B (spike)", 2: "C0 (normal)", 3: "C1 (normal)"}


    worst_mae["regime_name"] = worst_mae["dominant_regime"].map(regime_names)
    best_mae["regime_name"]  = best_mae["dominant_regime"].map(regime_names)

    display_cols = ["date", "regime_name", "daily_mae", "daily_mape", "daily_wape", "actual_mean", "pred_mean"]
    rename_map   = {"daily_mae": "MAE", "daily_mape": "MAPE%", "daily_wape": "WAPE%",
                    "actual_mean": "actual_mean_eur", "pred_mean": "pred_mean_eur"}

    print(f"=== 10 worst predicted days for {zone_cfg.zone} (by MAE) ===")
    print(worst_mae[display_cols].rename(columns=rename_map).round(2).to_string(index=False))

    print("=== 10 best predicted days for {zone_cfg.zone} (by MAE) ===")
    print(best_mae[display_cols].rename(columns=rename_map).round(2).to_string(index=False))

    print("=== Worst-day regime breakdown for {zone_cfg.zone} ===")
    print(worst_mae["regime_name"].value_counts().to_string())
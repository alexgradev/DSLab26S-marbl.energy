"""
src/evaluation.py — Metrics and result-comparison helpers.

All metric functions return floats in human-readable units:
  - MAE  : EUR/MWh
  - sMAPE: percent (e.g. 31.4, not 0.314)
  - WAPE : percent
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "mae_smape_wape",
    "avg_errors",
    "build_results_table",
    "plot_hourly_series",
]


def mae_smape_wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """Compute MAE, sMAPE (%), and WAPE (%) from raw numpy arrays.

    Args:
        y_true: Actual price values.
        y_pred: Predicted price values (same length).

    Returns:
        (mae, smape_pct, wape_pct)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    abs_err = np.abs(y_pred - y_true)

    # MAE
    mae = float(abs_err.mean())

    # sMAPE — symmetric; guarded against zero denominator
    denom_smape = np.abs(y_true) + np.abs(y_pred)
    mask = denom_smape != 0
    smape = float((2 * abs_err[mask] / denom_smape[mask]).mean() * 100) if mask.any() else float("nan")

    # WAPE — volume-weighted; robust to near-zero prices
    denom_wape = float(np.abs(y_true).sum())
    wape = float(abs_err.sum() / denom_wape * 100) if denom_wape != 0 else float("nan")

    return mae, smape, wape


def avg_errors(
    df: pd.DataFrame,
    pred_col: str = "weighted_pred_price",
    actual_col: str = "price_eur_mwh",
) -> tuple[float, float, float]:
    """Compute MAE, sMAPE (%), and WAPE (%) from a prediction DataFrame.

    Drops rows where either column is NaN before computing metrics.

    Args:
        df: DataFrame containing actual and predicted price columns.
        pred_col: Name of the predicted price column.
        actual_col: Name of the actual price column.

    Returns:
        (mae, smape_pct, wape_pct)
    """
    x = df[[actual_col, pred_col]].dropna()
    return mae_smape_wape(x[actual_col].to_numpy(), x[pred_col].to_numpy())


def build_results_table(results_dict: dict) -> pd.DataFrame:
    """Assemble per-zone per-approach metrics into a single display DataFrame.

    Args:
        results_dict: Nested dict of the form
            {zone: {approach_label: (mae, smape, wape), ...}, ...}

    Returns:
        DataFrame with columns [Zone, Approach, MAE, sMAPE%, WAPE%].
    """
    rows = []
    for zone, approaches in results_dict.items():
        for approach, (mae, smape, wape) in approaches.items():
            rows.append(
                {
                    "Zone": zone,
                    "Approach": approach,
                    "MAE (EUR/MWh)": round(mae, 10),
                    "sMAPE%": round(smape, 10),
                    "WAPE%": round(wape, 10),
                }
            )
    return pd.DataFrame(rows)


def plot_hourly_series(
    df: pd.DataFrame,
    title: str,
    pred_col: str = "weighted_pred_price",
    actual_col: str = "price_eur_mwh",
    n_hours: int = 48,
) -> None:
    """Plot actual vs predicted hourly prices for the first n_hours rows.

    Uses the integer row index on the x-axis so every hour is visible regardless
    of timezone or DST gaps in the datetime index.

    Args:
        df: DataFrame with actual and predicted price columns.
        title: Figure title.
        pred_col: Predicted price column name.
        actual_col: Actual price column name.
        n_hours: How many rows from the start to plot.
    """
    import matplotlib.pyplot as plt  # local import — avoids forcing matplotlib on import

    d = df.reset_index(drop=True).iloc[:n_hours]

    plt.figure(figsize=(12, 5))
    plt.plot(d[pred_col].to_numpy(), label="Predicted")
    plt.plot(d[actual_col].to_numpy(), label="Actual")
    plt.title(title)
    plt.xlabel("Hour index (row number)")
    plt.ylabel("Price (EUR/MWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()

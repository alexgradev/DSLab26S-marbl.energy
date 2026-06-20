"""Regime construction, classifiers (L1+L2), expert regressors, and the per-fold + walk-forward drivers."""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from src.config import ZoneConfig, ModelConfig
import skfuzzy as fuzz
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
import pickle
from pathlib import Path



# Single source of truth for PCA-excluded columns. Both run_fold and
# extract_pca_loadings import this; the cell-40 duplication is eliminated.
PCA_EXCLUDE_COLS: frozenset[str] = frozenset({
    "price_neg_frac_lag1", "was_negative_lag1", "neg_streak_length",
    "price_max_lag1", "spike_streak_length",
})


def compute_spike_threshold(
    df_hourly: pd.DataFrame,
    tr_start: str,
    tr_end: str,
    zone_cfg: ZoneConfig,
    model_cfg: ModelConfig,
) -> float:
    """Compute the Leaf B spike threshold from training-window hours only.

    Uses model_cfg.spike_pct on the percentile of training prices, where
    the training window is [tr_start, tr_end+23:00 UTC].

    Args:
        df_hourly: Full hourly DataFrame.
        tr_start: Training-window start date (ISO).
        tr_end: Training-window end date (ISO).
        zone_cfg: Provides the price column name.
        model_cfg: Provides the percentile.

    Returns:
        Spike threshold in EUR/MWh.
    """

    h_mask = (
        (df_hourly.index >= pd.Timestamp(tr_start, tz="UTC")) &
        (df_hourly.index <= pd.Timestamp(tr_end, tz="UTC") + pd.Timedelta(hours=23))
    )
    spike_threshold = float(np.percentile(
        df_hourly.loc[h_mask, zone_cfg.schema.price].dropna(), model_cfg.spike_pct
    ))
    print(f"  Spike threshold (fold): {spike_threshold:.2f} EUR/MWh")
    return spike_threshold
    


def build_leaf_masks(
    df_train: pd.DataFrame,
    df_hourly: pd.DataFrame,
    spike_threshold: float,
    zone_cfg: ZoneConfig,
    model_cfg: ModelConfig,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Derive Leaf A / Leaf B / Normal masks from actual same-day prices.

    Labels for day T reflect day T's own price behavior (negative-frac and
    daily max from df_hourly), not the lag-1 feature columns. This avoids
    label leakage from the feature matrix into the cluster definitions.

    Args:
        df_train: Daily feature matrix sliced to the training window.
        df_hourly: Full hourly DataFrame, used for ground-truth daily aggregates.
        spike_threshold: Fold-specific Leaf B threshold.
        zone_cfg: Provides the price column name.
        model_cfg: Provides leaf_a_neg_frac_threshold.

    Returns:
        Three boolean Series aligned to df_train.index: (mask_A, mask_B,
        mask_normal). Exactly one is True per row.
    """
    h_copy = df_hourly.copy()
    h_copy["_d"] = pd.to_datetime(h_copy.index.normalize().date)
    train_date_set = set(df_train.index.date)
    actual_agg = (
        h_copy[h_copy["_d"].dt.date.isin(train_date_set)]
        .groupby("_d")[zone_cfg.schema.price]
        .agg(neg_frac=lambda x: (x < 0).mean(), day_max="max")
    )
    
    actual_agg.index = pd.to_datetime(actual_agg.index)
    actual_agg = actual_agg.reindex(df_train.index)

    mask_A      = actual_agg["neg_frac"] > model_cfg.leaf_a_neg_frac_threshold
    mask_B      = (~mask_A) & (actual_agg["day_max"] > spike_threshold)
    mask_normal = ~mask_A & ~mask_B

    n_total = len(df_train)
    use_ridge_regime = set()
    for label, mask, name in [(0, mask_A, "Leaf A"), (1, mask_B, "Leaf B"), (2, mask_normal, "Normal")]:
        n = int(mask.sum())
        line = f"  {name}: {n} days ({100*n/n_total:.1f}%)"
        if n < model_cfg.min_cluster_size:
            line += " -- WARNING: small bucket, Ridge fallback for expert models"
            use_ridge_regime.update({label, label + 1} if label == 2 else {label})
        print(line)
    return mask_A, mask_B, mask_normal, use_ridge_regime


def fit_pca_and_fcm(
    df_train: pd.DataFrame,
    price_pivot_train: pd.DataFrame,
    mask_normal: pd.Series,
    model_cfg: ModelConfig,
) -> Tuple[StandardScaler, PCA, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Fit StandardScaler + PCA + Fuzzy-C-Means on the normal-day subset.

    Concatenates the daily feature matrix (excluding PCA_EXCLUDE_COLS) with
    the lagged price pivot, scales, runs PCA to model_cfg.pca_variance_target,
    then fits FCM with model_cfg.fkm_k_normal clusters.

    Args:
        df_train: Daily feature matrix for the training window.
        price_pivot_train: 24h pivot aligned to df_train.index.
        mask_normal: Boolean Series marking normal-day rows in df_train.
        model_cfg: PCA and FCM hyperparameters + random_seed.

    Returns:
        scaler: Fitted StandardScaler.
        pca: Fitted PCA.
        cntr: FCM cluster centers, shape (k, n_components).
        hard_labels: Hard FCM labels for valid normal-day rows, shape (n,).
        normal_idx_valid: DatetimeIndex of normal-day rows that survived
            NaN filtering (excludes the first row when pivot is NaN).
    """
    # PCA feature matrix for normal training days ----
    pca_exclude = {
        "price_neg_frac_lag1", "was_negative_lag1", "neg_streak_length",
        "price_max_lag1", "spike_streak_length",
    }
    pca_cols = [c for c in df_train.columns if c not in pca_exclude]

    normal_idx   = df_train.index[mask_normal.values]
    X_daily_norm = df_train.loc[normal_idx, pca_cols].values
    X_pivot_norm = price_pivot_train.reindex(normal_idx).values

    # First row of pivot may be NaN (lag-1 shift at start of training window).
    valid = (~np.isnan(X_pivot_norm).any(axis=1) & ~np.isnan(X_daily_norm).any(axis=1))
    X_daily_norm    = X_daily_norm[valid]
    X_pivot_norm    = X_pivot_norm[valid]
    normal_idx_valid = normal_idx[valid]

    X_combined = np.hstack([X_daily_norm, X_pivot_norm])
    scaler_pca = StandardScaler()
    X_scaled   = scaler_pca.fit_transform(X_combined)

    pca_probe = PCA(random_state=model_cfg.random_seed)
    pca_probe.fit(X_scaled)
    cumvar  = np.cumsum(pca_probe.explained_variance_ratio_)
    n_comp  = int(np.searchsorted(cumvar, model_cfg.pca_variance_target)) + 1
    n_comp  = min(n_comp, X_scaled.shape[1], X_scaled.shape[0] - 1)
    print(f"  PCA: {n_comp} components retain {cumvar[n_comp-1]:.3f} cumulative variance")

    pca_fit = PCA(n_components=n_comp, random_state=model_cfg.random_seed)
    X_pca   = pca_fit.fit_transform(X_scaled)


    # FKM on normal training days ----
    np.random.seed(model_cfg.random_seed)
    cntr, u, _, _, _, _, _ = fuzz.cmeans(
        X_pca.T, c=model_cfg.fkm_k_normal, m=model_cfg.fkm_m,
        error=model_cfg.fkm_error, maxiter=model_cfg.fkm_maxiter, seed=model_cfg.random_seed
    )
    hard_labels_normal = np.argmax(u, axis=0)

    for k in range(model_cfg.fkm_k_normal):
        n_k = int((hard_labels_normal == k).sum())
        warn = " -- WARNING: small cluster" if n_k < model_cfg.min_cluster_size else ""
        print(f"  FKM C{k}: {n_k} days{warn}")

    return scaler_pca, pca_fit, cntr, hard_labels_normal, normal_idx_valid

def assemble_regime_labels(
    df_train: pd.DataFrame,
    mask_A: pd.Series,
    mask_B: pd.Series,
    mask_normal: pd.Series,
    hard_labels_normal: np.ndarray,
    normal_idx_valid: pd.DatetimeIndex,
) -> pd.Series:
    """Combine leaf masks and FCM labels into a single regime label series.

    Encoding: 0=Leaf A, 1=Leaf B, 2=C0 (FCM cluster 0), 3=C1 (FCM cluster 1).
    Normal days excluded from PCA (first-row pivot NaN) are assigned to C0
    by convention, matching the original notebook.

    Args:
        df_train: Daily feature matrix.
        mask_A: Leaf A boolean mask.
        mask_B: Leaf B boolean mask.
        mask_normal: Normal-day boolean mask.
        hard_labels_normal: FCM hard labels for valid normal-day rows.
        normal_idx_valid: Index of valid normal-day rows.

    Returns:
        Integer Series of length len(df_train), one label per row.
    """
     # ---- STEP 7: Assemble regime labels for all training days ----
    regime_labels = pd.Series(np.nan, index=df_train.index, dtype=float)
    regime_labels.loc[df_train.index[mask_A.values]]  = 0
    regime_labels.loc[df_train.index[mask_B.values]]  = 1
    for pos, d in enumerate(normal_idx_valid):
        regime_labels.loc[d] = 2 + int(hard_labels_normal[pos])
    # Normal days excluded from PCA (first-row NaN in pivot) → assign to C0.
    still_nan = regime_labels.isna() & mask_normal.values
    regime_labels.loc[regime_labels.index[still_nan]] = 2
    assert regime_labels.isna().sum() == 0, "Unassigned regime labels remain."
    regime_labels = regime_labels.astype(int)
    return regime_labels


def train_classifier_l2(
    X_clf: np.ndarray,
    y_clf: np.ndarray,
    model_cfg: ModelConfig,
) -> XGBClassifier:
    """Fit the multinomial XGBoost gate over non-Leaf-A days (B vs C0 vs C1).

    Filters to y_clf != 0, shifts labels (B→0, C0→1, C1→2), and trains
    XGBClassifier with objective="multi:softprob", num_class=3.

    Args:
        X_clf: Classifier feature matrix (daily features ⊕ pivot hours).
        y_clf: 4-class regime labels (0/1/2/3).
        model_cfg: clf_params and random_seed.

    Returns:
        Fitted XGBClassifier.
    """
    non_negative_mask = (y_clf != 0)
    X_clf_sub = X_clf[non_negative_mask]
    y_clf_sub = y_clf[non_negative_mask] - 1  # B→0, C0→1, C1→2

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=model_cfg.random_seed,
        verbosity=0,
        **model_cfg.clf_params,
    )
    clf.fit(X_clf_sub, y_clf_sub)
    print(f"  L2 Classifier training accuracy: "
          f"{(clf.predict(X_clf_sub) == y_clf_sub).mean():.3f}")
    return clf

def train_classifier_l1(
    X_clf: np.ndarray,
    y_clf: np.ndarray,
    model_cfg: ModelConfig,
) -> Tuple[LogisticRegressionCV, StandardScaler]:
    """Fit the binary LogisticRegressionCV gate (Leaf A vs not-A).

    Standardizes inputs and trains LogisticRegressionCV with saga solver,
    class_weight='balanced', l1_ratios=(0,), use_legacy_attributes=False.

    Args:
        X_clf: Classifier feature matrix.
        y_clf: 4-class regime labels (used to derive binary target).
        model_cfg: l1_C_grid, l1_cv_folds, random_seed.

    Returns:
        clf_l1: Fitted LogisticRegressionCV.
        scaler_l1: Fitted StandardScaler (must be applied to test inputs).
    """
    # Level 1 Classifier (Negative Price Router) ----
    # Dedicated binary model answering: "What is the probability of a negative price day?"
    # Trained independently from the 3-class Level 2 classifier. 
    # Uses LogisticRegressionCV to handle the balanced target across all days.

    y_neg = (y_clf == 0).astype(int)

    scaler_l1 = StandardScaler()
    X_clf_scaled_l1 = scaler_l1.fit_transform(X_clf)

    clf_l1 = LogisticRegressionCV(
        Cs=list(model_cfg.l1_C_grid),
        cv=model_cfg.l1_cv_folds,
        scoring="balanced_accuracy",
        class_weight="balanced",
        solver="saga",
        max_iter=5000,
        random_state=model_cfg.random_seed,
        l1_ratios=(0,),
        use_legacy_attributes=False,
    )
    clf_l1.fit(X_clf_scaled_l1, y_neg)

    l1_probs = clf_l1.predict_proba(X_clf_scaled_l1)[:, 1]
    print(f"  L1 Classifier: best C={float(clf_l1.C_):.4f}  "
          f"train_AUC={roc_auc_score(y_neg, l1_probs):.3f}")

    return clf_l1, scaler_l1
    


def train_expert_models(
    df_train_clf: pd.DataFrame,
    clf_cols: List[str],
    y_clf: np.ndarray,
    price_mat: pd.DataFrame,
    use_ridge_regime: Set[int],
    model_cfg: ModelConfig,
) -> Dict[int, Dict[int, Any]]:
    """Train the 96 hour-specific expert regressors (4 regimes × 24 hours).

    For each (regime, hour): selects training rows of that regime, drops
    NaN target hours, fits an XGBRegressor - or a RidgeCV pipeline if the
    regime is flagged as small (in use_ridge_regime).

    Args:
        df_train_clf: Training feature matrix (daily ⊕ pivot, NaN-dropped).
        clf_cols: Feature column ordering used by the classifiers.
        y_clf: 4-class regime labels aligned to df_train_clf.index.
        price_mat: (date × hour) pivot of actual hourly prices, used as
            the regression target.
        use_ridge_regime: Set of regime labels (0-3) too small for XGB.
        model_cfg: reg_params, ridge_alpha_grid, random_seed.

    Returns:
        Nested dict expert_models[regime][hour] → fitted estimator.
    """
    # ---- STEP 9: 96 expert models (4 regimes × 24 hours) ----
    # Build a (date × hour) price matrix for target extraction.
    expert_models: Dict[int, Dict[int, Any]] = {r: {} for r in range(4)}

    for regime in range(4):
        r_dates = df_train_clf.index[y_clf == regime]
        X_reg = df_train_clf.loc[r_dates, clf_cols].values
        Y_reg = price_mat.reindex(r_dates).values

        for hour in range(24):
            y_h = Y_reg[:, hour]
            valid = ~np.isnan(y_h)
            if regime in use_ridge_regime:
                mdl = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=list(model_cfg.ridge_alpha_grid))),
                ])
            else:
                mdl = XGBRegressor(
                    random_state=model_cfg.random_seed,
                    verbosity=0,
                    **model_cfg.reg_params,
                )
            mdl.fit(X_reg[valid], y_h[valid])
            expert_models[regime][hour] = mdl

    print("  Expert models trained: 96")
    return expert_models

def predict_fold(
    df_test_clf: pd.DataFrame,
    clf_cols: List[str],
    classifier_l1: LogisticRegressionCV,
    scaler_l1: StandardScaler,
    classifier_l2: XGBClassifier,
    expert_models: Dict[int, Dict[int, Any]],
    price_mat: pd.DataFrame,
) -> pd.DataFrame:
    """Generate 24h forecasts for the test window via hierarchical soft blend.

    Joint probabilities follow the cascade:
        p_A     = clf_l1.predict_proba(...)[:, 1]
        p_B|¬A, p_C0|¬A, p_C1|¬A = clf_l2.predict_proba(...)
        p_B  = (1-p_A) * p_B|¬A
        p_C0 = (1-p_A) * p_C0|¬A
        p_C1 = (1-p_A) * p_C1|¬A
    Final forecast is the probability-weighted average across the four
    regime expert curves.

    Args:
        df_test_clf: Test feature matrix (daily ⊕ pivot, NaN-dropped).
        clf_cols: Feature column ordering.
        classifier_l1: Fitted Leaf A gate.
        scaler_l1: Fitted scaler matched to classifier_l1.
        classifier_l2: Fitted 3-class gate.
        expert_models: Nested expert dict from train_expert_models.
        price_mat: (date × hour) pivot of actual hourly prices, used to
            attach ground-truth columns h00..h23.

    Returns:
        DataFrame with columns: date, h00..h23 (actuals), pred_h00..pred_h23,
        p_A, p_B, p_C0, p_C1, effective_regime.
    """
    X_test_mat = df_test_clf[clf_cols].values
    X_test_scaled_l1 = scaler_l1.transform(X_test_mat)

    l1_probs = classifier_l1.predict_proba(X_test_scaled_l1)[:, 1]
    regime_probs = classifier_l2.predict_proba(X_test_mat)

    results_rows = []
    for i, date in enumerate(df_test_clf.index):
        x_row = X_test_mat[i : i + 1]

        regime_preds = np.array([
            [expert_models[r][h].predict(x_row)[0] for h in range(24)]
            for r in range(4)
        ])

        p_A = l1_probs[i]
        p_not_A = 1.0 - p_A
        probs_sub = regime_probs[i]
        p_B  = p_not_A * probs_sub[0]
        p_C0 = p_not_A * probs_sub[1]
        p_C1 = p_not_A * probs_sub[2]

        y_hat = (p_A * regime_preds[0]
                 + p_B * regime_preds[1]
                 + p_C0 * regime_preds[2]
                 + p_C1 * regime_preds[3])
        effective_regime = int(np.argmax([p_A, p_B, p_C0, p_C1]))

        act = (price_mat.loc[date].values
               if date in price_mat.index else np.full(24, np.nan))

        row = {"date": date}
        for h in range(24):
            row[f"h{h:02d}"] = act[h]
            row[f"pred_h{h:02d}"] = y_hat[h]
        row.update(p_A=p_A, p_B=p_B, p_C0=p_C0, p_C1=p_C1,
                   effective_regime=effective_regime)
        results_rows.append(row)

    return pd.DataFrame(results_rows)


def run_fold(
    df_daily_full: pd.DataFrame,
    price_pivot_full: pd.DataFrame,
    df_hourly_full: pd.DataFrame,
    fold_def: Tuple[str, str, str, str],
    zone_cfg: ZoneConfig,
    model_cfg: ModelConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute the entire per-fold pipeline end-to-end.

    Steps:
      1. Slice training / test windows from the full daily matrix.
      2. Compute fold-specific spike threshold from training hours only.
      3. Rebuild features with the fold's spike_threshold so
         spike_streak_length is in-distribution for the test window.
      4. Build leaf masks from actual same-day prices.
      5. Fit PCA + FCM on normal training days.
      6. Assemble regime labels for all training days.
      7. Train L1 (binary Leaf A) and L2 (multinomial B/C0/C1) classifiers.
      8. Train 96 expert models (RidgeCV for under-min_cluster_size regimes).
      9. Generate test predictions via predict_fold.

    No global state is read or written.

    Args:
        df_daily_full: Daily feature matrix (full date range).
        price_pivot_full: 24h lagged pivot (full date range).
        df_hourly_full: Full hourly DataFrame (needed for ground-truth
            labels and rebuilding features with the fold spike threshold).
        fold_def: (tr_start, tr_end, te_start, te_end) ISO date strings.
        zone_cfg: Used by build_daily_features and price-column dispatch.
        model_cfg: All hyperparameters and random_seed.

    Returns:
        df_results: Prediction DataFrame from predict_fold.
        models_dict: Keys: scaler_pca, pca, cntr, spike_threshold,
            classifier_l1, scaler_l1, classifier_l2, expert_models,
            clf_feature_cols, use_ridge_regime.
    """
    from src import features  # local import to avoid circular dependency

    tr_start, tr_end, te_start, te_end = fold_def
    print(f"  Train: {tr_start} → {tr_end}")
    print(f"  Test:  {te_start} → {te_end}")

    # STEP 2 - fold-specific spike threshold
    spike_threshold = compute_spike_threshold(
        df_hourly_full, tr_start, tr_end, zone_cfg, model_cfg
    )

    # STEP 3 - rebuild features with this fold's threshold
    df_with_spike = features.build_daily_features(
        df_hourly_full, zone_cfg, spike_threshold=spike_threshold
    )
    df_train = df_with_spike.loc[tr_start:tr_end].copy()
    df_test  = df_with_spike.loc[te_start:te_end].copy()
    pivot_train = price_pivot_full.reindex(df_train.index)
    pivot_test  = price_pivot_full.reindex(df_test.index)

    # STEP 4 - leaf masks + ridge-fallback decision
    mask_A, mask_B, mask_normal, use_ridge_regime = build_leaf_masks(
        df_train, df_hourly_full, spike_threshold, zone_cfg, model_cfg
    )

    # STEPS 5+6 - PCA + FCM on normal training days
    scaler_pca, pca_fit, cntr, hard_labels, normal_idx_valid = fit_pca_and_fcm(
        df_train, pivot_train, mask_normal, model_cfg
    )

    # STEP 7 - assemble regime labels
    regime_labels = assemble_regime_labels(
        df_train, mask_A, mask_B, mask_normal, hard_labels, normal_idx_valid
    )

    # Build classifier feature matrix (daily ⊕ pivot, NaN-dropped)
    df_train_clf = pd.concat([df_train, pivot_train], axis=1).dropna()
    df_test_clf  = pd.concat([df_test,  pivot_test],  axis=1).dropna()
    clf_cols = df_train_clf.columns.tolist()
    X_clf = df_train_clf[clf_cols].values
    y_clf = regime_labels.reindex(df_train_clf.index).values

    # STEPS 8a/8b - classifiers
    classifier_l2 = train_classifier_l2(X_clf, y_clf, model_cfg)
    classifier_l1, scaler_l1 = train_classifier_l1(X_clf, y_clf, model_cfg)

    # Build the (date × hour) price matrix used by experts and predict_fold
    pc = zone_cfg.schema.price
    h2 = df_hourly_full.copy()
    h2["_d"] = pd.to_datetime(h2.index.normalize().date)
    h2["_h"] = h2.index.hour
    price_mat = h2.pivot_table(index="_d", columns="_h", values=pc, aggfunc="mean")
    price_mat.index = pd.to_datetime(price_mat.index)
    price_mat.columns = list(range(24))

    # STEP 9 - experts
    expert_models = train_expert_models(
        df_train_clf, clf_cols, y_clf, price_mat, use_ridge_regime, model_cfg
    )

    # STEP 10 - predict
    df_results = predict_fold(
        df_test_clf, clf_cols,
        classifier_l1, scaler_l1, classifier_l2,
        expert_models, price_mat,
    )

    # STEP 11 - package
    models_dict = {
        "scaler_pca":       scaler_pca,
        "pca":              pca_fit,
        "cntr":             cntr,
        "spike_threshold":  spike_threshold,
        "classifier_l1":    classifier_l1,
        "scaler_l1":        scaler_l1,
        "classifier_l2":    classifier_l2,
        "expert_models":    expert_models,
        "clf_feature_cols": clf_cols,
        "use_ridge_regime": use_ridge_regime,
    }
    return df_results, models_dict


def walk_forward(
    df_daily_full: pd.DataFrame,
    price_pivot_full: pd.DataFrame,
    df_hourly_full: pd.DataFrame,
    zone_cfg: ZoneConfig,
    model_cfg: ModelConfig,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Run run_fold for every entry in zone_cfg.fold_definitions.

    Concatenates fold prediction DataFrames into a single sorted
    DataFrame spanning the full out-of-sample evaluation surface. Stores
    each fold's models for post-hoc inspection.

    Args:
        df_daily_full: Daily feature matrix.
        price_pivot_full: 24h lagged price pivot.
        df_hourly_full: Full hourly DataFrame.
        zone_cfg: Zone configuration.
        model_cfg: Model configuration .

    Returns:
        df_predictions: All folds concatenated, sorted by date.
        all_models: List of models_dict, one per fold (same order as
            model_cfg.fold_definitions).
    """
    folds = zone_cfg.fold_definitions


    # Validate fold definitions (was cell 30 in the original notebook)
    print(f"{'Fold':<5} {'Train start':<13} {'Train end':<13} "
          f"{'Train days':<12} {'Test start':<13} {'Test end':<13} {'Test days'}")
    print("-" * 85)
    fold_stats = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        if not (pd.Timestamp(tr_e) < pd.Timestamp(te_s)):
            raise ValueError(
                f"Fold {i+1}: train_end ({tr_e}) must precede test_start ({te_s})"
            )
        tr_days = int(((df_daily_full.index >= tr_s) &
                       (df_daily_full.index <= tr_e)).sum())
        te_days = int(((df_daily_full.index >= te_s) &
                       (df_daily_full.index <= te_e)).sum())
        fold_stats.append((tr_days, te_days))
        print(f"{i+1:<5} {tr_s:<13} {tr_e:<13} {tr_days:<12} "
              f"{te_s:<13} {te_e:<13} {te_days}")

    for i in range(1, len(folds)):
        if fold_stats[i][0] <= fold_stats[i-1][0]:
            raise ValueError(
                f"Fold {i+1} training window ({fold_stats[i][0]} days) "
                f"is not larger than fold {i} ({fold_stats[i-1][0]} days)"
            )
    print("\nAll fold assertions passed.\n")

    # Execute folds
    all_results: List[pd.DataFrame] = []
    all_models:  List[Dict[str, Any]] = []

    for i, fold_def in enumerate(folds):
        print(f"\n{'='*55}")
        print(f"Fold {i+1} of {len(folds)}")
        print(f"{'='*55}")
        results, models = run_fold(
            df_daily_full, price_pivot_full, df_hourly_full,
            fold_def, zone_cfg, model_cfg,
        )
        all_results.append(results)
        all_models.append(models)
        print(f"Fold {i+1} complete - {len(results)} test days predicted")

    # Concatenate and sort
    df_predictions = pd.concat(all_results, ignore_index=True)
    df_predictions = df_predictions.sort_values("date").reset_index(drop=True)

    print(f"\nTotal predicted days: {len(df_predictions)}")
    print(f"Date range: {df_predictions['date'].min()} → "
          f"{df_predictions['date'].max()}")

    return df_predictions, all_models




def load_or_train(
    zone_cfg,
    model_cfg,
    df_daily: pd.DataFrame,
    price_pivot_lag1: pd.DataFrame,
    df_hourly: pd.DataFrame,
    force_retrain: bool = False,
):
    """Return cached (df_preds, all_fold_models) for this zone, or train from scratch."""
    cache_dir   = model_cfg.results_dir / "cache" / zone_cfg.zone
    preds_path  = cache_dir / "df_preds.parquet"
    models_path = cache_dir / "all_fold_models.pkl"

    if not force_retrain and preds_path.exists() and models_path.exists():
        print(f"[{zone_cfg.zone}] Loading cached results from {cache_dir}")
        df_preds = pd.read_parquet(preds_path)
        with open(models_path, "rb") as f:
            all_fold_models = pickle.load(f)
        print(f"  Loaded {len(df_preds)} predictions, {len(all_fold_models)} folds")
        return df_preds, all_fold_models

    print(f"[{zone_cfg.zone}] No cache found - running walk_forward")
    df_preds, all_fold_models = walk_forward(
        df_daily, price_pivot_lag1, df_hourly, zone_cfg, model_cfg
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(preds_path)
    with open(models_path, "wb") as f:
        pickle.dump(all_fold_models, f)
    print(f"  Saved cache to {cache_dir}")

    return df_preds, all_fold_models





def extract_pca_loadings(
    df_daily_full: pd.DataFrame,
    price_pivot_full: pd.DataFrame,
    fold_models: Dict[str, Any],
) -> pd.DataFrame:
    """Extract a PCA loadings DataFrame from a fitted fold for inspection.  

    Uses the same PCA_EXCLUDE_COLS constant as run_fold, so the column
    alignment is guaranteed identical (no risk of drift). Output is the
    structured (n_features × n_components) loadings matrix used in your
    iterative feature audit.

    Args:
        df_daily_full: Daily feature matrix.
        price_pivot_full: 24h lagged price pivot.
        fold_models: A single fold's models_dict from run_fold.

    Returns:
        DataFrame of PCA loadings, rows = feature names, columns = PC_1..PC_k.
    """

    # Reconstruct the same column alignment used inside fit_pca_and_fcm
    pca_cols = [c for c in df_daily_full.columns if c not in PCA_EXCLUDE_COLS]
    pivot_cols = [f"pivot_hour_{h}" for h in price_pivot_full.columns]
    all_combined_cols = pca_cols + pivot_cols

    # Pull components matrix from the fitted PCA
    loadings = fold_models["pca"].components_

    # Sanity check - catches silent drift if PCA_EXCLUDE_COLS ever diverges
    if loadings.shape[1] != len(all_combined_cols):
        raise ValueError(
            f"PCA loadings have {loadings.shape[1]} features but "
            f"{len(all_combined_cols)} column labels were reconstructed. "
            "Check that PCA_EXCLUDE_COLS matches the feature matrix "
            "used inside fit_pca_and_fcm."
        )

    df_loadings = pd.DataFrame(
        loadings,
        columns=all_combined_cols,
        index=[f"PC_{i+1}" for i in range(loadings.shape[0])],
    ).T

    return df_loadings


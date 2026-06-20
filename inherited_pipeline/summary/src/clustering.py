"""
src/clustering.py — Ward linkage hierarchical clustering for daily price profiles.

Key design decisions:
  - Clustering is ALWAYS fitted on training data only (no leakage by construction).
  - Test days are assigned to the nearest training centroid via Euclidean distance.
  - The 70/30 split uses `int(np.floor(n_days * 0.70))` — identical to the original
    zone-specific notebooks, so saved CSV files remain reproducible.
  - `run_zone_clustering` unifies the three near-identical zone notebooks into one call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "fit_ward_linkage",
    "assign_to_nearest_centroid",
    "compute_centroids",
    "assign_clusters_train_test",
    "run_zone_clustering",
    "plot_cluster_centroids",
    "plot_cluster_calendar_for_year",
]


# ------------------------------------------------------------------
# Core clustering primitives
# ------------------------------------------------------------------

def fit_ward_linkage(X_train: np.ndarray) -> np.ndarray:
    """Compute the Ward linkage matrix on training price profiles.

    Args:
        X_train: Array of shape (n_train_days, 24) — hourly prices per day.

    Returns:
        Linkage matrix Z of shape (n_train - 1, 4) as returned by scipy.
    """
    from scipy.cluster.hierarchy import linkage
    return linkage(X_train, method="ward", metric="euclidean")


def assign_to_nearest_centroid(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each vector to its nearest centroid using Euclidean distance.

    Labels are 1-based to match fcluster's convention.

    Args:
        vectors: Array of shape (n, 24).
        centroids: Array of shape (k, 24).

    Returns:
        1-D integer array of shape (n,) with values in 1..k.
    """
    dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
    return dists.argmin(axis=1) + 1  # +1 → 1-based labels matching fcluster


def compute_centroids(X_train: np.ndarray, train_labels: np.ndarray, k: int) -> np.ndarray:
    """Compute the mean 24h price profile (centroid) for each cluster.

    Args:
        X_train: Array of shape (n_train, 24).
        train_labels: 1-based integer labels for each training row.
        k: Number of clusters.

    Returns:
        Array of shape (k, 24) — one centroid per cluster.
    """
    return np.array([
        X_train[train_labels == c].mean(axis=0)
        for c in range(1, k + 1)
    ])


def assign_clusters_train_test(
    Z: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut the dendrogram at k clusters, compute centroids, assign test days.

    Args:
        Z: Linkage matrix from fit_ward_linkage.
        X_train: Training price profiles, shape (n_train, 24).
        X_test: Test price profiles, shape (n_test, 24).
        train_mask: Boolean mask over all days (True = training day).
        test_mask: Boolean mask over all days (True = test day).
        k: Number of clusters.

    Returns:
        (train_labels, test_labels, centroids) where labels are 1-based integers.
    """
    from scipy.cluster.hierarchy import fcluster
    train_labels = fcluster(Z, t=k, criterion="maxclust")
    centroids    = compute_centroids(X_train, train_labels, k)
    test_labels  = assign_to_nearest_centroid(X_test, centroids)
    return train_labels, test_labels, centroids


# ------------------------------------------------------------------
# Full zone clustering pipeline
# ------------------------------------------------------------------

def run_zone_clustering(
    zone: str,
    k: int,
    price_clean_dir: Path = Path("data/clean"),
    output_dir: Path = Path("data/processed"),
    years_back: int = 3,
) -> pd.DataFrame:
    """Run the full Ward clustering pipeline for one zone and write the cluster CSV.

    Faithfully reproduces the logic from the three zone-specific pattern detection
    notebooks. The 70/30 train/test split uses `int(np.floor(n_days * 0.70))`.

    Args:
        zone: Zone label (e.g. "DK1") — used to locate the preprocessed price CSV
              and name the output file.
        k: Number of clusters (k_focus) chosen for this zone.
        price_clean_dir: Directory containing {zone}_preprocessed.csv.
        output_dir: Directory for the output {zone}_date_cluster.csv.
        years_back: Keep only the last N calendar years (0 or None = keep all).

    Returns:
        DataFrame with columns [date, cluster] for all days in the selected window.
    """
    price_clean_dir = Path(price_clean_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load preprocessed CSV ---
    csv_path = price_clean_dir / f"{zone}_preprocessed.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Preprocessed price file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # --- Optional time-window filter (last N calendar years) ---
    if years_back:
        max_year = df["date"].dt.year.max()
        min_year = max_year - years_back + 1
        df = df[df["date"].dt.year >= min_year].reset_index(drop=True)

    # --- Build 24h price matrix ---
    hour_cols = sorted([c for c in df.columns if c.startswith("h")])
    X = df[hour_cols].to_numpy(dtype=float)

    # --- 70 / 30 train/test split (identical formula to original notebooks) ---
    n_days  = len(df)
    n_train = int(np.floor(n_days * 0.70))
    train_mask = np.zeros(n_days, dtype=bool)
    train_mask[:n_train] = True
    test_mask = ~train_mask

    X_train = X[train_mask]
    X_test  = X[test_mask]

    print(f"{zone}: {n_days} days → train={n_train}, test={n_days - n_train}, k={k}")

    # --- Hierarchical clustering on train only ---
    Z = fit_ward_linkage(X_train)
    train_labels, test_labels, _ = assign_clusters_train_test(
        Z, X_train, X_test, train_mask, test_mask, k
    )

    # --- Reconstruct full label array in original day order ---
    all_labels = np.empty(n_days, dtype=int)
    all_labels[train_mask] = train_labels
    all_labels[test_mask]  = test_labels

    # --- Cluster size report ---
    sizes = pd.Series(all_labels).value_counts().sort_index()
    print(f"  Cluster sizes:\n{sizes.to_string()}")
    small = sizes[sizes < 60]
    if not small.empty:
        print(f"  WARNING: clusters {small.index.tolist()} have < 60 days (see CLAUDE.md Bug 7)")

    # --- Write output CSV ---
    df_out = pd.DataFrame({"date": df["date"].values, "cluster": all_labels})
    out_path = output_dir / f"{zone}_date_cluster.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return df_out


# ------------------------------------------------------------------
# Visualisation helpers
# ------------------------------------------------------------------

def plot_cluster_centroids(
    centroids_df: pd.DataFrame,
    k: int,
    zone: str,
) -> None:
    """Plot the 24h mean price profile for each cluster.

    Args:
        centroids_df: DataFrame of shape (k, 24) — one row per cluster, columns h00…h23.
        k: Number of clusters (for colour-map sizing).
        zone: Zone label used in the figure title.
    """
    import matplotlib.pyplot as plt

    base_cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(10, 4))
    for i, (cluster_id, row) in enumerate(centroids_df.iterrows()):
        plt.plot(row.values, label=f"Cluster {cluster_id}", color=base_cmap(i % base_cmap.N))
    plt.title(f"{zone}: cluster centroid profiles (k={k})")
    plt.xlabel("Hour of day")
    plt.ylabel("Price (EUR/MWh)")
    plt.xticks(range(24), [f"{h:02d}" for h in range(24)], rotation=45, fontsize=7)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cluster_calendar_for_year(
    df_year: pd.DataFrame,
    year: int,
    cluster_col: str,
    cmap,
    norm,
    k: int,
    zone: str,
) -> None:
    """Plot a calendar-style heatmap of cluster assignments for one year.

    Each cell in the grid is one day; rows are week indices and columns are weekdays
    (0 = Monday, 6 = Sunday). Missing days are shown in white.

    Args:
        df_year: DataFrame for one year with columns [week_index, weekday, cluster_col].
        year: Calendar year (used in the figure title).
        cluster_col: Name of the column holding cluster labels.
        cmap: Discrete ListedColormap instance.
        norm: BoundaryNorm matching the colourmap.
        k: Number of clusters (for the colourmap tick labels).
        zone: Zone label used in the figure title.
    """
    import matplotlib.pyplot as plt

    unique_clusters = np.sort(df_year[cluster_col].unique())
    cluster_to_idx  = {cl: idx + 1 for idx, cl in enumerate(unique_clusters)}

    n_weeks   = int(df_year["week_index"].max()) + 1
    n_weekdays = 7
    calendar_mat = np.full((n_weeks, n_weekdays), np.nan)

    for _, row in df_year.iterrows():
        w = int(row["week_index"])
        d = int(row["weekday"])
        calendar_mat[w, d] = cluster_to_idx[row[cluster_col]]

    plt.figure(figsize=(10, 12))
    plt.imshow(calendar_mat, aspect="auto", cmap=cmap, norm=norm)

    for w in range(n_weeks):
        for d in range(n_weekdays):
            val = calendar_mat[w, d]
            if not np.isnan(val):
                cl_label = unique_clusters[int(val) - 1]
                plt.text(d, w, str(cl_label), ha="center", va="center",
                         fontsize=7, color="black")

    plt.yticks(np.arange(n_weeks), [f"W{w}" for w in range(n_weeks)])
    plt.xticks(np.arange(n_weekdays), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.title(f"{zone}: cluster calendar {year} (k={k})")
    plt.xlabel("Weekday")
    plt.ylabel("Week index")

    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=plt.gca(), orientation="vertical", fraction=0.02, pad=0.04
    )
    cb.set_ticks(np.arange(1, len(unique_clusters) + 1))
    cb.set_ticklabels([str(c) for c in unique_clusters])
    cb.set_label("Cluster")

    plt.tight_layout()
    plt.show()

"""
src/utils.py — Shared utilities used across the pipeline.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "get_project_root",
    "load_secrets",
    "add_dates_to_output",
    "ZONE_SHORT_CODE",
]

# Canonical mapping from 3-char zone identifiers to the 2-char codes used in
# model file names (e.g. xgb_cluster_model_DK.joblib).
ZONE_SHORT_CODE: dict[str, str] = {
    "DK1": "DK",
    "ES":  "ES",
    "NO2": "NO",
}


def get_project_root() -> Path:
    """Return the summary/ directory (parent of src/), anchored to this file's location."""
    return Path(__file__).resolve().parent.parent


def load_secrets(secrets_path: Path | None = None) -> dict[str, str]:
    """Load key=value pairs from a .env file and return them as a dict (does not mutate os.environ)."""
    if secrets_path is None:
        secrets_path = get_project_root() / "config" / "secrets.env"

    secrets: dict[str, str] = {}
    if not Path(secrets_path).exists():
        return secrets

    with open(secrets_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            secrets[key.strip()] = value.strip().strip('"').strip("'")

    return secrets


def add_dates_to_output(pred_df: "pd.DataFrame", date_series: "pd.Series") -> "pd.DataFrame":
    """Re-attach a saved date column to a prediction DataFrame that had dates dropped before modelling."""
    import pandas as pd  # local import to avoid top-level side-effects
    out = pred_df.copy()
    out.insert(0, "date", date_series.values)
    return out

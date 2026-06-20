"""Zone-specific feature builders, attached via ZoneConfig.extra_feature_builders."""
import pandas as pd
from src.config import ColumnSchema


def precip_rolling_features(
    df_daily: pd.DataFrame,
    df_hourly: pd.DataFrame,
    schema: ColumnSchema,
) -> pd.DataFrame:
    """Append precip_roll7 and precip_roll30 (hydro reservoir proxy for NO2).

    Both windows are shift(1) lagged to preserve the T-1 leak guarantee.

    Args:
        df_daily: Daily feature DataFrame mid-construction.
        df_hourly: Hourly DataFrame (used to recompute daily precip if needed).
        schema: Provides the precipitation column name.

    Returns:
        df_daily with two new columns appended; unchanged if the
        precipitation column is absent.
    """
    ...
from pathlib import Path
import sys

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data_new"

sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from src.zones import ZONES
from src import data
from src import features
from src import model


zone_cfg = ZONES["DK1"]

# run the new data pipeline
df_hourly_new = data.impute(data.load_and_validate(zone_cfg), zone_cfg)

#  load the OLD reference
df_hourly_old = pd.read_parquet("tests/golden/df_hourly.parquet")

# compare
try:
    # Try to compare the dataframes
    pd.testing.assert_frame_equal(df_hourly_new, df_hourly_old, atol=1e-9, check_freq=False)
    
    # If the line above succeeds, it moves to this line
    print("data.py: HOURLY PARITY OK")

except AssertionError as e:
    # If the assertion fails, it jumps here instead of crashing your script
    print("HOURLY PARITY FAILED!")
    print(f"Details: {e}")

df_daily_new = features.build_daily_features(data.impute(data.load_and_validate(zone_cfg),zone_cfg),zone_cfg, spike_threshold=None)

df_daily_old = pd.read_parquet("tests/golden/df_daily.parquet")

# compare
try:
    # Try to compare the dataframes
    pd.testing.assert_frame_equal(df_daily_new, df_daily_old, atol=1e-9, check_freq=False)
    
    # If the line above succeeds, it moves to this line
    print("data.py: DAILY PARITY OK")

except AssertionError as e:
    # If the assertion fails, it jumps here instead of crashing your script
    print("DAILY PARITY FAILED!")
    print(f"Details: {e}")


price_pivot_lag1_new = features.build_price_pivot_lag1(df_hourly_new, df_daily_new, zone_cfg)
price_pivot_lag1_old = pd.read_parquet("tests/golden/price_pivot_lag1.parquet")

# compare
try:
    # Try to compare the dataframes
    pd.testing.assert_frame_equal(price_pivot_lag1_new, price_pivot_lag1_old, atol=1e-9, check_freq=False)
    
    # If the line above succeeds, it moves to this line
    print("data.py: PRICE PIVOT LAG1 PARITY OK")

except AssertionError as e:
    # If the assertion fails, it jumps here instead of crashing your script
    print("PRICE PIVOT LAG1 PARITY FAILED!")
    print(f"Details: {e}")



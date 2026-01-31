import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import traceback

class Config:
    DATA_PERIOD_YEARS = 10
    INTERVAL = "1d"
    USE_HEIKIN_ASHI = True
    KEY_VALUE = 2
    ATR_PERIOD = 10
    LOOK_AHEAD_BARS = 60
    TARGET_PROFIT = 0.105
    STOP_LOSS = 0.05
    COMMISSION_PCT = 0.002
    SLIPPAGE_PCT = 0.001
    INDEX_SYMBOL = "XU100.IS"

def test_adgyo():
    symbol = "ADGYO.IS"
    config = Config()
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(config.DATA_PERIOD_YEARS * 365))
        
        print(f"Downloading {symbol}...")
        df = yf.download(symbol, start=start_date, end=end_date, interval=config.INTERVAL, progress=False)
        
        index_df = yf.download(config.INDEX_SYMBOL, start=start_date, end=end_date, interval=config.INTERVAL, progress=False)

        print(f"Columns before cleanup: {df.columns}")
        
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        
        if isinstance(df.columns, pd.MultiIndex) or (len(df.columns) > 0 and isinstance(df.columns[0], tuple)):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        for col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        print(f"Columns after cleanup: {df.columns}")

        # UT Bot - simplified logic that mirrors model_egitimi.py
        from model_egitimi import calculate_heikin_ashi, calculate_ut_bot, calculate_pivot_points, engineer_features, create_target_label
        
        print("Calculating HA...")
        df = calculate_heikin_ashi(df)
        
        print("Calculating UT Bot...")
        df = calculate_ut_bot(df, config.KEY_VALUE, config.ATR_PERIOD, config.USE_HEIKIN_ASHI)
        
        print("Calculating Pivot Points & RS...")
        # Join index first
        idx_cols = ['Close', 'High', 'Low']
        idx_data = index_df[idx_cols].copy()
        if hasattr(idx_data.columns, 'nlevels') and idx_data.columns.nlevels > 1:
            idx_data.columns = idx_data.columns.get_level_values(0)
        idx_data.columns = [f'Index_{c}' for c in idx_data.columns]
        df = df.join(idx_data, how='left')
        df[idx_data.columns] = df[idx_data.columns].ffill()
        
        df = calculate_pivot_points(df, swing_length=10)
        
        print("Engineering features...")
        df = engineer_features(df)
        
        print("Creating targets...")
        df = create_target_label(df, config.LOOK_AHEAD_BARS, config)
        
        print("Success!")
        
    except Exception:
        print("\n=== TRACEBACK ===")
        traceback.print_exc()

if __name__ == "__main__":
    test_adgyo()

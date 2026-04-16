import pandas as pd
import numpy as np
import os

def load_and_prep_data(filepath, freq='daily'):
    """
    Loads price data from CSV and calculates log returns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
        
    df = pd.read_csv(filepath, sep=';', parse_dates=['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    
    # Safely convert to numeric in case string commas persisted
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    
    if freq == 'weekly':
        df = df.resample('W-FRI').last()
        df.dropna(inplace=True)
        
    # Calculate percentage log returns (useful for stable GARCH convergence)
    returns = np.log(df / df.shift(1)).dropna() * 100 
    
    return returns

import pandas as pd
from src.imputation_strategies import (
    impute_with_mean,
    impute_with_median,
    impute_with_interpolation,
)
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, IMPUTATION_METHOD

def initial_cleaning(df):
    # Basic cleaning steps
    df = df.drop_duplicates()
    # Convert date columns to datetime
    df['date'] = pd.to_datetime(df['date'])
    return df

def handle_missing_values(df, method):
    if method == 'mean':
        return impute_with_mean(df)
    elif method == 'median':
        return impute_with_median(df)
    elif method == 'interpolation':
        return impute_with_interpolation(df)
    else:
        raise ValueError(f"Imputation method {method} not recognized.")

def preprocess_data():
    df = pd.read_parquet(RAW_DATA_PATH)
    df = initial_cleaning(df)
    df = handle_missing_values(df, method=IMPUTATION_METHOD)
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    return df

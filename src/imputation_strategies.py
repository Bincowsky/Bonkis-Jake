import pandas as pd

def impute_with_mean(df):
    return df.fillna(df.mean())

def impute_with_median(df):
    return df.fillna(df.median())

def impute_with_interpolation(df):
    return df.interpolate(method='time')

"""
data_loader.py -- Load, validate, preprocess any CSV dataset
"""
import pandas as pd
import numpy as np
import io
import urllib.request
from sklearn.preprocessing import LabelEncoder
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"  Loaded: {csv_path}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    return df
def load_from_url(url, sep=' ', header=None, column_names=None):
    print(f"  Downloading from: {url}")
    with urllib.request.urlopen(url) as response:
        raw = response.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(raw), sep=sep, header=header)
    if column_names:
        df.columns = column_names
    print(f"  Downloaded: {df.shape[0]} rows x {df.shape[1]} cols")
    return df
def validate_dataset(df, outcome_col):
    warnings = []
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found.")
    if df[outcome_col].isnull().sum() > 0:
        warnings.append(f"'{outcome_col}' has missing values -- rows will be dropped.")
    if len(df) < 50:
        warnings.append(f"Only {len(df)} rows -- results may be unreliable.")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        warnings.append(f"Missing values in: {list(missing_cols.index)} -- will be filled.")
    return warnings
def preprocess(df, outcome_col):
    df = df.copy()
    df = df.dropna(subset=[outcome_col])
    for col in df.columns:
        if col == outcome_col:
            continue
        if df[col].isnull().sum() > 0:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)
def encode_dataframe(df, outcome_col):
    df_enc   = df.copy()
    encoders = {}
    for col in df_enc.columns:
        if col == outcome_col:
            continue
        if not pd.api.types.is_numeric_dtype(df_enc[col]) or str(df_enc[col].dtype) == 'category':
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
    if not pd.api.types.is_numeric_dtype(df_enc[outcome_col]):
        le = LabelEncoder()
        df_enc[outcome_col] = le.fit_transform(df_enc[outcome_col])
        encoders[outcome_col] = le
    return df_enc, encoders

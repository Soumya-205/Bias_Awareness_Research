"""
mitigation.py -- Mitigation strategies for flagged features
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
def mitigate_feature_splitting(df, feature_name):
    df_new = df.copy()
    vc = df[feature_name].value_counts()
    df_new[f'{feature_name}_frequency_bucket'] = df[feature_name].map(
        lambda x: 'high' if vc.get(x, 0) > vc.median() else 'low')
    df_new[f'{feature_name}_is_common'] = df[feature_name].map(
        lambda x: 1 if vc.get(x, 0) > vc.quantile(0.75) else 0)
    return df_new.drop(columns=[feature_name])
def mitigate_similarity_reweighting(df, feature_name, outcome_col, k=5):
    features = [c for c in df.columns if c != outcome_col]
    df_enc   = df[features].copy()
    for col in df_enc.columns:
        if not pd.api.types.is_numeric_dtype(df_enc[col]):
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(df_enc.values)
    _, indices = nbrs.kneighbors(df_enc.values)
    feat_vals = df[feature_name].values
    weights   = np.ones(len(df))
    for i in range(len(df)):
        different = sum(1 for j in indices[i][1:] if feat_vals[j] != feat_vals[i])
        weights[i] = 1.0 + (different / k)
    return weights / weights.mean()
def mitigate_feature_suppression(df, feature_name):
    return df.drop(columns=[feature_name])
def apply_mitigation(df, flagged_features, outcome_col, method='splitting'):
    df_mitigated   = df.copy()
    log            = {}
    sample_weights = {}
    for feat in flagged_features:
        if feat not in df_mitigated.columns:
            log[feat] = 'skipped (already removed)'
            continue
        if method == 'splitting':
            df_mitigated = mitigate_feature_splitting(df_mitigated, feat)
            log[feat] = f'split into {feat}_frequency_bucket and {feat}_is_common'
        elif method == 'reweighting':
            w = mitigate_similarity_reweighting(df_mitigated, feat, outcome_col)
            sample_weights[feat] = w
            log[feat] = f'reweighting computed (mean weight: {w.mean():.3f})'
        elif method == 'suppression':
            df_mitigated = mitigate_feature_suppression(df_mitigated, feat)
            log[feat] = 'feature removed'
        else:
            log[feat] = f'unknown method: {method} -- skipped'
    return {'df_mitigated': df_mitigated, 'log': log, 'sample_weights': sample_weights}

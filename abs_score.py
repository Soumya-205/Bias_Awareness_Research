"""
abs_score.py -- Attribute Bias Score (ABS) and its four signals
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from config import ABS_WEIGHTS, KNN_NEIGHBORS
def compute_neighborhood_inconsistency(feature, outcome, k=KNN_NEIGHBORS):
    if not pd.api.types.is_numeric_dtype(outcome) or str(outcome.dtype) == 'category':
        le = LabelEncoder()
        outcome_vals = le.fit_transform(outcome)
    else:
        outcome_vals = outcome.values

    is_low_card = (not pd.api.types.is_numeric_dtype(feature) or str(feature.dtype) == 'category'
                   or feature.nunique() <= 10)

    if is_low_card:
        # FIX: sklearn's NearestNeighbors does not break ties randomly -- for a
        # low-cardinality feature, every point sharing a value gets distance 0
        # to hundreds of others, and the tree traversal returns the SAME fixed
        # subset of "neighbors" for every query with that value. If that one
        # fixed subset happens to share an outcome, NI silently computes to
        # 0.0 for the whole feature, regardless of the feature's true
        # within-group outcome variance. Bypass k-NN entirely for
        # low-cardinality features and compute the real within-group outcome
        # std directly, weighted by group size.
        df_tmp = pd.DataFrame({'f': feature.values, 'y': outcome_vals})
        grouped = df_tmp.groupby('f')['y'].agg(['std', 'count'])
        grouped['std'] = grouped['std'].fillna(0.0)
        weighted_ni = (grouped['std'] * grouped['count']).sum() / grouped['count'].sum()
        return round(float(weighted_ni), 4)

    feat_vals = feature.values.reshape(-1, 1).astype(float)
    r = feat_vals.max() - feat_vals.min()
    if r > 0:
        feat_vals = (feat_vals - feat_vals.min()) / r
    n = len(feat_vals)
    actual_k = min(k, n - 1)
    if actual_k < 1:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=actual_k + 1).fit(feat_vals)
    _, indices = nbrs.kneighbors(feat_vals)
    inconsistencies = [np.std(outcome_vals[indices[i][1:]]) for i in range(n)]
    return round(np.mean(inconsistencies), 4)
def compute_dominance(feature):
    vc = feature.value_counts(normalize=True)
    if len(vc) <= 1:
        return 1.0
    return round(float(vc.iloc[0] - vc.iloc[1:].mean()), 4)
def compute_cardinality(feature, n_rows):
    return round(feature.nunique() / n_rows, 4)
def compute_instability(feature, time_col=None, df=None):
    if time_col is None or df is None:
        return 0.0
    dominant = feature.value_counts().index[0]
    periods  = sorted(df[time_col].unique())
    if len(periods) < 2:
        return 0.0
    freqs  = [(feature[df[time_col] == t] == dominant).mean() for t in periods]
    drifts = [abs(freqs[i] - freqs[i+1]) for i in range(len(freqs)-1)]
    return round(np.mean(drifts), 4)
def compute_abs(feature, outcome, n_rows, time_col=None, df=None):
    w  = ABS_WEIGHTS
    ni = compute_neighborhood_inconsistency(feature, outcome)
    d  = compute_dominance(feature)
    c  = compute_cardinality(feature, n_rows)
    i  = compute_instability(feature, time_col, df)
    score = w['neighborhood_inconsistency']*ni + w['dominance']*d + w['cardinality']*c + w['instability']*i
    return {
        'abs_score': round(score, 4),
        'signals': {'neighborhood_inconsistency': ni, 'dominance': d, 'cardinality': c, 'instability': i}
    }
def compute_abs_all_features(df, outcome_col, exclude=None, time_col=None):
    exclude  = exclude or []
    features = [c for c in df.columns if c != outcome_col and c not in exclude]
    n_rows   = len(df)
    return {f: compute_abs(df[f], df[outcome_col], n_rows, time_col, df) for f in features}

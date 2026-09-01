"""
mutual_information.py -- MI computation (KSG + categorical + binning fallback)
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy.special import digamma
from config import KSG_K, BINNING_FALLBACK_BINS, LOW_CARDINALITY_THRESHOLD
def compute_mi_categorical(x, y):
    xy = pd.crosstab(x, y, normalize=True)
    px = x.value_counts(normalize=True)
    py = y.value_counts(normalize=True)
    mi = 0.0
    for xi in xy.index:
        for yi in xy.columns:
            p_xy = xy.loc[xi, yi] if yi in xy.columns else 0
            p_x  = px.get(xi, 0)
            p_y  = py.get(yi, 0)
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))
    return max(0.0, mi)
def compute_mi_ksg(x, y, k=KSG_K):
    if not pd.api.types.is_numeric_dtype(y) or str(y.dtype) == 'category':
        le = LabelEncoder()
        y_enc = le.fit_transform(y).reshape(-1, 1)
    else:
        y_enc = y.values.reshape(-1, 1)
    x_vals = x.values.reshape(-1, 1)
    n = len(x_vals)
    xy = np.hstack([x_vals, y_enc])
    nbrs_xy = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xy)
    dist_xy, _ = nbrs_xy.kneighbors(xy)
    eps = dist_xy[:, k]
    nbrs_x = NearestNeighbors(metric='chebyshev').fit(x_vals)
    nx = np.array([len(nbrs_x.radius_neighbors([x_vals[i]], radius=eps[i], return_distance=False)[0])-1 for i in range(n)])
    nbrs_y = NearestNeighbors(metric='chebyshev').fit(y_enc)
    ny = np.array([len(nbrs_y.radius_neighbors([y_enc[i]], radius=eps[i], return_distance=False)[0])-1 for i in range(n)])
    mi = digamma(k) - np.mean(digamma(nx+1) + digamma(ny+1)) + digamma(n)
    return max(0.0, float(mi))
def compute_mi(feature, outcome):
    is_cat = (not pd.api.types.is_numeric_dtype(feature) or str(feature.dtype) == 'category' or feature.nunique() <= LOW_CARDINALITY_THRESHOLD)
    if is_cat:
        return compute_mi_categorical(feature, outcome)
    try:
        mi = compute_mi_ksg(feature, outcome)
        if mi > 0:
            return mi
    except Exception:
        pass
    binned = pd.cut(feature, bins=BINNING_FALLBACK_BINS, labels=False, duplicates='drop')
    binned = binned.fillna(0).astype(int).astype(str)
    return compute_mi_categorical(binned, outcome)

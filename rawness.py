"""
rawness.py -- Dataset Rawness Score
"""
import numpy as np
from mutual_information import compute_mi
def compute_rawness_score(df, outcome_col):
    features  = [c for c in df.columns if c != outcome_col]
    outcome   = df[outcome_col]
    mi_scores = {f: compute_mi(df[f], outcome) for f in features}
    total_mi  = sum(mi_scores.values())
    if total_mi == 0:
        return {'rawness_score': 0.0, 'feature_mi': {f: 0.0 for f in features},
                'feature_weights': {f: 0.0 for f in features}, 'top_features': []}
    weights      = {f: mi_scores[f] / total_mi for f in features}
    weighted_sum = sum(weights[f] * mi_scores[f] for f in features)
    mi_max       = max(mi_scores.values())
    rs           = weighted_sum / mi_max if mi_max > 0 else 0.0
    sorted_mi    = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    return {
        'rawness_score':   round(rs, 4),
        'feature_mi':      {f: round(v, 4) for f, v in mi_scores.items()},
        'feature_weights': {f: round(v, 4) for f, v in weights.items()},
        'top_features':    [(f, round(mi, 4)) for f, mi in sorted_mi[:5]]
    }
def interpret_rawness(rs):
    if rs > 0.7:   return "HIGH -- strong decision shortcuts detected"
    elif rs > 0.4: return "MODERATE -- some shortcut features present"
    else:          return "LOW -- decision power is well distributed"

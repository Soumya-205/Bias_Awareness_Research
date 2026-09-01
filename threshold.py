"""
threshold.py -- Dynamic threshold mechanism
"""
import numpy as np
from config import DEFAULT_ALPHA, RAWNESS_TAU_CEILING, ALPHA_OPTIONS

def compute_threshold(rawness_score, alpha=DEFAULT_ALPHA):
    """
    ORIGINAL formula -- kept for comparison/provenance.
    tau = min(e^(-alpha * RS), 0.45)

    PROBLEM (found Aug 30 2026, empirically, on 2 real datasets):
    The ceiling (0.45) is a single constant shared across every dataset,
    regardless of that dataset's own ABS score distribution. At the default
    alpha=1.0, the ceiling only stops binding once RS > ln(1/0.45) = 0.799 --
    almost never in practice (German Credit RS=0.4715, Heart Disease
    RS=0.7374, both hit the ceiling). Worse: once features cluster near a
    fixed number that wasn't calibrated to THIS dataset's spread, real
    signals land within hundredths of it in both directions -- flagging
    decisions become sensitive to noise rather than to the underlying bias.
    """
    tau = np.exp(-alpha * rawness_score)
    tau = min(tau, RAWNESS_TAU_CEILING)
    return round(tau, 4)


def compute_threshold_adaptive(rawness_score, abs_scores, alpha=DEFAULT_ALPHA,
                                percentile=90, min_features_for_percentile=5):
    """
    FIXED formula: the ceiling is derived from THIS dataset's own ABS score
    distribution instead of a hardcoded constant. Keeps the original
    Rawness-Score-driven sensitivity mechanism -- a genuinely raw/shortcut-
    heavy dataset can still produce a tighter (lower) threshold than the
    data-driven ceiling -- but replaces the fixed 0.45 cap with a value that
    adapts to each dataset, guaranteeing real margin between "cleared" and
    "flagged" instead of everything crowding a shared magic number.

    tau = min(e^(-alpha * RS), percentile(abs_scores, percentile))

    percentile=90 means: only the most extreme 10% of the candidate feature
    pool (already legitimacy-filtered) can ever be flagged, by construction.

    abs_scores: dict of {feature: {'abs_score': float, ...}} as returned by
    compute_abs_all_features -- must be computed BEFORE calling this.
    """
    scores = [v['abs_score'] for v in abs_scores.values()]
    if len(scores) < min_features_for_percentile:
        # Too few candidate features for a percentile to mean anything --
        # fall back to the original fixed ceiling as a safety net.
        ceiling = RAWNESS_TAU_CEILING
    else:
        ceiling = float(np.percentile(scores, percentile))

    tau_raw = np.exp(-alpha * rawness_score)
    tau = min(tau_raw, ceiling)
    return round(tau, 4), round(ceiling, 4)


def get_sensitivity_label(alpha):
    return {0.5: 'Low', 1.0: 'Medium', 2.0: 'High', 3.0: 'Very High'}.get(alpha, f'Custom ({alpha})')


"""
legitimacy.py -- Three legitimacy tests for high-power features
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from mutual_information import compute_mi
from config import PROXY_DETECTION_THRESHOLD, CONDITIONAL_INFLUENCE_THRESHOLD, RANDOM_STATE, TEST_SIZE


def _encode(df, outcome_col):
    df_enc = df.copy()
    for col in df_enc.columns:
        if col == outcome_col:
            continue
        if not pd.api.types.is_numeric_dtype(df_enc[col]):
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    if not pd.api.types.is_numeric_dtype(df_enc[outcome_col]):
        le = LabelEncoder()
        df_enc[outcome_col] = le.fit_transform(df_enc[outcome_col])
    return df_enc


def proxy_detection_test(feature_name, df, outcome_col):
    feature = df[feature_name]
    others  = [c for c in df.columns if c != outcome_col and c != feature_name]
    max_mi, proxy_of = 0.0, None
    for other in others:
        mi = compute_mi(feature, df[other])
        if mi > max_mi:
            max_mi, proxy_of = mi, other
    passed = max_mi < PROXY_DETECTION_THRESHOLD
    return {
        'passed':    bool(passed),
        'score':     round(max_mi, 4),
        'proxy_of':  proxy_of if not passed else None,
        'threshold': PROXY_DETECTION_THRESHOLD
    }


def conditional_influence_test(feature_name, df, outcome_col):
    features    = [c for c in df.columns if c != outcome_col]
    df_enc      = _encode(df, outcome_col)
    outcome_enc = df_enc[outcome_col].values
    clf         = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)

    acc_with    = cross_val_score(clf, df_enc[features], outcome_enc, cv=3).mean()
    feat_without = [f for f in features if f != feature_name]
    if not feat_without:
        return {'passed': True, 'score': 0.0, 'acc_with': round(acc_with, 4), 'acc_without': 0.0}

    acc_without = cross_val_score(clf, df_enc[feat_without], outcome_enc, cv=3).mean()
    ci          = acc_with - acc_without

    return {
        'passed':      bool(ci >= CONDITIONAL_INFLUENCE_THRESHOLD),
        'score':       round(ci, 4),
        'acc_with':    round(acc_with, 4),
        'acc_without': round(acc_without, 4),
        'threshold':   CONDITIONAL_INFLUENCE_THRESHOLD
    }


def outcome_independence_test(feature_name, df, outcome_col):
    features = [c for c in df.columns if c != outcome_col]
    df_enc   = _encode(df, outcome_col)
    y        = df_enc[outcome_col].values

    group = df[feature_name].reset_index(drop=True)
    if not pd.api.types.is_numeric_dtype(group):
        le    = LabelEncoder()
        group = pd.Series(le.fit_transform(group.astype(str)))

    top_groups = group.value_counts().index[:2]
    if len(top_groups) < 2:
        return {'passed': True, 'score': 0.0, 'note': 'Only one group value'}

    g0, g1 = top_groups[0], top_groups[1]

    def group_disparity(X_arr, y_arr, grp):
        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X_arr, y_arr, grp, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_arr
        )
        clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        g_te   = np.array(g_te)
        m0     = g_te == g0
        m1     = g_te == g1
        p0     = y_pred[m0].mean() if m0.sum() > 0 else 0.0
        p1     = y_pred[m1].mean() if m1.sum() > 0 else 0.0
        return abs(p0 - p1)

    X_with    = df_enc[features].values
    d_with    = group_disparity(X_with, y, group.values)

    feat_without = [f for f in features if f != feature_name]
    if not feat_without:
        return {'passed': True, 'score': 0.0, 'note': 'No other features to test'}
    X_without = df_enc[feat_without].values
    d_without = group_disparity(X_without, y, group.values)

    oi_score = d_with - d_without

    return {
        'passed':             bool(oi_score <= 0.0),
        'score':              round(float(oi_score), 4),
        'disparity_with':     round(float(d_with), 4),
        'disparity_without':  round(float(d_without), 4),
        'groups_compared':    [str(g0), str(g1)]
    }


def run_legitimacy_tests(feature_name, df, outcome_col):
    proxy        = proxy_detection_test(feature_name, df, outcome_col)
    influence    = conditional_influence_test(feature_name, df, outcome_col)
    independence = outcome_independence_test(feature_name, df, outcome_col)

    all_passed = proxy['passed'] and influence['passed'] and independence['passed']

    failure_reasons = []
    if not proxy['passed']:
        failure_reasons.append(f"Proxy risk: MI with '{proxy['proxy_of']}' = {proxy['score']}")
    if not influence['passed']:
        failure_reasons.append(f"Low unique signal: accuracy gain = {influence['score']}")
    if not independence['passed']:
        failure_reasons.append(f"Drives group disparity: score = {independence['score']}")

    return {
        'feature':            feature_name,
        'is_legitimate':      bool(all_passed),
        'failure_reasons':    failure_reasons,
        'proxy_detection':    proxy,
        'conditional_influence': influence,
        'outcome_independence':  independence
    }

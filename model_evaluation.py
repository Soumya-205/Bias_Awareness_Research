"""
model_evaluation.py -- Train RF + LR, compute fairness metrics, compare before/after
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import TEST_SIZE, RANDOM_STATE, RF_N_ESTIMATORS, LR_MAX_ITER, FAIRNESS_ACCURACY_TOLERANCE


def _encode(df, outcome_col):
    df_enc = df.copy()
    for col in df_enc.columns:
        if col == outcome_col:
            continue
        if not pd.api.types.is_numeric_dtype(df_enc[col]) or str(df_enc[col].dtype) == 'category':
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    if not pd.api.types.is_numeric_dtype(df_enc[outcome_col]):
        le = LabelEncoder()
        df_enc[outcome_col] = le.fit_transform(df_enc[outcome_col])
    return df_enc


def compute_fairness_metrics(y_true, y_pred, group):
    group_vals = np.array(group)
    top_two    = pd.Series(group_vals).value_counts().index[:2]
    if len(top_two) < 2:
        return {'note': 'Only one group value'}
    g0, g1 = top_two[0], top_two[1]
    m0, m1 = group_vals == g0, group_vals == g1

    def sm(arr): return float(arr.mean()) if len(arr) > 0 else 0.0

    p0 = sm(y_pred[m0])
    p1 = sm(y_pred[m1])
    tpr0 = sm(y_pred[m0 & (y_true == 1)]) if (m0 & (y_true == 1)).sum() > 0 else 0
    tpr1 = sm(y_pred[m1 & (y_true == 1)]) if (m1 & (y_true == 1)).sum() > 0 else 0
    fpr0 = sm(y_pred[m0 & (y_true == 0)]) if (m0 & (y_true == 0)).sum() > 0 else 0
    fpr1 = sm(y_pred[m1 & (y_true == 0)]) if (m1 & (y_true == 0)).sum() > 0 else 0

    return {
        'groups_compared':           [str(g0), str(g1)],
        'group_0_positive_rate':     round(p0, 4),
        'group_1_positive_rate':     round(p1, 4),
        'demographic_parity_gap':    round(abs(p0 - p1), 4),
        'equalized_odds_tpr_gap':    round(abs(tpr0 - tpr1), 4),
        'equalized_odds_fpr_gap':    round(abs(fpr0 - fpr1), 4),
    }


def get_feature_importances(model, feature_names, model_name):
    raw   = model.feature_importances_ if model_name == 'RandomForest' else np.abs(model.coef_[0])
    total = raw.sum()
    norm  = raw / total if total > 0 else raw
    d     = {f: round(float(v), 4) for f, v in zip(feature_names, norm)}
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))


def evaluate_models(df, outcome_col, sensitive_col, label, sample_weights=None):
    print(f"\n  --- Training models [{label}] ---")

    df_enc       = _encode(df, outcome_col)
    feature_cols = [c for c in df_enc.columns if c != outcome_col]
    X = df_enc[feature_cols].values
    y = df_enc[outcome_col].values

    if not pd.api.types.is_numeric_dtype(df[sensitive_col]):
        le    = LabelEncoder()
        group = pd.Series(le.fit_transform(df[sensitive_col].astype(str)))
    else:
        group = df[sensitive_col].reset_index(drop=True)

    X_train, X_test, y_train, y_test, grp_train, grp_test, idx_train, idx_test = train_test_split(
        X, y, group, np.arange(len(X)),
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_w = sample_weights[idx_train] if sample_weights is not None else None

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'RandomForest':       RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE),
        'LogisticRegression': LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)
    }

    results = {}
    for name, model in models.items():
        print(f"    Training {name}...", end=' ')
        X_tr = X_train_s if name == 'LogisticRegression' else X_train
        X_te = X_test_s  if name == 'LogisticRegression' else X_test

        if train_w is not None:
            model.fit(X_tr, y_train, sample_weight=train_w)
        else:
            model.fit(X_tr, y_train)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = round(accuracy_score(y_test, y_pred), 4)
        try:    auc = round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None
        except: auc = None

        fairness    = compute_fairness_metrics(y_test, y_pred, grp_test)
        importances = get_feature_importances(model, feature_cols, name)
        top5        = list(importances.items())[:5]

        print(f"Accuracy: {acc} | AUC: {auc} | DP Gap: {fairness.get('demographic_parity_gap', 'N/A')}")
        results[name] = {
            'accuracy': acc, 'auc_roc': auc,
            'fairness': fairness,
            'feature_importances': importances,
            'top_features': top5
        }

    return results


def compare_results(before, after, flagged_features):
    comparison = {}
    for name in ['RandomForest', 'LogisticRegression']:
        b, a = before[name], after[name]
        acc_ch = round(a['accuracy'] - b['accuracy'], 4)
        dp_ch  = round(a['fairness'].get('demographic_parity_gap', 0) - b['fairness'].get('demographic_parity_gap', 0), 4)
        tpr_ch = round(a['fairness'].get('equalized_odds_tpr_gap', 0) - b['fairness'].get('equalized_odds_tpr_gap', 0), 4)
        fpr_ch = round(a['fairness'].get('equalized_odds_fpr_gap', 0) - b['fairness'].get('equalized_odds_fpr_gap', 0), 4)

        before_top       = [f for f, _ in b['top_features']]
        after_top        = [f for f, _ in a['top_features']]
        removed_from_top = [f for f in flagged_features if f in before_top and f not in after_top]

        interp = []
        interp.append("Accuracy maintained." if acc_ch >= -FAIRNESS_ACCURACY_TOLERANCE
                       else f"Accuracy dropped {abs(acc_ch)*100:.1f}% -- fairness-accuracy tradeoff.")
        interp.append(f"Demographic Parity IMPROVED (gap -{abs(dp_ch):.4f})." if dp_ch < 0
                       else "Demographic Parity unchanged or worsened.")
        interp.append(f"Equalized Odds TPR IMPROVED (gap -{abs(tpr_ch):.4f})." if tpr_ch < 0
                       else "Equalized Odds TPR unchanged or worsened.")
        interp.append(f"Flagged features dropped from top importance: {removed_from_top}." if removed_from_top
                       else "Flagged features still in top importance.")

        comparison[name] = {
            'accuracy_change': acc_ch, 'dp_gap_change': dp_ch,
            'tpr_gap_change':  tpr_ch, 'fpr_gap_change': fpr_ch,
            'flagged_removed_from_top': removed_from_top,
            'interpretations': interp,
            'before': b, 'after': a
        }
    return comparison


def compare_mitigation_methods(df, outcome_col, sensitive_col, flagged_features):
    from mitigation import apply_mitigation

    print("\n\n  === MITIGATION METHOD COMPARISON (RandomForest) ===")
    print(f"  {'Method':<15} {'Accuracy':>9}  {'DP Gap':>8}  {'TPR Gap':>9}  {'FPR Gap':>9}")
    print(f"  {'-'*58}")

    baseline = evaluate_models(df, outcome_col, sensitive_col, label='Baseline')
    b = baseline['RandomForest']
    print(f"  {'Baseline':<15} {b['accuracy']:>9.4f}  "
          f"{b['fairness'].get('demographic_parity_gap',0):>8.4f}  "
          f"{b['fairness'].get('equalized_odds_tpr_gap',0):>9.4f}  "
          f"{b['fairness'].get('equalized_odds_fpr_gap',0):>9.4f}")

    method_results = {'baseline': baseline}

    for method in ['splitting', 'reweighting', 'suppression']:
        mit    = apply_mitigation(df, flagged_features, outcome_col, method=method)
        df_mit = mit['df_mitigated']

        eval_sensitive = sensitive_col
        if sensitive_col not in df_mit.columns:
            alt = f'{sensitive_col}_frequency_bucket'
            eval_sensitive = alt if alt in df_mit.columns else outcome_col

        sw = mit['sample_weights'].get(flagged_features[0]) if method == 'reweighting' and flagged_features else None
        res = evaluate_models(df_mit, outcome_col, eval_sensitive, label=method, sample_weights=sw)
        r   = res['RandomForest']

        print(f"  {method:<15} {r['accuracy']:>9.4f}  "
              f"{r['fairness'].get('demographic_parity_gap',0):>8.4f}  "
              f"{r['fairness'].get('equalized_odds_tpr_gap',0):>9.4f}  "
              f"{r['fairness'].get('equalized_odds_fpr_gap',0):>9.4f}")

        method_results[method] = res

    best = min(
        ['splitting', 'reweighting', 'suppression'],
        key=lambda m: method_results[m]['RandomForest']['fairness'].get('demographic_parity_gap', 1)
    )
    print(f"\n  Best method for Demographic Parity: {best}")
    return method_results

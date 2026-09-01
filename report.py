"""
report.py -- Print and save pipeline reports
"""
import json
import numpy as np

def _fix(obj):
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, dict):          return {k: _fix(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_fix(i) for i in obj]
    return obj

def print_header(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

def print_rawness_report(r):
    print(f"\n  -- STEP 1: Rawness Score --")
    print(f"  Score: {r['rawness_score']}  ({r['interpretation']})")
    print(f"\n  Feature MI Scores:")
    for feat, mi in sorted(r['feature_mi'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {feat:<30} {mi:.4f}  {'#'*int(mi*200)}")

def print_legitimacy_report(results):
    print(f"\n  -- STEP 2: Legitimacy Tests --")
    for feat, r in results.items():
        status = "[OK] LEGITIMATE" if r['is_legitimate'] else "[!!] SUSPICIOUS"
        print(f"  {feat}: {status}")
        for reason in r.get('failure_reasons', []):
            print(f"       -> {reason}")

def print_threshold_report(tau, alpha, label):
    print(f"\n  -- STEP 3: Dynamic Threshold --")
    print(f"  tau = e^(-{alpha} x RS) capped at 0.45 = {tau}  [{label} sensitivity]")

def print_threshold_report_adaptive(tau, ceiling, percentile, alpha, label):
    print(f"\n  -- STEP 3/4: Adaptive Threshold (data-driven ceiling) --")
    print(f"  Ceiling = {percentile}th percentile of this dataset's ABS score distribution = {ceiling}")
    print(f"  tau = min(e^(-{alpha} x RS), {ceiling}) = {tau}  [{label} sensitivity]")

def print_abs_report(abs_results, tau):
    print(f"\n  -- STEP 4: Attribute Bias Scores --")
    print(f"  {'Feature':<30} {'ABS':>6}  {'NI':>6}  {'Dom':>6}  {'Card':>6}  {'Inst':>6}  Status")
    print(f"  {'-'*82}")
    for feat, res in abs_results.items():
        s      = res['signals']
        status = "[FLAG]" if res['abs_score'] > tau else "[OK]  "
        print(f"  {feat:<30} {res['abs_score']:>6.4f}  {s['neighborhood_inconsistency']:>6.4f}  "
              f"{s['dominance']:>6.4f}  {s['cardinality']:>6.4f}  {s['instability']:>6.4f}  {status}")

def print_flagging_report(flagged, legitimate):
    print(f"\n  -- STEP 5: Flagging Summary --")
    print(f"  Flagged  : {flagged if flagged else 'None'}")
    print(f"  Cleared  : {legitimate if legitimate else 'None'}")

def print_mitigation_report(log):
    print(f"\n  -- STEP 6: Mitigation --")
    if not log:
        print("  No mitigation needed.")
    for feat, action in log.items():
        print(f"  {feat}: {action}")

def print_model_comparison(comparison):
    print_header("MODEL EVALUATION: BEFORE vs AFTER MITIGATION")
    for name, comp in comparison.items():
        b, a = comp['before'], comp['after']
        print(f"\n  [{name}]")
        print(f"  {'Metric':<40} {'Before':>8}  {'After':>8}  {'Change':>8}")
        print(f"  {'-'*68}")
        rows = [
            ("Accuracy",               b['accuracy'],   a['accuracy'],   comp['accuracy_change']),
            ("Demographic Parity Gap", b['fairness'].get('demographic_parity_gap',0), a['fairness'].get('demographic_parity_gap',0), comp['dp_gap_change']),
            ("EO TPR Gap",             b['fairness'].get('equalized_odds_tpr_gap',0), a['fairness'].get('equalized_odds_tpr_gap',0), comp['tpr_gap_change']),
            ("EO FPR Gap",             b['fairness'].get('equalized_odds_fpr_gap',0), a['fairness'].get('equalized_odds_fpr_gap',0), comp['fpr_gap_change']),
        ]
        for label, bv, av, ch in rows:
            print(f"  {label:<40} {bv:>8.4f}  {av:>8.4f}  {ch:>+8.4f}")
        print(f"\n  Top features BEFORE : {[f for f,_ in b['top_features']]}")
        print(f"  Top features AFTER  : {[f for f,_ in a['top_features']]}")
        print(f"\n  Interpretation:")
        for line in comp['interpretations']:
            print(f"    -> {line}")

def print_pipeline_summary(results):
    print_header("PIPELINE COMPLETE - SUMMARY")
    print(f"  Dataset   : {results['dataset']}")
    print(f"  Rows      : {results['n_rows']}")
    print(f"  Rawness   : {results['rawness']['rawness_score']}")
    print(f"  Threshold : {results['threshold']}")
    print(f"  Legitimate: {results['legitimate_features']}")
    print(f"  Flagged   : {results['flagged_features']}")
    print(f"  Mitigation: {results['mitigation_method']}")
    print(f"  Completed : {results['timestamp']}")
    print(f"{'='*65}\n")

def save_json_report(results, path):
    with open(path, 'w') as f:
        json.dump(_fix(results), f, indent=2)
    print(f"  Full JSON report saved to: {path}")

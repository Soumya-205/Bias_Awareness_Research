"""
run.py -- Main entry point for the Bias-Aware AI Pipeline.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_from_url, load_csv, validate_dataset, preprocess
from mutual_information import compute_mi
from rawness import compute_rawness_score, interpret_rawness
from threshold import compute_threshold, compute_threshold_adaptive, get_sensitivity_label
from abs_score import compute_abs_all_features
from legitimacy import run_legitimacy_tests
from mitigation import apply_mitigation
from model_evaluation import evaluate_models, compare_results, compare_mitigation_methods
from report import (
    print_header, print_rawness_report, print_legitimacy_report,
    print_threshold_report, print_threshold_report_adaptive, print_abs_report, print_flagging_report,
    print_mitigation_report, print_model_comparison,
    print_pipeline_summary, save_json_report
)
from config import DEFAULT_ALPHA, MIN_MEANINGFUL_DISPARITY


def run_pipeline(df, csv_path, outcome_col, sensitive_col,
                 non_negotiable=None, alpha=DEFAULT_ALPHA,
                 mitigation_method='splitting', time_col=None,
                 run_mitigation_comparison=True):

    non_negotiable = non_negotiable or []
    results = {
        'dataset':          csv_path,
        'outcome_column':   outcome_col,
        'sensitive_column': sensitive_col,
        'n_rows':           len(df),
        'n_features':       len([c for c in df.columns if c != outcome_col]),
        'alpha':            alpha,
        'mitigation_method': mitigation_method,
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print_header("BIAS-AWARE AI PIPELINE")
    print(f"  Dataset  : {csv_path}")
    print(f"  Outcome  : {outcome_col}")
    print(f"  Sensitive: {sensitive_col}")
    print(f"  Features : {[c for c in df.columns if c != outcome_col]}")

    warnings = validate_dataset(df, outcome_col)
    if warnings:
        print("\n  [!] Warnings:")
        for w in warnings:
            print(f"      {w}")
    df = preprocess(df, outcome_col)

    features = [c for c in df.columns if c != outcome_col]
    if time_col and time_col in features:
        features.remove(time_col)

    rawness_results = compute_rawness_score(df[features + [outcome_col]], outcome_col)
    rawness_results['interpretation'] = interpret_rawness(rawness_results['rawness_score'])
    results['rawness'] = rawness_results
    print_rawness_report(rawness_results)
    rs        = rawness_results['rawness_score']
    mi_scores = rawness_results['feature_mi']

    avg_mi          = np.mean(list(mi_scores.values()))
    high_mi_features = [f for f, mi in mi_scores.items() if mi >= avg_mi]
    legitimacy_results  = {}
    legitimate_features = []
    illegitimate_features = []

    for feat in high_mi_features:
        leg = run_legitimacy_tests(feat, df, outcome_col)
        legitimacy_results[feat] = leg
        if leg['is_legitimate']:
            legitimate_features.append(feat)
        else:
            illegitimate_features.append(feat)

    results['legitimacy_tests']   = legitimacy_results
    results['legitimate_features'] = legitimate_features
    print_legitimacy_report(legitimacy_results)

    for feat in non_negotiable:
        if feat in illegitimate_features:
            print(f"\n  [!!] WARNING: '{feat}' failed legitimacy tests but is non-negotiable.")
            print(f"       MI Score: {mi_scores.get(feat, 'N/A')}")
            print(f"       Risk: may be a proxy or driving group disparity.")
            illegitimate_features.remove(feat)
            if feat not in legitimate_features:
                legitimate_features.append(feat)

    tau               = compute_threshold(rs, alpha)
    sensitivity_label = get_sensitivity_label(alpha)
    results['threshold'] = tau
    print_threshold_report(tau, alpha, sensitivity_label)

    abs_results = compute_abs_all_features(
        df, outcome_col, exclude=legitimate_features, time_col=time_col
    )
    results['abs_scores'] = abs_results
    print_abs_report(abs_results, tau)

    flagged_features     = [f for f, r in abs_results.items() if r['abs_score'] > tau]
    results['flagged_features'] = flagged_features
    print_flagging_report(flagged_features, legitimate_features)

    mitigation_output = apply_mitigation(df, flagged_features, outcome_col, method=mitigation_method)
    df_mitigated      = mitigation_output['df_mitigated']
    results['mitigation_log'] = mitigation_output['log']
    print_mitigation_report(mitigation_output['log'])

    print_header("MODEL TRAINING AND EVALUATION")

    eval_sensitive_after = sensitive_col
    if sensitive_col not in df_mitigated.columns:
        alt = f'{sensitive_col}_frequency_bucket'
        eval_sensitive_after = alt if alt in df_mitigated.columns else outcome_col

    sw = None
    if mitigation_method == 'reweighting' and flagged_features:
        sw = mitigation_output['sample_weights'].get(flagged_features[0])

    before_results = evaluate_models(df, outcome_col, sensitive_col, label='BEFORE mitigation')
    after_results  = evaluate_models(df_mitigated, outcome_col, eval_sensitive_after,
                                     label='AFTER mitigation', sample_weights=sw)

    comparison = compare_results(before_results, after_results, flagged_features)
    results['model_evaluation'] = {
        'before': before_results, 'after': after_results,
        'comparison': comparison, 'flagged_features': flagged_features
    }
    print_model_comparison(comparison)

    if run_mitigation_comparison and flagged_features:
        method_comparison = compare_mitigation_methods(
            df, outcome_col, sensitive_col, flagged_features
        )
        results['mitigation_comparison'] = {
            m: v['RandomForest'] for m, v in method_comparison.items()
        }

    print_pipeline_summary(results)
    report_path = csv_path.replace('.csv', '_full_report.json')
    save_json_report(results, report_path)

    try:
        from visualize import generate_all_charts
        prefix = csv_path.replace('.csv', '').replace('/', '_')
        generate_all_charts(results, prefix=prefix)
    except ImportError:
        print("  [!] matplotlib not installed -- skipping charts. Run: pip install matplotlib")

    return results


def run_pipeline_adaptive(df, csv_path, outcome_col, sensitive_col,
                          non_negotiable=None, alpha=DEFAULT_ALPHA,
                          mitigation_method='splitting', time_col=None,
                          run_mitigation_comparison=True, percentile=90):
    """
    Same 9-step pipeline as run_pipeline(), but with the FIXED adaptive
    threshold (see threshold.py -- compute_threshold_adaptive): the ceiling
    is derived from this dataset's own ABS score distribution instead of a
    hardcoded 0.45 shared across every dataset. ABS scores are computed
    before the threshold (order swapped internally vs. the original design,
    since the ceiling now depends on them), but reported in the same
    step order for readability.
    """
    non_negotiable = non_negotiable or []
    results = {
        'dataset':          csv_path,
        'outcome_column':   outcome_col,
        'sensitive_column': sensitive_col,
        'n_rows':           len(df),
        'n_features':       len([c for c in df.columns if c != outcome_col]),
        'alpha':            alpha,
        'threshold_method': 'adaptive_percentile',
        'percentile':       percentile,
        'mitigation_method': mitigation_method,
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print_header("BIAS-AWARE AI PIPELINE (ADAPTIVE THRESHOLD)")
    print(f"  Dataset  : {csv_path}")
    print(f"  Outcome  : {outcome_col}")
    print(f"  Sensitive: {sensitive_col}")
    print(f"  Features : {[c for c in df.columns if c != outcome_col]}")

    warnings = validate_dataset(df, outcome_col)
    if warnings:
        print("\n  [!] Warnings:")
        for w in warnings:
            print(f"      {w}")
    df = preprocess(df, outcome_col)

    features = [c for c in df.columns if c != outcome_col]
    if time_col and time_col in features:
        features.remove(time_col)

    rawness_results = compute_rawness_score(df[features + [outcome_col]], outcome_col)
    rawness_results['interpretation'] = interpret_rawness(rawness_results['rawness_score'])
    results['rawness'] = rawness_results
    print_rawness_report(rawness_results)
    rs        = rawness_results['rawness_score']
    mi_scores = rawness_results['feature_mi']

    avg_mi          = np.mean(list(mi_scores.values()))
    high_mi_features = [f for f, mi in mi_scores.items() if mi >= avg_mi]
    legitimacy_results  = {}
    legitimate_features = []
    illegitimate_features = []

    for feat in high_mi_features:
        leg = run_legitimacy_tests(feat, df, outcome_col)
        legitimacy_results[feat] = leg
        if leg['is_legitimate']:
            legitimate_features.append(feat)
        else:
            illegitimate_features.append(feat)

    results['legitimacy_tests']   = legitimacy_results
    results['legitimate_features'] = legitimate_features
    print_legitimacy_report(legitimacy_results)

    for feat in non_negotiable:
        if feat in illegitimate_features:
            print(f"\n  [!!] WARNING: '{feat}' failed legitimacy tests but is non-negotiable.")
            print(f"       MI Score: {mi_scores.get(feat, 'N/A')}")
            print(f"       Risk: may be a proxy or driving group disparity.")
            illegitimate_features.remove(feat)
            if feat not in legitimate_features:
                legitimate_features.append(feat)

    # STEP 4 computed BEFORE the threshold, since the ceiling now depends on it
    abs_results = compute_abs_all_features(
        df, outcome_col, exclude=legitimate_features, time_col=time_col
    )
    results['abs_scores'] = abs_results

    tau, ceiling = compute_threshold_adaptive(rs, abs_results, alpha=alpha, percentile=percentile)
    sensitivity_label = get_sensitivity_label(alpha)
    results['threshold'] = tau
    results['threshold_ceiling'] = ceiling
    print_threshold_report_adaptive(tau, ceiling, percentile, alpha, sensitivity_label)
    print_abs_report(abs_results, tau)

    flagged_via_abs = [f for f, r in abs_results.items() if r['abs_score'] > tau]

    # FIX (Sep 1 2026): auto-flag any tested feature that failed the Outcome
    # Independence test specifically -- that test directly measures "does
    # this feature increase real group disparity," and was being computed
    # but silently discarded by the old design, which only ever looked at
    # the ABS score to decide what gets flagged. Found on COMPAS: score_factor
    # and priors_count both failed this test (real, literature-consistent
    # disparity-driving features) but never crossed the ABS threshold and
    # were never flagged. Respects non_negotiable overrides -- a feature a
    # human has explicitly marked as required is not force-flagged here.
    flagged_via_disparity = [
        feat for feat, leg in legitimacy_results.items()
        if not leg['outcome_independence']['passed']
        and leg['outcome_independence']['score'] > MIN_MEANINGFUL_DISPARITY
        and feat not in legitimate_features
    ]

    flagged_features = sorted(set(flagged_via_abs) | set(flagged_via_disparity))
    results['flagged_features'] = flagged_features
    results['flagged_via_abs'] = flagged_via_abs
    results['flagged_via_disparity'] = flagged_via_disparity
    print_flagging_report(flagged_features, legitimate_features)
    if flagged_via_disparity:
        print(f"  (of which, flagged for driving real group disparity, not ABS score: {flagged_via_disparity})")

    mitigation_output = apply_mitigation(df, flagged_features, outcome_col, method=mitigation_method)
    df_mitigated      = mitigation_output['df_mitigated']
    results['mitigation_log'] = mitigation_output['log']
    print_mitigation_report(mitigation_output['log'])

    print_header("MODEL TRAINING AND EVALUATION")

    eval_sensitive_after = sensitive_col
    if sensitive_col not in df_mitigated.columns:
        alt = f'{sensitive_col}_frequency_bucket'
        eval_sensitive_after = alt if alt in df_mitigated.columns else outcome_col

    sw = None
    if mitigation_method == 'reweighting' and flagged_features:
        sw = mitigation_output['sample_weights'].get(flagged_features[0])

    before_results = evaluate_models(df, outcome_col, sensitive_col, label='BEFORE mitigation')
    after_results  = evaluate_models(df_mitigated, outcome_col, eval_sensitive_after,
                                     label='AFTER mitigation', sample_weights=sw)

    comparison = compare_results(before_results, after_results, flagged_features)
    results['model_evaluation'] = {
        'before': before_results, 'after': after_results,
        'comparison': comparison, 'flagged_features': flagged_features
    }
    print_model_comparison(comparison)

    if run_mitigation_comparison and flagged_features:
        method_comparison = compare_mitigation_methods(
            df, outcome_col, sensitive_col, flagged_features
        )
        results['mitigation_comparison'] = {
            m: v['RandomForest'] for m, v in method_comparison.items()
        }

    print_pipeline_summary(results)
    report_path = csv_path.replace('.csv', '_adaptive_full_report.json')
    save_json_report(results, report_path)

    try:
        from visualize import generate_all_charts
        prefix = csv_path.replace('.csv', '').replace('/', '_') + '_adaptive'
        generate_all_charts(results, prefix=prefix)
    except ImportError:
        print("  [!] matplotlib not installed -- skipping charts. Run: pip install matplotlib")

    return results

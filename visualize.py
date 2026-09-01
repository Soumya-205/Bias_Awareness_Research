"""
visualize.py -- Generate charts from pipeline results
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = 'charts'


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def chart_mi_scores(results, prefix='dataset'):
    _ensure_dir()
    mi     = results['rawness']['feature_mi']
    feats  = list(mi.keys())
    scores = list(mi.values())

    sorted_pairs = sorted(zip(scores, feats), reverse=True)
    scores, feats = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ['#e74c3c' if s > np.mean(scores) else '#3498db' for s in scores]
    bars    = ax.barh(feats, scores, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Mutual Information Score', fontsize=12)
    ax.set_title('Feature Decision Power (Mutual Information with Outcome)', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(scores), color='gray', linestyle='--', linewidth=1, label='Average MI')

    high_patch = mpatches.Patch(color='#e74c3c', label='Above average (tested for legitimacy)')
    low_patch  = mpatches.Patch(color='#3498db', label='Below average')
    ax.legend(handles=[high_patch, low_patch], loc='lower right')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/{prefix}_mi_scores.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def chart_abs_scores(results, prefix='dataset'):
    _ensure_dir()
    abs_data  = results['abs_scores']
    tau       = results['threshold']
    flagged   = results['flagged_features']

    feats  = list(abs_data.keys())
    scores = [abs_data[f]['abs_score'] for f in feats]

    sorted_pairs = sorted(zip(scores, feats), reverse=True)
    scores, feats = zip(*sorted_pairs)

    colors = ['#e74c3c' if f in flagged else '#2ecc71' for f in feats]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(feats, scores, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(tau, color='black', linestyle='--', linewidth=2, label=f'Threshold (tau={tau})')

    ax.set_xlabel('Attribute Bias Score', fontsize=12)
    ax.set_title('Attribute Bias Scores -- Features Above Threshold are Flagged', fontsize=14, fontweight='bold')

    flag_patch  = mpatches.Patch(color='#e74c3c', label='Flagged (ABS > tau)')
    clear_patch = mpatches.Patch(color='#2ecc71', label='Cleared (ABS <= tau)')
    ax.legend(handles=[flag_patch, clear_patch, plt.Line2D([0],[0], color='black', linestyle='--', label=f'tau={tau}')],
              loc='lower right')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/{prefix}_abs_scores.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def chart_before_after(results, prefix='dataset'):
    _ensure_dir()
    comparison = results['model_evaluation']['comparison']
    metrics    = ['accuracy', 'demographic_parity_gap', 'equalized_odds_tpr_gap', 'equalized_odds_fpr_gap']
    labels     = ['Accuracy', 'DP Gap', 'EO TPR Gap', 'EO FPR Gap']
    models     = ['RandomForest', 'LogisticRegression']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Before vs After Mitigation -- Model Evaluation', fontsize=15, fontweight='bold')

    for idx, model_name in enumerate(models):
        ax   = axes[idx]
        comp = comparison[model_name]
        b    = comp['before']
        a    = comp['after']

        before_vals = [
            b['accuracy'],
            b['fairness'].get('demographic_parity_gap', 0),
            b['fairness'].get('equalized_odds_tpr_gap', 0),
            b['fairness'].get('equalized_odds_fpr_gap', 0)
        ]
        after_vals = [
            a['accuracy'],
            a['fairness'].get('demographic_parity_gap', 0),
            a['fairness'].get('equalized_odds_tpr_gap', 0),
            a['fairness'].get('equalized_odds_fpr_gap', 0)
        ]

        x     = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, before_vals, width, label='Before', color='#e67e22', alpha=0.85)
        ax.bar(x + width/2, after_vals,  width, label='After',  color='#27ae60', alpha=0.85)

        ax.set_title(model_name, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Score', fontsize=11)
        ax.legend()
        ax.set_ylim(0, 1.0)

        for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
            change = av - bv
            color  = '#27ae60' if change < 0 else '#e74c3c' if change > 0 else 'gray'
            if i == 0:
                color = '#27ae60' if change >= 0 else '#e74c3c'
            ax.annotate(f'{change:+.3f}', xy=(i + width/2, av + 0.01),
                        ha='center', fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/{prefix}_before_after.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def chart_mitigation_comparison(results, prefix='dataset'):
    _ensure_dir()
    if 'mitigation_comparison' not in results:
        print("  Skipping mitigation comparison chart -- no comparison data.")
        return

    comp    = results['mitigation_comparison']
    methods = [m for m in ['baseline', 'splitting', 'reweighting', 'suppression'] if m in comp]
    metrics = {
        'Accuracy':       lambda r: r['accuracy'],
        'DP Gap':         lambda r: r['fairness'].get('demographic_parity_gap', 0),
        'EO TPR Gap':     lambda r: r['fairness'].get('equalized_odds_tpr_gap', 0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Mitigation Method Comparison (RandomForest)', fontsize=14, fontweight='bold')
    colors = ['#95a5a6', '#3498db', '#e67e22', '#9b59b6']

    for idx, (metric_name, metric_fn) in enumerate(metrics.items()):
        ax     = axes[idx]
        values = [metric_fn(comp[m]) for m in methods]
        bars   = ax.bar(methods, values, color=colors[:len(methods)], edgecolor='white')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1.0)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/{prefix}_mitigation_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def generate_all_charts(results, prefix='dataset'):
    print(f"\n  Generating charts in '{OUTPUT_DIR}/' folder...")
    try:
        chart_mi_scores(results, prefix)
        chart_abs_scores(results, prefix)
        chart_before_after(results, prefix)
        chart_mitigation_comparison(results, prefix)
        print(f"  All charts saved to: {OUTPUT_DIR}/")
    except Exception as e:
        print(f"  [!] Chart generation failed: {e}")
        print(f"      Install matplotlib: pip install matplotlib")

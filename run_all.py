"""
run_all.py -- Runs all four datasets through the FIXED, adaptive-threshold
pipeline, one after another. Just run:

    python run_all.py

Charts and full JSON reports land in charts/ and the working directory,
same as any single run.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import run_pipeline_adaptive
from data_loader import load_csv


def divider(title):
    print("\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)


# 1. German Credit -- finance, original flagship dataset
divider("GERMAN CREDIT (finance)")
df = load_csv('german_credit.csv')
run_pipeline_adaptive(
    df=df, csv_path='german_credit.csv', outcome_col='credit_risk',
    sensitive_col='personal_status', non_negotiable=['credit_amount', 'duration'],
    alpha=1.0, mitigation_method='splitting', run_mitigation_comparison=True, percentile=90
)

# 2. Heart Disease -- healthcare, cross-domain validation
divider("HEART DISEASE (healthcare)")
df = load_csv('heart_disease.csv')
run_pipeline_adaptive(
    df=df, csv_path='heart_disease.csv', outcome_col='target',
    sensitive_col='sex', non_negotiable=[],
    alpha=1.0, mitigation_method='splitting', run_mitigation_comparison=True, percentile=90
)

# 3. COMPAS, one-hot race encoding -- criminal justice, the famous ProPublica benchmark
divider("COMPAS -- one-hot race (propublica_data_for_fairml.csv)")
df = load_csv('propublica_data_for_fairml.csv')
run_pipeline_adaptive(
    df=df, csv_path='propublica_data_for_fairml.csv', outcome_col='Two_yr_Recidivism',
    sensitive_col='African_American', non_negotiable=[],
    alpha=1.0, mitigation_method='splitting', run_mitigation_comparison=True, percentile=90
)

# 4. COMPAS, single categorical race column -- same population, different encoding
divider("COMPAS -- single categorical race (compas_clean.csv)")
df = load_csv('compas_clean.csv')
run_pipeline_adaptive(
    df=df, csv_path='compas_clean.csv', outcome_col='two_year_recid',
    sensitive_col='race', non_negotiable=[],
    alpha=1.0, mitigation_method='splitting', run_mitigation_comparison=True, percentile=90
)

print("\n\nAll four datasets complete. Charts in charts/, full reports as *_adaptive_full_report.json")

"""
prep_compas.py -- Clean the ORIGINAL (non-one-hot) COMPAS file, following
ProPublica's own published filtering criteria, so results are comparable
to the literature. Keeps race as a single multi-category column (Caucasian,
African-American, Hispanic, etc.) instead of one-hot encoding it.
"""
import pandas as pd

df = pd.read_csv('compas_raw.csv')
print(f"Raw: {df.shape}")

# ProPublica's own documented filtering
df = df[
    (df['days_b_screening_arrest'] <= 30) &
    (df['days_b_screening_arrest'] >= -30) &
    (df['is_recid'] != -1) &
    (df['c_charge_degree'] != 'O') &
    (df['score_text'] != 'N/A')
].reset_index(drop=True)
print(f"After ProPublica filtering: {df.shape}")

keep_cols = [
    'sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count',
    'juv_other_count', 'priors_count', 'c_charge_degree', 'two_year_recid'
]
df = df[keep_cols].dropna().reset_index(drop=True)
print(f"Final: {df.shape}")
print(df['race'].value_counts())
print(df.groupby('race')['two_year_recid'].mean())

df.to_csv('compas_clean.csv', index=False)
print("Saved: compas_clean.csv")

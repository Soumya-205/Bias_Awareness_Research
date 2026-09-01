# Bias-Aware AI Pipeline -- Fixed + Extended (Sep 1 2026)

This is the pipeline with four real fixes found and verified by testing it
on 4 datasets across 3 domains (finance, healthcare, criminal justice).
Original architecture and authorship: Tanishk Gangwar / team.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
python run_all.py
```

This runs all 4 datasets end to end and writes charts/ (16 PNGs) and
4 *_adaptive_full_report.json files. Takes 1-2 minutes.

To run one dataset at a time, see the pattern in run_all.py -- each block
calls `run_pipeline_adaptive()` from run.py with that dataset's
outcome/sensitive columns.

## What's fixed, in order found

### 1. Pandas 3.0 compatibility (data_loader.py, legitimacy.py, mitigation.py,
   model_evaluation.py, mutual_information.py, abs_score.py)
`dtype == object` was used everywhere to detect text/categorical columns.
Pandas 3.0 (now the default `pip install pandas`) reads plain strings as a
new `str` dtype, not `object` -- every check silently failed, so
categorical columns never got encoded and crashed straight into sklearn.
This broke German Credit (the flagship demo) on any current install.
Fixed: replaced with `not pd.api.types.is_numeric_dtype(col)`, which is
correct regardless of pandas version.

### 2. Neighborhood Inconsistency signal (abs_score.py)
The signal with the highest ABS weight (40%) used sklearn's
NearestNeighbors to find each point's "nearest neighbors" and measure
outcome variance among them. For any binary/low-cardinality feature,
hundreds of points tie at distance 0, and sklearn's tree traversal
returns the SAME fixed subset of neighbors for every query sharing that
value -- silently zeroing the signal regardless of the feature's true
variance. Verified: `sex` in the Heart Disease dataset has real
within-group outcome std of 0.44-0.50 (near max possible), not 0.
Fixed: for low-cardinality features, compute the real within-group
outcome std directly instead of relying on k-NN.

### 3. Fixed threshold ceiling -> adaptive, data-driven ceiling (threshold.py, run.py)
The original tau = min(e^(-alpha*RS), 0.45) used ONE constant (0.45)
shared across every dataset. At default alpha=1.0 this ceiling almost
never lifts (needs RS > 0.799) -- German Credit (RS=0.47) and Heart
Disease (RS=0.74) both hit it every time. Real bias scores landed within
hundredths of this arbitrary number in both directions.
Fixed: `compute_threshold_adaptive()` derives the ceiling from THAT
dataset's own ABS score distribution (90th percentile by default) instead
of a hardcoded constant, while keeping the original Rawness-Score-driven
sensitivity mechanism intact. Old function kept in threshold.py for
comparison. New entry point: `run_pipeline_adaptive()` in run.py (original
`run_pipeline()` untouched, still works).

### 4. Outcome-Independence-driven auto-flagging (run.py, Step 5)
Legitimacy testing (Step 2) already computes whether a feature "drives
group disparity" (the Outcome Independence test) -- but the old Step 5
flagging decision ONLY looked at ABS score, discarding this diagnosis.
On COMPAS this meant `score_factor` (COMPAS's own risk score -- the exact
feature ProPublica's 2016 investigation centered on) and `priors_count`
(the classic over-policing proxy) both correctly failed Outcome
Independence but never got flagged, because their ABS scores stayed
under threshold.
Fixed: any tested feature that fails Outcome Independence specifically is
now flagged regardless of ABS score (respects non_negotiable overrides).
Scope note: this only catches features that were escalated to legitimacy
testing in the first place (above-average MI) -- it does not by itself
fix low-MI sensitive attributes like one-hot `African_American`.

## A structural finding, not yet fixed: Dominance signal favors rarity

Confirmed on COMPAS, two ways:
- One-hot encoding `race` into 5 binary columns caused `African_American`
  (51% of the dataset, the actual documented bias axis) to score LOWEST
  of 11 features (ABS=0.21), while `Native_American` (11 people, 0.2% of
  the dataset) got flagged -- purely because a rare one-hot flag is
  mechanically "dominant" in its zero-value, regardless of real bias.
- Re-running with `race` as a single categorical column raised its score
  56% (0.21 -> 0.32) but it still didn't cross threshold -- `juv_fel_count`
  (also a rare-value feature, 96% dominance) got flagged instead.
This means the Dominance signal (30% of ABS weight) has a built-in bias
toward flagging whatever is statistically rarest, independent of real
disparity. Not yet fixed -- candidate approaches: a minimum sample-size
floor before Dominance counts for a category, or reworking Dominance to
measure outcome disparity by category rather than marginal rarity.

## Cross-dataset results summary

| Dataset | Domain | Flagged (after all fixes) | Best mitigation method |
|---|---|---|---|
| German Credit | Finance | foreign_worker, other_debtors | Splitting |
| Heart Disease | Healthcare | fbs, ca | Suppression |
| COMPAS (one-hot) | Criminal justice | Native_American, score_factor | -- |
| COMPAS (categorical) | Criminal justice | juv_fel_count, priors_count | Reweighting |

No mitigation method wins on more than one dataset -- strong evidence
that mitigation strategy is dataset-dependent, not universal, which is
consistent with (and now backed by 3 domains, not 1) the project's
original 3-way mitigation comparison design.

## Files

- Fixed: abs_score.py, config.py (unchanged, kept for reference),
  data_loader.py, legitimacy.py, mitigation.py, model_evaluation.py,
  mutual_information.py, threshold.py, report.py, run.py
- Unchanged: rawness.py, visualize.py
- Data: german_credit.csv, heart_disease.csv,
  propublica_data_for_fairml.csv (one-hot COMPAS),
  compas_clean.csv (categorical COMPAS, generated by prep_compas.py)
- prep_compas.py: cleans the raw ProPublica COMPAS file using their own
  documented filtering criteria
- run_all.py: runs all 4 datasets in one command

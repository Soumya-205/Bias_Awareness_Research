"""
config.py -- All tunable parameters for the bias-aware pipeline.
Change values here
"""
# Mutual Information
KSG_K                     = 3
BINNING_FALLBACK_BINS     = 10
LOW_CARDINALITY_THRESHOLD = 10
# Rawness Score
RAWNESS_TAU_CEILING = 0.45
# Threshold Mechanism
DEFAULT_ALPHA = 1.0
ALPHA_OPTIONS = {
    0.5: "Low       -- minimal flags, trust your data",
    1.0: "Medium    -- default, good starting point",
    2.0: "High      -- stricter detection",
    3.0: "Very High -- maximum sensitivity, expect false positives"
}
# ABS Signal Weights (must sum to 1.0)
ABS_WEIGHTS = {
    'neighborhood_inconsistency': 0.40,
    'dominance':                  0.30,
    'cardinality':                0.20,
    'instability':                0.10,
}
# Legitimacy Test Thresholds
PROXY_DETECTION_THRESHOLD       = 0.5
CONDITIONAL_INFLUENCE_THRESHOLD = 0.02
KNN_NEIGHBORS                   = 5


MIN_MEANINGFUL_DISPARITY = 0.02
# Model Training
TEST_SIZE       = 0.3
RANDOM_STATE    = 42
RF_N_ESTIMATORS = 100
LR_MAX_ITER     = 3000
# Fairness
FAIRNESS_ACCURACY_TOLERANCE = 0.02
# Output
OUTPUT_DIR         = "outputs"
BIAS_REPORT_SUFFIX = "_bias_report.json"
EVAL_REPORT_SUFFIX = "_eval_report.json"

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
# FIX: Use imblearn's Pipeline to handle samplers like RandomOverSampler
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# =========================
# Configuration
# =========================
DATA_FILE = 'unified_exoplanet_catalog_v2.csv'
N_SPLITS = 5  # Standard for K-Fold CV
RANDOM_SEED = 42

# =========================
# Load and Preprocess Unified Dataset
# =========================
print(f"Loading data from {DATA_FILE} for Cross-Validation...")
df = pd.read_csv(DATA_FILE) 

# Map unified string labels to binary:
label_col = 'disposition'
df[label_col] = df[label_col].replace({
    'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0,
    'APC': 0, 'FA': 0, 'REFUTED': 0
})

# Drop rows without a clear label and ensure label is integer type
df = df.dropna(subset=[label_col])
df[label_col] = df[label_col].astype(int)

# =========================
# Feature Selection
# =========================
exclude_cols = [
    label_col, 
    'source_id', 'candidate_id', 'pl_name', 'mission_source',
    'k_fpflag_ss', 'k_fpflag_co', 'k_fpflag_ec', 'fp_flag_nt', 
    'disposition_score',
    # Exclude all unique string/categorical columns
    'k_vet_stat', 'k_comment', 'k_fittype', 'k_tce_delivname', 'k_datalink_dvr',
    't_toipfx', 't_ctoi_alias', 't_toi_created', 't_rowupdate',
    'k2_hostname', 'k2_id', 'k2_tic_id', 'k2_gaia_id', 'a_discoverymethod',
    'a_disc_year', 'a_disc_facility', 'a_rv_flag', 'a_tran_flag', 'a_ima_flag', 'a_st_spectype'
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric_cols if c not in exclude_cols]

X = df[features]
y = df[label_col]

print(f"Features selected for training ({len(features)} total features).")
print(f"Target distribution:\n{y.value_counts()}")


# =========================
# LightGBM and Oversampling Pipeline Setup
# =========================

# LightGBM Parameters
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'verbose': -1,
    'use_missing': True,
    'zero_as_missing': False,
    'seed': RANDOM_SEED,
}

# 1. Define the Pipeline
# Step 1: Oversampler (applied ONLY to the training fold)
# Step 2: LightGBM Classifier
pipeline = Pipeline([
    ('oversampler', RandomOverSampler(random_state=RANDOM_SEED)),
    ('classifier', lgb.LGBMClassifier(**lgbm_params))
])

# 2. Define the K-Fold strategy
cv_strategy = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# 3. Define the scoring metric
# FINAL FIX: Explicitly setting the function to use 'predict_proba' (if 'needs_proba' fails)
# The `roc_auc_score` function, when used via `make_scorer`, needs to know it should call 
# predict_proba on the estimator. The modern way to do this is with `make_scorer(roc_auc_score, response_method='predict_proba')`.
auc_scorer = make_scorer(roc_auc_score, response_method='predict_proba')

print(f"\n--- Running {N_SPLITS}-Fold Stratified Cross-Validation with Oversampling Pipeline ---")

# 4. Perform Cross-Validation
# The imblearn Pipeline ensures that the oversampling only happens on the training split of each fold.
auc_scores = cross_val_score(
    estimator=pipeline,
    X=X,
    y=y,
    cv=cv_strategy,
    scoring=auc_scorer,
    n_jobs=-1,  # Use all available cores
    verbose=0
)


# =========================
# Display Results
# =========================
print("\n--- Cross-Validation Results ---")
print("Individual Fold ROC-AUC Scores (N=5):")
for i, score in enumerate(auc_scores):
    print(f"  Fold {i+1}: {score:.4f}")

mean_auc = auc_scores.mean()
std_auc = auc_scores.std()

print("-" * 30)
print(f"Average ROC-AUC Score: {mean_auc:.4f}")
print(f"Standard Deviation:    {std_auc:.4f}")
print("-" * 30)

print("\nExpected: The mean AUC should now be much closer to your initial 0.93 score, confirming that performance is stable and high when using the oversampling technique.")

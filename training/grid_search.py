import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
# FIX: Use imblearn's Pipeline to handle samplers like RandomOverSampler
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# =========================
# Configuration
# =========================
DATA_FILE = 'unified_exoplanet_catalog_v2.csv'
# Using 3 splits for tuning speed. Increase to 5 for final, highly precise search if needed.
N_SPLITS_TUNING = 3  
RANDOM_SEED = 42

# Define the scorer to explicitly request probability outputs
# Using response_method='predict_proba' ensures we get the high AUC scores.
AUC_SCORER = make_scorer(roc_auc_score, response_method='predict_proba') 

# =========================
# Load and Preprocess Unified Dataset
# =========================
print(f"Loading data from {DATA_FILE} for Hyperparameter Tuning...")
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
    # Exclude all unique string/categorical columns (identifiers, etc.)
    'k_vet_stat', 'k_comment', 'k_fittype', 'k_tce_delivname', 'k_datalink_dvr',
    't_toipfx', 't_ctoi_alias', 't_toi_created', 't_rowupdate',
    'k2_hostname', 'k2_id', 'k2_tic_id', 'k2_gaia_id', 'a_discoverymethod',
    'a_disc_year', 'a_disc_facility', 'a_rv_flag', 'a_tran_flag', 'a_ima_flag', 'a_st_spectype'
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric_cols if c not in exclude_cols]

X = df[features]
y = df[label_col]

print(f"Features selected for tuning ({len(features)} total features).")
print(f"Target distribution:\n{y.value_counts()}")


# =========================
# LightGBM and Oversampling Pipeline Setup
# =========================

# Base LightGBM Parameters (parameters that will NOT be tuned)
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'use_missing': True,
    'zero_as_missing': False,
    'seed': RANDOM_SEED,
}

# 1. Define the Pipeline
# RandomOverSampler ensures balanced training data within each CV fold.
pipeline = Pipeline([
    ('oversampler', RandomOverSampler(random_state=RANDOM_SEED)),
    ('classifier', lgb.LGBMClassifier(**lgbm_params))
])

# 2. Define the parameter grid for Grid Search
param_grid = {
    # We prefix parameters with 'classifier__' because LGBM is the second step in the pipeline.
    # Lower learning rate combined with higher estimators is usually better.
    'classifier__learning_rate': [0.01, 0.03, 0.05],
    'classifier__n_estimators': [300, 500, 800],
    # Controls the complexity of the trees
    'classifier__num_leaves': [20, 31, 40],
    # Controls depth to prevent overfitting
    'classifier__max_depth': [-1, 10] 
}

# 3. Define the K-Fold strategy for the internal search
cv_strategy = StratifiedKFold(n_splits=N_SPLITS_TUNING, shuffle=True, random_state=RANDOM_SEED)

# 4. Initialize Grid Search
print(f"\n--- Starting Grid Search ({N_SPLITS_TUNING}-Fold CV) ---")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=AUC_SCORER,
    cv=cv_strategy,
    verbose=2,
    n_jobs=-1, # Use all available cores for parallel processing
    refit=True # Retrain the final model with the best params on ALL data
)

# 5. Execute Grid Search
grid_search.fit(X, y)

# =========================
# Display Results
# =========================
print("\n--- Hyperparameter Tuning Results ---")
print(f"Best ROC-AUC Score found: {grid_search.best_score_:.4f}")
print("Best Parameters found:")
# Clean the parameter names for better display
best_params_display = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
print(best_params_display)

print("\n--- Final Model Training Complete ---")
print("The 'grid_search.best_estimator_' now holds the best performing model (a Pipeline).")

# =========================
# Save Results and Model
# =========================
MODEL_FILE = 'exoplanet_unified_lgbm_best.txt'

# Save the best model (which is the refitted pipeline's classifier step)
grid_search.best_estimator_.named_steps['classifier'].save_model(MODEL_FILE)
print(f"\nBest model saved as {MODEL_FILE} using the optimal parameters found.")

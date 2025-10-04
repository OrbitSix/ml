import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load Unified Dataset
# =========================
# *** CHANGE 1: Update file name to match the new unified output ***
DATA_FILE = 'unified_exoplanet_catalog_v2.csv'
print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE) 

# =========================
# Choose Unified Label Column and Map
# =========================
label_col = 'disposition'

# Map unified string labels to binary:
# CONFIRMED/CANDIDATE = 1 (Positive class: likely a planet)
# FALSE POSITIVE = 0 (Negative class: not a planet)
df[label_col] = df[label_col].replace({
    'CONFIRMED': 1, 
    'CANDIDATE': 1,
    'FALSE POSITIVE': 0,
    # These mappings handle any strings that were missed during unification or from TESS (APC, FA)
    'APC': 0, 
    'FA': 0,
    'REFUTED': 0
})

# Drop rows without a clear label and ensure label is integer type
df = df.dropna(subset=[label_col])
df[label_col] = df[label_col].astype(int)

# =========================
# Feature Selection
# =========================
# List of columns to explicitly exclude from the feature set.
exclude_cols = [
    label_col, 
    'source_id',        # Identifier
    'candidate_id',     # Identifier
    'pl_name',          # Identifier/String
    'mission_source',   # Categorical String (could be encoded, but dropped)
    
    # Kepler Flags (already included in numerical columns, but excluded here for being binary flags)
    'k_fpflag_ss', 'k_fpflag_co', 'k_fpflag_ec', 'fp_flag_nt', 
    'disposition_score',
    
    # Exclude all unique string/categorical columns defined in the unification step
    # Kepler Categorical
    'k_vet_stat', 'k_comment', 'k_fittype', 'k_tce_delivname', 'k_datalink_dvr',
    # TESS Categorical
    't_toipfx', 't_ctoi_alias', 't_toi_created', 't_rowupdate',
    
    # *** CHANGE 2: Add K2 Categorical/String Exclusions ***
    'k2_hostname', 'k2_id', 'k2_tic_id', 'k2_gaia_id', 'a_discoverymethod',
    'a_disc_year', 'a_disc_facility', 'a_rv_flag', 'a_tran_flag', 'a_ima_flag', 'a_st_spectype'
]

# Select all numerical columns. This is key:
# It automatically includes all core features, all error terms, the sentinel values, 
# AND the critical 'is_..._missing' indicator flags (for all three missions).
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

features = [c for c in numeric_cols if c not in exclude_cols]

X = df[features]
y = df[label_col]

# We rely on the sentinel values (-9999.0 / -1) already present from the unification script.
# Any remaining NaNs will be left for LGBM's internal NaN handler.

print(f"Features selected for training ({len(features)} total features).")
print(f"Sample features: {features[:5]}...{features[-5:]}")
print(f"Target distribution:\n{y.value_counts()}")


# =========================
# Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# LightGBM Dataset and Oversampling
# =========================
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

lgb_train = lgb.Dataset(X_train_res, label=y_train_res)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# =========================
# LightGBM Parameters
# =========================
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,        # slower → better generalization
    'num_leaves': 40,
    'verbose': -1,
    # Crucial for cross-mission data: allow LGBM to use NaNs as a distinct category
    'use_missing': True,
    'zero_as_missing': False, # Treat 0s as 0s, not NaNs
}

# =========================
# Train Model
# =========================
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    num_boost_round=1000,
    # early_stopping_rounds=50,
    # verbose_eval=100
)

# =========================
# Evaluate
# =========================
y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_prob > 0.5).astype(int)

report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\n--- Evaluation on Unified Dataset (Kepler + TESS + K2) ---")
print("Classification Report:\n", report)
print("ROC-AUC:", roc_auc)

# =========================
# Save Results and Model
# =========================
REPORT_FILE = 'classification_report_unified_lgbm.txt'
MODEL_FILE = 'exoplanet_unified_lgbm.txt'

with open(REPORT_FILE, 'w') as f:
    f.write(report)
    f.write(f"\nROC-AUC: {roc_auc}\n")

print(f"\nClassification report saved to {REPORT_FILE}")

model.save_model(MODEL_FILE)
print(f"Model saved as {MODEL_FILE}")

# =========================
# Feature Importance Plot
# =========================
print("Displaying Feature Importance...")
lgb.plot_importance(model, max_num_features=20, importance_type='gain', figsize=(10,7))
plt.title("Top 20 Feature Importances (Unified Kepler + TESS + K2)")
plt.tight_layout()
plt.show()

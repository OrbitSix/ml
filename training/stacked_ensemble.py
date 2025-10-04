import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
# --- Required Imports ---
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score 
# ------------------------
from sklearn.linear_model import LogisticRegression # Kept for potential comparison, but not used as meta-learner
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import numpy as np
import logging

# Set logging level for better visibility during run
logging.basicConfig(level=logging.INFO)

# =========================
# Configuration & File Paths
# =========================
KEPLER_FILE = '../datasets/keplar.csv'  # Original Kepler Candidate File
TESS_FILE = '../datasets/tess.csv'      # Original TESS Candidate File
K2_FILE = '../datasets/k2.csv'             # Original K2 Archive/Catalog File

RANDOM_SEED = 42
N_SPLITS = 5

# --- Tuned parameters for better generalization ---
TUNED_PARAMS = {
    'feature_fraction': 0.85,    # Subsample of features per tree
    'bagging_fraction': 0.80,    # Subsample of rows per tree
    'bagging_freq': 1,           # Perform bagging every iteration
    'min_child_samples': 20,     # Min data in one leaf
}

# --- UPDATED PARAMETERS FOR BASE MODELS (Kepler & K2) ---
LGBM_BASE_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'n_estimators': 500,
    'num_leaves': 40,
    'verbose': -1,
    'seed': RANDOM_SEED,
    'n_jobs': -1 ,
    **TUNED_PARAMS 
}

# --- NEW PARAMETERS FOR TESS BASE MODEL (Option A: More Capacity, Slower Learning) ---
# This attempts to overcome the noise/sparsity of TESS data by using deeper trees and more steps.
LGBM_TESS_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.015,  # Slower learning
    'n_estimators': 1000,    # More estimators (rounds)
    'num_leaves': 61,       # Higher capacity
    'verbose': -1,
    'seed': RANDOM_SEED,
    'n_jobs': -1 ,
    **TUNED_PARAMS 
}

# --- NEW PARAMETERS FOR LIGHTGBM META-LEARNER (Option C: Simple, Non-linear Stack) ---
# This model uses the 3 probability scores as features and should be quick to train.
LGBM_META_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 200,    # Increased from 100
    'num_leaves': 16,       # Increased from 8
    'verbose': -1,
    'seed': RANDOM_SEED,
    'n_jobs': -1,
    **TUNED_PARAMS 
}

# =========================
# 1. Feature Mapping Definitions for Each Mission (Leakage-Proof)
#    (Functions remain unchanged for feature consistency)
# =========================

def preprocess_kepler(df):
    """Selects and maps features unique to the Kepler Candidate dataset, strictly removing leakage."""
    logging.info("Preprocessing Kepler data: identifying leakage columns.")

    # 1. Rename core columns (if needed for consistency, though we use all numeric remaining cols)
    KEPLER_MAPPING = {
        'koi_period': 'pl_orbper_days', 'koi_duration': 'pl_dur_hours', 
        'koi_prad': 'pl_prad_re', 'koi_srho': 'k_srho',
    }
    df = df.rename(columns=KEPLER_MAPPING)
    
    # Target Column
    target_col = 'koi_disposition'
    
    # 2. Columns to drop (Identifiers and CRITICAL LEAKAGE)
    KEPLER_DROP = [
        # --- CRITICAL LEAKAGE ---
        'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 
        'koi_fpflag_ec', 'koi_pdisposition', 'koi_vet_stat', 
        # --- Identifiers/Strings/Metadata ---
        'kepid', 'koi_name', 'kepler_name', 'koi_vet_date', 'koi_disp_prov', 
        'koi_comment', 'koi_fittype', 'koi_limbdark_mod', 'koi_parm_prov', 
        'koi_tce_delivname', 'koi_quarters', 'koi_trans_mod', 
        'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sparprov',
    ]
    
    # 3. Separate features (X) and target (y)
    y = df[target_col].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}).astype(int)
    
    # Drop target and leakage columns before selecting final features
    df_clean = df.drop(columns=[target_col] + KEPLER_DROP, errors='ignore') 
    
    # Select all remaining numerical columns as features
    X = df_clean.select_dtypes(include=np.number)
    
    logging.info(f"Kepler Features Remaining: {len(X.columns)}")
    return X, y

def preprocess_tess(df):
    """Selects and maps features unique to the TESS Candidate dataset, strictly removing leakage."""
    logging.info("Preprocessing TESS data: identifying leakage columns.")
    
    # 1. Rename core columns (using provided column names)
    TESS_MAPPING = {
        'pl_orbper': 'pl_orbper_days', 'pl_trandurh': 'pl_dur_hours', 
        'pl_rade': 'pl_prad_re', 'st_tmag': 'st_mag_tess',
    }
    df = df.rename(columns=TESS_MAPPING)
    
    # Target Column
    target_col = 'tfopwg_disp'
    
    # 2. Columns to drop (Identifiers, Strings, and CRITICAL LEAKAGE)
    TESS_DROP = [
        # --- CRITICAL LEAKAGE / Flags (Implicit Vetting) ---
        'pl_tranmidlim', 'pl_tranmidsymerr', 'pl_orbperlim', 'pl_orbpersymerr', 
        'pl_trandurhlim', 'pl_trandurhsymerr', 'pl_trandeplim', 'pl_trandepsymerr',
        'pl_radelim', 'pl_radesymerr', 'pl_insollim', 'pl_insolsymerr', 
        'pl_eqtlim', 'pl_eqtsymerr', 'st_tmaglim', 'st_tmagsymerr', 
        'st_distlim', 'st_distsymerr', 'st_tefflim', 'st_teffsymerr', 
        'st_logglim', 'st_loggsymerr', 'st_radlim', 'st_radsymerr',
        'st_pmralim', 'st_pmrasymerr', 'st_pmdeclim', 'st_pmdecsymerr',
        # --- Identifiers/Strings/Metadata ---
        'toi', 'toipfx', 'tid', 'ctoi_alias', 'pl_pnum', 'rastr', 'decstr', 
        'toi_created', 'rowupdate',
    ]
    
    # 3. Separate features (X) and target (y)
    # Mapping TESS TFOPWG dispositions: CP/KP/PC (Confirmed/Known/Candidate) -> 1, FP (False Positive) -> 0
    y = df[target_col].replace({'PC': 1, 'KP': 1, 'CP': 1, 'FA': 0, 'FP': 0, 'APC': 0}).astype(int)
    
    # Drop target and leakage columns before selecting final features
    df_clean = df.drop(columns=[target_col] + TESS_DROP, errors='ignore') 
    
    # Select all remaining numerical columns as features
    X = df_clean.select_dtypes(include=np.number)
    
    logging.info(f"TESS Features Remaining: {len(X.columns)}")
    return X, y

def preprocess_k2(df):
    """Selects and maps features unique to the K2/Archive dataset, strictly removing leakage."""
    logging.info("Preprocessing K2 data: identifying leakage columns.")
    
    # 1. Rename core columns (using provided column names)
    K2_MAPPING = {
        'pl_orbper': 'pl_orbper_days', 'pl_rade': 'pl_prad_re', 
        'st_teff': 'st_teff_k', 'st_mass': 'st_mass', 
        'sy_pnum': 'a_sy_pnum',
    }
    df = df.rename(columns=K2_MAPPING)
    
    # Target Column
    target_col = 'disposition'
    
    # 2. Columns to drop (Identifiers, Discovery Flags, and CRITICAL LEAKAGE)
    K2_DROP = [
        # --- CRITICAL LEAKAGE / Discovery Flags (Vetting Process Indicators) ---
        'pl_controv_flag', 'discoverymethod', 'disc_year', 'disc_refname', 
        'disc_pubdate', 'disc_locale', 'disc_facility', 'disc_telescope', 
        'disc_instrument', 
        # Discovery method flags (direct leakage from the vetting process)
        'rv_flag', 'pul_flag', 'ptv_flag', 'tran_flag', 'ast_flag', 'obm_flag', 
        'micro_flag', 'etv_flag', 'ima_flag', 'dkin_flag', 'default_flag',
        # Limit/Symmetric Flags (metadata)
        'pl_orbperlim', 'pl_orbsmaxlim', 'pl_radelim', 'pl_radjlim', 'pl_masselim',
        'pl_massjlim', 'pl_msinielim', 'pl_msinijlim', 'pl_cmasselim', 'pl_cmassjlim',
        'pl_bmasselim', 'pl_bmassjlim', 'pl_denslim', 'pl_orbeccenlim', 'pl_insollim', 
        'pl_eqtlim', 'pl_eqtsymerr', 'pl_orbincllim', 'pl_tranmidlim', 'pl_impparlim', 'pl_trandeplim',
        'pl_trandurlim', 'pl_ratdorlim', 'pl_ratrorlim', 'pl_occdeplim', 'pl_orbtperlim',
        'pl_orblperlim', 'pl_rvamplim', 'pl_projobliqlim', 'pl_trueobliqlim',
        'st_tefflim', 'st_radlim', 'st_masslim', 'st_metlim', 'st_lumlim', 
        'st_logglim', 'st_agelim', 'st_denslim', 'st_vsinlim', 'st_rotplim',
        'st_radvlim',
        # --- Identifiers/Strings/Metadata ---
        'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 'epic_candname', 
        'hd_name', 'hip_name', 'tic_id', 'gaia_id', 'disp_refname', 'st_refname', 
        'sy_refname', 'soltype', 'pl_refname', 'pl_bmassprov', 'pl_tsystemref', 
        'rastr', 'decstr', 'st_spectype', 'rowupdate', 'pl_pubdate', 'releasedate', 
        'k2_campaigns',
    ]
    
    # 3. Separate features (X) and target (y)
    y = df[target_col].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0, 'REFUTED': 0}).astype(int)
    
    # Drop target and leakage columns before selecting final features
    df_clean = df.drop(columns=[target_col] + K2_DROP, errors='ignore') 
    
    # Select all remaining numerical columns as features
    X = df_clean.select_dtypes(include=np.number)
    
    logging.info(f"K2 Features Remaining: {len(X.columns)}")
    return X, y

# =========================
# 2. Load and Split Data
# =========================

try:
    # Load and preprocess each dataset
    X_k_all, y_k_all = preprocess_kepler(pd.read_csv(KEPLER_FILE, comment='#'))
    X_t_all, y_t_all = preprocess_tess(pd.read_csv(TESS_FILE, comment='#'))
    X_k2_all, y_k2_all = preprocess_k2(pd.read_csv(K2_FILE, comment='#'))
    
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure the three original CSV files are in the directory.")
    raise

# Perform the train/test split on each mission's data
X_k_train, X_k_test, y_k_train, y_k_test = train_test_split(X_k_all, y_k_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_k_all)
X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_t_all, y_t_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_t_all)
X_k2_train, X_k2_test, y_k2_train, y_k2_test = train_test_split(X_k2_all, y_k2_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_k2_all)

# =========================
# 3. Base Model Training and OOF Prediction Generation
# =========================

# Updated the function to accept the specific params dictionary
def train_and_generate_oof(X_train, y_train, model_name, n_splits=N_SPLITS, params=LGBM_BASE_PARAMS):
    """
    Trains a base model and generates Out-of-Fold (OOF) predictions for stacking.
    Uses the provided params dictionary.
    """
    logging.info(f"\n--- Training Base Model: {model_name} ---")
        
    oof_preds = np.zeros(y_train.shape[0])
    base_model = lgb.LGBMClassifier(**params)
    
    # Imblearn pipeline ensures Random Over Sampling (ROS) is applied only to training folds
    pipeline = Pipeline([
        ('oversampler', RandomOverSampler(random_state=RANDOM_SEED)),
        ('classifier', base_model)
    ])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, oof_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_oof = X_train.iloc[train_idx], X_train.iloc[oof_idx]
        y_fold_train, y_fold_oof = y_train.iloc[train_idx], y_train.iloc[oof_idx]

        # Train the pipeline on the current fold
        pipeline.fit(X_fold_train, y_fold_train)
        
        # Predict on the OOF data for stacking
        oof_preds[oof_idx] = pipeline.predict_proba(X_fold_oof)[:, 1]

    logging.info(f"OOF AUC for {model_name}: {roc_auc_score(y_train, oof_preds):.4f}")
    
    # Train the final base model on ALL training data for test prediction consistency
    pipeline.fit(X_train, y_train)
    return pipeline, oof_preds


# --- Run Training for all three Base Models ---
# Kepler and K2 use the default BASE params
kepler_model, oof_k = train_and_generate_oof(X_k_train, y_k_train, "Kepler", params=LGBM_BASE_PARAMS)
k2_model, oof_k2 = train_and_generate_oof(X_k2_train, y_k2_train, "K2", params=LGBM_BASE_PARAMS)
# TESS uses the new, more complex TESS-specific params (Option A)
tess_model, oof_t = train_and_generate_oof(X_t_train, y_t_train, "TESS", params=LGBM_TESS_PARAMS)


# =========================
# 4. Meta-Learner Preparation
# =========================

# --- 4a. Training Data Construction (Unchanged) ---
oof_k_r = oof_k.reshape(-1, 1)
oof_t_r = oof_t.reshape(-1, 1)
oof_k2_r = oof_k2.reshape(-1, 1)

X_k_meta_train = np.hstack([oof_k_r, np.zeros_like(oof_k_r), np.zeros_like(oof_k_r)])
X_t_meta_train = np.hstack([np.zeros_like(oof_t_r), oof_t_r, np.zeros_like(oof_t_r)])
X_k2_meta_train = np.hstack([np.zeros_like(oof_k2_r), np.zeros_like(oof_k2_r), oof_k2_r])

X_meta_train = np.vstack([X_k_meta_train, X_t_meta_train, X_k2_meta_train])
y_meta_train = np.concatenate([y_k_train, y_t_train, y_k2_train])


# --- 4b. Test Data Construction (Unchanged) ---
p_k_test = kepler_model.predict_proba(X_k_test)[:, 1].reshape(-1, 1)
p_t_test = tess_model.predict_proba(X_t_test)[:, 1].reshape(-1, 1)
p_k2_test = k2_model.predict_proba(X_k2_test)[:, 1].reshape(-1, 1)

X_k_meta_test = np.hstack([p_k_test, np.zeros_like(p_k_test), np.zeros_like(p_k_test)])
X_t_meta_test = np.hstack([np.zeros_like(p_t_test), p_t_test, np.zeros_like(p_t_test)])
X_k2_meta_test = np.hstack([np.zeros_like(p_k2_test), np.zeros_like(p_k2_test), p_k2_test])

X_meta_test = np.vstack([X_k_meta_test, X_t_meta_test, X_k2_meta_test])
y_meta_test = np.concatenate([y_k_test, y_t_test, y_k2_test])

# =========================
# 5. Train and Evaluate Meta-Learner (UPDATED to LightGBM - Option C)
# =========================

logging.info("\n--- Training Meta-Learner: LightGBM Classifier ---")
# Replacing LogisticRegression with a LightGBM Classifier for non-linear weighting
meta_model = lgb.LGBMClassifier(**LGBM_META_PARAMS) 
meta_model.fit(X_meta_train, y_meta_train)

# Final probability prediction on the combined test set
final_stacked_preds = meta_model.predict_proba(X_meta_test)[:, 1]
final_auc = roc_auc_score(y_meta_test, final_stacked_preds)

# Calculate hard class predictions (0 or 1) using the standard 0.5 threshold
final_stacked_classes = (final_stacked_preds > 0.5).astype(int)
final_accuracy = accuracy_score(y_meta_test, final_stacked_classes)


# =========================
# Display Final Results
# =========================
print("\n#####################################################")
print("### STACKED ENSEMBLE FINAL PERFORMANCE REPORT (V3) ###")
print(f"Total Test Samples Evaluated: {len(y_meta_test)}")
print("-" * 50)
print(f"Final Stacked ROC-AUC Score: {final_auc:.4f}")
print(f"Final Stacked ACCURACY Score: {final_accuracy:.4f}")
print("-" * 50)
print("Classification Report (0=False Positive, 1=Confirmed/Candidate):")
print(classification_report(y_meta_test, final_stacked_classes))
print("#####################################################")
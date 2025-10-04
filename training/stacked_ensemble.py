import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score 
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import numpy as np
import logging
import pickle  # Import for saving
import os      # Import for path handling

logging.basicConfig(level=logging.INFO)

# =========================
# Configuration & File Paths
# =========================
KEPLER_FILE = '../datasets/keplar.csv'
TESS_FILE = '../datasets/tess.csv'
K2_FILE = '../datasets/k2.csv'
# --- NEW: Model Saving Configuration ---
SAVE_DIR = '../trained_full_models/'
os.makedirs(SAVE_DIR, exist_ok=True) # Ensure the directory exists

RANDOM_SEED = 42
N_SPLITS = 5
TUNED_PARAMS = {'feature_fraction': 0.85, 'bagging_fraction': 0.80, 'bagging_freq': 1, 'min_child_samples': 20}
LGBM_BASE_PARAMS = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.03, 'n_estimators': 500, 'num_leaves': 40, 'verbose': -1, 'seed': RANDOM_SEED, 'n_jobs': -1 , **TUNED_PARAMS}
LGBM_TESS_PARAMS = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.015, 'n_estimators': 1000, 'num_leaves': 61, 'verbose': -1, 'seed': RANDOM_SEED, 'n_jobs': -1 , **TUNED_PARAMS}
LGBM_META_PARAMS = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.05, 'n_estimators': 200, 'num_leaves': 16, 'verbose': -1, 'seed': RANDOM_SEED, 'n_jobs': -1, **TUNED_PARAMS}

# =========================
# 1. Feature Mapping (MODIFIED to return feature list)
# =========================

def preprocess_kepler(df):
    logging.info("Preprocessing Kepler data...")
    KEPLER_MAPPING = {'koi_period': 'pl_orbper_days', 'koi_duration': 'pl_dur_hours', 'koi_prad': 'pl_prad_re', 'koi_srho': 'k_srho'}
    df = df.rename(columns=KEPLER_MAPPING)
    target_col = 'koi_disposition'
    KEPLER_DROP = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_pdisposition', 'koi_vet_stat', 'kepid', 'koi_name', 'kepler_name', 'koi_vet_date', 'koi_disp_prov', 'koi_comment', 'koi_fittype', 'koi_limbdark_mod', 'koi_parm_prov', 'koi_tce_delivname', 'koi_quarters', 'koi_trans_mod', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sparprov']
    y = df[target_col].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}).astype(int)
    df_clean = df.drop(columns=[target_col] + KEPLER_DROP, errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    logging.info(f"Kepler Features Remaining: {len(X.columns)}")
    return X, y, list(X.columns) # <-- Return feature list

def preprocess_tess(df):
    logging.info("Preprocessing TESS data...")
    TESS_MAPPING = {'pl_orbper': 'pl_orbper_days', 'pl_trandurh': 'pl_dur_hours', 'pl_rade': 'pl_prad_re', 'st_tmag': 'st_mag_tess'}
    df = df.rename(columns=TESS_MAPPING)
    target_col = 'tfopwg_disp'
    TESS_DROP = ['pl_tranmidlim', 'pl_tranmidsymerr', 'pl_orbperlim', 'pl_orbpersymerr', 'pl_trandurhlim', 'pl_trandurhsymerr', 'pl_trandeplim', 'pl_trandepsymerr', 'pl_radelim', 'pl_radesymerr', 'pl_insollim', 'pl_insolsymerr', 'pl_eqtlim', 'pl_eqtsymerr', 'st_tmaglim', 'st_tmagsymerr', 'st_distlim', 'st_distsymerr', 'st_tefflim', 'st_teffsymerr', 'st_logglim', 'st_loggsymerr', 'st_radlim', 'st_radsymerr', 'st_pmralim', 'st_pmrasymerr', 'st_pmdeclim', 'st_pmdecsymerr', 'toi', 'toipfx', 'tid', 'ctoi_alias', 'pl_pnum', 'rastr', 'decstr', 'toi_created', 'rowupdate']
    y = df[target_col].replace({'PC': 1, 'KP': 1, 'CP': 1, 'FA': 0, 'FP': 0, 'APC': 0}).astype(int)
    df_clean = df.drop(columns=[target_col] + TESS_DROP, errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    logging.info(f"TESS Features Remaining: {len(X.columns)}")
    return X, y, list(X.columns) # <-- Return feature list

def preprocess_k2(df):
    logging.info("Preprocessing K2 data...")
    K2_MAPPING = {'pl_orbper': 'pl_orbper_days', 'pl_rade': 'pl_prad_re', 'st_teff': 'st_teff_k', 'st_mass': 'st_mass', 'sy_pnum': 'a_sy_pnum'}
    df = df.rename(columns=K2_MAPPING)
    target_col = 'disposition'
    K2_DROP = ['pl_controv_flag', 'discoverymethod', 'disc_year', 'disc_refname', 'disc_pubdate', 'disc_locale', 'disc_facility', 'disc_telescope', 'disc_instrument', 'rv_flag', 'pul_flag', 'ptv_flag', 'tran_flag', 'ast_flag', 'obm_flag', 'micro_flag', 'etv_flag', 'ima_flag', 'dkin_flag', 'default_flag', 'pl_orbperlim', 'pl_orbsmaxlim', 'pl_radelim', 'pl_radjlim', 'pl_masselim', 'pl_massjlim', 'pl_msinielim', 'pl_msinijlim', 'pl_cmasselim', 'pl_cmassjlim', 'pl_bmasselim', 'pl_bmassjlim', 'pl_denslim', 'pl_orbeccenlim', 'pl_insollim', 'pl_eqtlim', 'pl_eqtsymerr', 'pl_orbincllim', 'pl_tranmidlim', 'pl_impparlim', 'pl_trandeplim', 'pl_trandurlim', 'pl_ratdorlim', 'pl_ratrorlim', 'pl_occdeplim', 'pl_orbtperlim', 'pl_orblperlim', 'pl_rvamplim', 'pl_projobliqlim', 'pl_trueobliqlim', 'st_tefflim', 'st_radlim', 'st_masslim', 'st_metlim', 'st_lumlim', 'st_logglim', 'st_agelim', 'st_denslim', 'st_vsinlim', 'st_rotplim', 'st_radvlim', 'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 'epic_candname', 'hd_name', 'hip_name', 'tic_id', 'gaia_id', 'disp_refname', 'st_refname', 'sy_refname', 'soltype', 'pl_refname', 'pl_bmassprov', 'pl_tsystemref', 'rastr', 'decstr', 'st_spectype', 'rowupdate', 'pl_pubdate', 'releasedate', 'k2_campaigns']
    y = df[target_col].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0, 'REFUTED': 0}).astype(int)
    df_clean = df.drop(columns=[target_col] + K2_DROP, errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    logging.info(f"K2 Features Remaining: {len(X.columns)}")
    return X, y, list(X.columns) # <-- Return feature list

# =========================
# 2. Load and Split Data
# =========================

try:
    # Capture the feature lists upon loading
    X_k_all, y_k_all, features_k = preprocess_kepler(pd.read_csv(KEPLER_FILE, comment='#'))
    X_t_all, y_t_all, features_t = preprocess_tess(pd.read_csv(TESS_FILE, comment='#'))
    X_k2_all, y_k2_all, features_k2 = preprocess_k2(pd.read_csv(K2_FILE, comment='#'))
    
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure files are in the directory.")
    raise

# (Rest of the script from section 2 to section 5 remains the same...)

X_k_train, X_k_test, y_k_train, y_k_test = train_test_split(X_k_all, y_k_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_k_all)
X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_t_all, y_t_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_t_all)
X_k2_train, X_k2_test, y_k2_train, y_k2_test = train_test_split(X_k2_all, y_k2_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_k2_all)

def train_and_generate_oof(X_train, y_train, model_name, n_splits=N_SPLITS, params=LGBM_BASE_PARAMS):
    logging.info(f"\n--- Training Base Model: {model_name} ---")
    oof_preds = np.zeros(y_train.shape[0])
    base_model = lgb.LGBMClassifier(**params)
    pipeline = Pipeline([('oversampler', RandomOverSampler(random_state=RANDOM_SEED)), ('classifier', base_model)])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for fold, (train_idx, oof_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_oof = X_train.iloc[train_idx], X_train.iloc[oof_idx]
        y_fold_train, y_fold_oof = y_train.iloc[train_idx], y_train.iloc[oof_idx]
        pipeline.fit(X_fold_train, y_fold_train)
        oof_preds[oof_idx] = pipeline.predict_proba(X_fold_oof)[:, 1]
    logging.info(f"OOF AUC for {model_name}: {roc_auc_score(y_train, oof_preds):.4f}")
    pipeline.fit(X_train, y_train)
    return pipeline, oof_preds

kepler_model, oof_k = train_and_generate_oof(X_k_train, y_k_train, "Kepler", params=LGBM_BASE_PARAMS)
k2_model, oof_k2 = train_and_generate_oof(X_k2_train, y_k2_train, "K2", params=LGBM_BASE_PARAMS)
tess_model, oof_t = train_and_generate_oof(X_t_train, y_t_train, "TESS", params=LGBM_TESS_PARAMS)

oof_k_r = oof_k.reshape(-1, 1); oof_t_r = oof_t.reshape(-1, 1); oof_k2_r = oof_k2.reshape(-1, 1)
X_k_meta_train = np.hstack([oof_k_r, np.zeros_like(oof_k_r), np.zeros_like(oof_k_r)])
X_t_meta_train = np.hstack([np.zeros_like(oof_t_r), oof_t_r, np.zeros_like(oof_t_r)])
X_k2_meta_train = np.hstack([np.zeros_like(oof_k2_r), np.zeros_like(oof_k2_r), oof_k2_r])
X_meta_train = np.vstack([X_k_meta_train, X_t_meta_train, X_k2_meta_train])
y_meta_train = np.concatenate([y_k_train, y_t_train, y_k2_train])

p_k_test = kepler_model.predict_proba(X_k_test)[:, 1].reshape(-1, 1)
p_t_test = tess_model.predict_proba(X_t_test)[:, 1].reshape(-1, 1)
p_k2_test = k2_model.predict_proba(X_k2_test)[:, 1].reshape(-1, 1)
X_k_meta_test = np.hstack([p_k_test, np.zeros_like(p_k_test), np.zeros_like(p_k_test)])
X_t_meta_test = np.hstack([np.zeros_like(p_t_test), p_t_test, np.zeros_like(p_t_test)])
X_k2_meta_test = np.hstack([np.zeros_like(p_k2_test), np.zeros_like(p_k2_test), p_k2_test])
X_meta_test = np.vstack([X_k_meta_test, X_t_meta_test, X_k2_meta_test])
y_meta_test = np.concatenate([y_k_test, y_t_test, y_k2_test])

logging.info("\n--- Training Meta-Learner: LightGBM Classifier ---")
meta_model = lgb.LGBMClassifier(**LGBM_META_PARAMS) 
meta_model.fit(X_meta_train, y_meta_train)

final_stacked_preds = meta_model.predict_proba(X_meta_test)[:, 1]
final_auc = roc_auc_score(y_meta_test, final_stacked_preds)
final_stacked_classes = (final_stacked_preds > 0.5).astype(int)
final_accuracy = accuracy_score(y_meta_test, final_stacked_classes)

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


# =========================
# 6. Save Models and Features (NEW SECTION)
# =========================
logging.info("\n--- Saving All Full-Feature Models and Feature Lists ---")

# --- Save Models ---
models_to_save = {
    'kepler_full_model.pkl': kepler_model,
    'tess_full_model.pkl': tess_model,
    'k2_full_model.pkl': k2_model,
    'meta_full_model.pkl': meta_model,
}
for filename, model in models_to_save.items():
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Saved: {path}")

# --- Save Feature Lists ---
model_features = {
    'kepler_features': features_k,
    'tess_features': features_t,
    'k2_features': features_k2,
}
features_path = os.path.join(SAVE_DIR, 'full_model_features.pkl')
with open(features_path, 'wb') as f:
    pickle.dump(model_features, f)
logging.info(f"Saved: {features_path}")
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split

# --- Configuration (Must match training script) ---
SAVE_DIR = 'trained_models/'
KEPLER_FILE = './datasets/keplar.csv' 
TESS_FILE = './datasets/tess.csv'
K2_FILE = './datasets/k2.csv' # <-- ADDED K2 FILE PATH
RANDOM_SEED = 42

# --- Feature Preprocessing (Must match training script) ---
def preprocess_kepler_predict(df):
    KEPLER_MAPPING = {'koi_period': 'pl_orbper_days', 'koi_duration': 'pl_dur_hours', 
                      'koi_prad': 'pl_prad_re', 'koi_srho': 'k_srho'}
    df = df.rename(columns=KEPLER_MAPPING)
    target_col = 'koi_disposition'
    KEPLER_DROP = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 
                   'koi_fpflag_ec', 'koi_pdisposition', 'koi_vet_stat', 
                   'kepid', 'koi_name', 'kepler_name', 'koi_vet_date', 'koi_disp_prov', 
                   'koi_comment', 'koi_fittype', 'koi_limbdark_mod', 'koi_parm_prov', 
                   'koi_tce_delivname', 'koi_quarters', 'koi_trans_mod', 
                   'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sparprov']
    y = df[target_col].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}).astype(int)
    df_clean = df.drop(columns=[target_col] + KEPLER_DROP, errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    return X, y

def preprocess_tess_predict(df):
    TESS_MAPPING = {'pl_orbper': 'pl_orbper_days', 'pl_trandurh': 'pl_dur_hours', 
                    'pl_rade': 'pl_prad_re', 'st_tmag': 'st_mag_tess'}
    df = df.rename(columns=TESS_MAPPING)
    target_col = 'tfopwg_disp'
    TESS_DROP = ['pl_tranmidlim', 'pl_tranmidsymerr', 'pl_orbperlim', 'pl_orbpersymerr', 
                 'pl_trandurhlim', 'pl_trandurhsymerr', 'pl_trandeplim', 'pl_trandepsymerr',
                 'pl_radelim', 'pl_radesymerr', 'pl_insollim', 'pl_insolsymerr', 
                 'pl_eqtlim', 'pl_eqtsymerr', 'st_tmaglim', 'st_tmagsymerr', 
                 'st_distlim', 'st_distsymerr', 'st_tefflim', 'st_teffsymerr', 
                 'st_logglim', 'st_loggsymerr', 'st_radlim', 'st_radsymerr',
                 'st_pmralim', 'st_pmrasymerr', 'st_pmdeclim', 'st_pmdecsymerr',
                 'toi', 'toipfx', 'tid', 'ctoi_alias', 'pl_pnum', 'rastr', 'decstr', 
                 'toi_created', 'rowupdate']
    y = df[target_col].replace({'PC': 1, 'KP': 1, 'CP': 1, 'FA': 0, 'FP': 0, 'APC': 0}).astype(int)
    df_clean = df.drop(columns=[target_col] + TESS_DROP, errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    return X, y

# --- ADDED: K2 Preprocessing Function ---
def preprocess_k2_predict(df):
    K2_MAPPING = {
        'pl_orbper': 'pl_orbper_days', 'pl_rade': 'pl_prad_re', 
        'st_teff': 'st_teff_k', 'st_mass': 'st_mass', 
        'sy_pnum': 'a_sy_pnum',
    }
    df = df.rename(columns=K2_MAPPING)
    target_col = 'disposition'
    K2_DROP = [
        'pl_controv_flag', 'discoverymethod', 'disc_year', 'disc_refname', 
        'disc_pubdate', 'disc_locale', 'disc_facility', 'disc_telescope', 
        'disc_instrument', 
        'rv_flag', 'pul_flag', 'ptv_flag', 'tran_flag', 'ast_flag', 'obm_flag', 
        'micro_flag', 'etv_flag', 'ima_flag', 'dkin_flag', 'default_flag',
        'pl_orbperlim', 'pl_orbsmaxlim', 'pl_radelim', 'pl_radjlim', 'pl_masselim',
        'pl_massjlim', 'pl_msinielim', 'pl_msinijlim', 'pl_cmasselim', 'pl_cmassjlim',
        'pl_bmasselim', 'pl_bmassjlim', 'pl_denslim', 'pl_orbeccenlim', 'pl_insollim', 
        'pl_eqtlim', 'pl_eqtsymerr', 'pl_orbincllim', 'pl_tranmidlim', 'pl_impparlim', 'pl_trandeplim',
        'pl_trandurlim', 'pl_ratdorlim', 'pl_ratrorlim', 'pl_occdeplim', 'pl_orbtperlim',
        'pl_orblperlim', 'pl_rvamplim', 'pl_projobliqlim', 'pl_trueobliqlim',
        'st_tefflim', 'st_radlim', 'st_masslim', 'st_metlim', 'st_lumlim', 
        'st_logglim', 'st_agelim', 'st_denslim', 'st_vsinlim', 'st_rotplim',
        'st_radvlim',
        'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 'epic_candname', 
        'hd_name', 'hip_name', 'tic_id', 'gaia_id', 'disp_refname', 'st_refname', 
        'sy_refname', 'soltype', 'pl_refname', 'pl_bmassprov', 'pl_tsystemref', 
        'rastr', 'decstr', 'st_spectype', 'rowupdate', 'pl_pubdate', 'releasedate', 
        'k2_campaigns',
    ]
    y = df[target_col].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0, 'REFUTED': 0}).astype(int)
    df_clean = df.drop(columns=[target_col] + K2_DROP, errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    return X, y


# --- Load Models ---
try:
    kepler_model = pickle.load(open(os.path.join(SAVE_DIR, 'kepler_base_model.pkl'), 'rb')) # Assumes HGB model name
    tess_model = pickle.load(open(os.path.join(SAVE_DIR, 'tess_base_model.pkl'), 'rb'))     # Assumes HGB model name
    k2_model = pickle.load(open(os.path.join(SAVE_DIR, 'k2_base_model.pkl'), 'rb'))         # <-- ADDED K2 MODEL LOAD
except FileNotFoundError:
    print("Error: Base models not found. Please ensure the models (e.g., *_hgb.pkl) are in the 'trained_models/' directory.")
    exit()

# --- Load Data and Select Test Rows ---
try:
    X_k_all, y_k_all = preprocess_kepler_predict(pd.read_csv(KEPLER_FILE, comment='#'))
    X_t_all, y_t_all = preprocess_tess_predict(pd.read_csv(TESS_FILE, comment='#'))
    X_k2_all, y_k2_all = preprocess_k2_predict(pd.read_csv(K2_FILE, comment='#')) # <-- ADDED K2 DATA LOAD
except FileNotFoundError:
    print("Error: Data files not found. Please ensure 'keplar.csv', 'tess.csv', and 'k2.csv' are accessible.")
    exit()

# Split to replicate the test set indices for all missions
_, X_k_test, _, y_k_test = train_test_split(X_k_all, y_k_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_k_all)
_, X_t_test, _, y_t_test = train_test_split(X_t_all, y_t_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_t_all)
_, X_k2_test, _, y_k2_test = train_test_split(X_k2_all, y_k2_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_k2_all) # <-- ADDED K2 SPLIT


# 1. Kepler Confirmed Check (Retained for context)
kepler_cp_row = X_k_test[y_k_test == 1].iloc[[0]]
kepler_prob_conf = kepler_model.predict_proba(kepler_cp_row)[0, 1]

# --- ADDED: Kepler False Positive Check ---
kepler_fp_row = X_k_test[y_k_test == 0].iloc[[0]]
kepler_prob_fp = kepler_model.predict_proba(kepler_fp_row)[0, 1]

# 2. TESS Confirmed Check (Retained for context)
tess_cp_row = X_t_test[y_t_test == 1].iloc[[0]]
tess_prob_conf = tess_model.predict_proba(tess_cp_row)[0, 1]

# --- ADDED: K2 Confirmed Check ---
k2_cp_row = X_k2_test[y_k2_test == 1].iloc[[0]]
k2_prob_conf = k2_model.predict_proba(k2_cp_row)[0, 1]

# --- ADDED: K2 False Positive Check ---
k2_fp_row = X_k2_test[y_k2_test == 0].iloc[[0]]
k2_prob_fp = k2_model.predict_proba(k2_fp_row)[0, 1]


print("\n--- Model Verification (Full Feature Input for Base Models) ---")
print("KEPLER MODEL CHECKS:")
print(f"  P(Confirmed) for **Confirmed** Kepler Test Row (Expected High): {kepler_prob_conf:.4f}")
print(f"  P(Confirmed) for **False Positive** Kepler Test Row (Expected Low): {kepler_prob_fp:.4f}")
print("-" * 30)
print("K2 MODEL CHECKS:")
print(f"  P(Confirmed) for **Confirmed** K2 Test Row (Expected High): {k2_prob_conf:.4f}")
print(f"  P(Confirmed) for **False Positive** K2 Test Row (Expected Low): {k2_prob_fp:.4f}")
print("-" * 30)
print(f"TESS Model P(Confirmed) for Confirmed Test Row: {tess_prob_conf:.4f}")
print("-------------------------------------------------------------")
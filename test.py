import pandas as pd
import numpy as np
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)

# =========================
# Configuration & File Paths
# =========================
MODEL_SAVE_DIR = './trained_full_models' 

KEPLER_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'kepler_full_model.pkl')
TESS_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'tess_full_model.pkl')
K2_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'k2_full_model.pkl')
META_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'meta_full_model.pkl')
FEATURE_DICT_PATH = os.path.join(MODEL_SAVE_DIR, 'full_model_features.pkl')

KEPLER_FILE = './datasets/keplar.csv'
TESS_FILE = './datasets/tess.csv'
K2_FILE = './datasets/k2.csv'

# =========================
# 1. Loading Functions
# =========================
def load_all_models():
    """Loads all four full-feature models from disk."""
    try:
        with open(KEPLER_MODEL_PATH, 'rb') as f: k_model = pickle.load(f)
        with open(TESS_MODEL_PATH, 'rb') as f: t_model = pickle.load(f)
        with open(K2_MODEL_PATH, 'rb') as f: k2_model = pickle.load(f)
        with open(META_MODEL_PATH, 'rb') as f: meta_model = pickle.load(f)
        logging.info("All four full-feature models loaded successfully.")
        return k_model, t_model, k2_model, meta_model
    except FileNotFoundError as e:
        logging.error(f"Error loading models: {e.filename}. Make sure training script was run.")
        return None, None, None, None

def load_all_features():
    """Loads the full-feature lists from the dictionary file."""
    try:
        with open(FEATURE_DICT_PATH, 'rb') as f: feature_dict = pickle.load(f)
        return (feature_dict.get('kepler_features'), 
                feature_dict.get('tess_features'), 
                feature_dict.get('k2_features'))
    except FileNotFoundError: 
        logging.error(f"Feature dictionary not found at {FEATURE_DICT_PATH}")
        return None, None, None

# =========================
# 2. Preprocessing Functions (Copied from training script for consistency)
# =========================
def preprocess_kepler(df):
    logging.info("Preprocessing Kepler data...")
    KEPLER_MAPPING = {'koi_period': 'pl_orbper_days', 'koi_duration': 'pl_dur_hours', 'koi_prad': 'pl_prad_re', 'koi_srho': 'k_srho'}
    df = df.rename(columns=KEPLER_MAPPING)
    KEPLER_DROP = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_pdisposition', 'koi_vet_stat', 'kepid', 'koi_name', 'kepler_name', 'koi_vet_date', 'koi_disp_prov', 'koi_comment', 'koi_fittype', 'koi_limbdark_mod', 'koi_parm_prov', 'koi_tce_delivname', 'koi_quarters', 'koi_trans_mod', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sparprov']
    df_clean = df.drop(columns=KEPLER_DROP + ['koi_disposition'], errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    return X

def preprocess_tess(df):
    logging.info("Preprocessing TESS data...")
    TESS_MAPPING = {'pl_orbper': 'pl_orbper_days', 'pl_trandurh': 'pl_dur_hours', 'pl_rade': 'pl_prad_re', 'st_tmag': 'st_mag_tess'}
    df = df.rename(columns=TESS_MAPPING)
    TESS_DROP = ['pl_tranmidlim', 'pl_tranmidsymerr', 'pl_orbperlim', 'pl_orbpersymerr', 'pl_trandurhlim', 'pl_trandurhsymerr', 'pl_trandeplim', 'pl_trandepsymerr', 'pl_radelim', 'pl_radesymerr', 'pl_insollim', 'pl_insolsymerr', 'pl_eqtlim', 'pl_eqtsymerr', 'st_tmaglim', 'st_tmagsymerr', 'st_distlim', 'st_distsymerr', 'st_tefflim', 'st_teffsymerr', 'st_logglim', 'st_loggsymerr', 'st_radlim', 'st_radsymerr', 'st_pmralim', 'st_pmrasymerr', 'st_pmdeclim', 'st_pmdecsymerr', 'toi', 'toipfx', 'tid', 'ctoi_alias', 'pl_pnum', 'rastr', 'decstr', 'toi_created', 'rowupdate']
    df_clean = df.drop(columns=TESS_DROP + ['tfopwg_disp'], errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    return X

def preprocess_k2(df):
    logging.info("Preprocessing K2 data...")
    K2_MAPPING = {'pl_orbper': 'pl_orbper_days', 'pl_rade': 'pl_prad_re', 'st_teff': 'st_teff_k', 'st_mass': 'st_mass', 'sy_pnum': 'a_sy_pnum'}
    df = df.rename(columns=K2_MAPPING)
    K2_DROP = ['pl_controv_flag', 'discoverymethod', 'disc_year', 'disc_refname', 'disc_pubdate', 'disc_locale', 'disc_facility', 'disc_telescope', 'disc_instrument', 'rv_flag', 'pul_flag', 'ptv_flag', 'tran_flag', 'ast_flag', 'obm_flag', 'micro_flag', 'etv_flag', 'ima_flag', 'dkin_flag', 'default_flag', 'pl_orbperlim', 'pl_orbsmaxlim', 'pl_radelim', 'pl_radjlim', 'pl_masselim', 'pl_massjlim', 'pl_msinielim', 'pl_msinijlim', 'pl_cmasselim', 'pl_cmassjlim', 'pl_bmasselim', 'pl_bmassjlim', 'pl_denslim', 'pl_orbeccenlim', 'pl_insollim', 'pl_eqtlim', 'pl_eqtsymerr', 'pl_orbincllim', 'pl_tranmidlim', 'pl_impparlim', 'pl_trandeplim', 'pl_trandurlim', 'pl_ratdorlim', 'pl_ratrorlim', 'pl_occdeplim', 'pl_orbtperlim', 'pl_orblperlim', 'pl_rvamplim', 'pl_projobliqlim', 'pl_trueobliqlim', 'st_tefflim', 'st_radlim', 'st_masslim', 'st_metlim', 'st_lumlim', 'st_logglim', 'st_agelim', 'st_denslim', 'st_vsinlim', 'st_rotplim', 'st_radvlim', 'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 'epic_candname', 'hd_name', 'hip_name', 'tic_id', 'gaia_id', 'disp_refname', 'st_refname', 'sy_refname', 'soltype', 'pl_refname', 'pl_bmassprov', 'pl_tsystemref', 'rastr', 'decstr', 'st_spectype', 'rowupdate', 'pl_pubdate', 'releasedate', 'k2_campaigns']
    df_clean = df.drop(columns=K2_DROP + ['disposition'], errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    return X

# =========================
# 3. Prediction Function
# =========================
def align_dataframe(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """Aligns a DataFrame to have the exact columns in the exact order as the feature_list."""
    aligned_df = pd.DataFrame(columns=feature_list, index=df.index)
    for col in df.columns:
        if col in aligned_df.columns:
            aligned_df[col] = df[col]
    return aligned_df.fillna(0) # Use a simple imputation for missing values

models_tuple = None
features_tuple = None
def prepare():
    global models_tuple, features_tuple
    models_tuple = load_all_models()
    features_tuple = load_all_features()
    
def predict(df: pd.DataFrame) -> dict:
    """
    Takes a raw DataFrame, identifies its source, preprocesses it, and
    returns the final stacked prediction.

    Args:
        df: The input DataFrame, matching a mission's original format.
        models: A tuple containing the (kepler, tess, k2, meta) models.
        features: A tuple containing the (kepler, tess, k2) feature lists.

    Returns:
        A dictionary containing the base and final predictions.
    """
    global models_tuple, features_tuple
    models = models_tuple
    features = features_tuple
    kepler_model, tess_model, k2_model, meta_model = models
    k_feats, t_feats, k2_feats = features
    
    # 1. Identify the data source based on signature columns
    if 'koi_disposition' in df.columns:
        mission = 'Kepler'
        X_processed = preprocess_kepler(df)
        X_aligned = align_dataframe(X_processed, k_feats)
        base_model = kepler_model
        
    elif 'tfopwg_disp' in df.columns:
        mission = 'TESS'
        X_processed = preprocess_tess(df)
        X_aligned = align_dataframe(X_processed, t_feats)
        base_model = tess_model
        
    elif 'disposition' in df.columns:
        mission = 'K2'
        X_processed = preprocess_k2(df)
        X_aligned = align_dataframe(X_processed, k2_feats)
        base_model = k2_model
        
    else:
        raise ValueError("Could not identify mission source from DataFrame columns.")

    logging.info(f"Identified DataFrame as {mission} data.")

    # 2. Get the base prediction from the appropriate model
    base_pred = base_model.predict_proba(X_aligned)[:, 1].reshape(-1, 1)

    # 3. Construct the sparse input for the meta-model
    if mission == 'Kepler':
        X_meta_input = np.hstack([base_pred, np.zeros_like(base_pred), np.zeros_like(base_pred)])
    elif mission == 'TESS':
        X_meta_input = np.hstack([np.zeros_like(base_pred), base_pred, np.zeros_like(base_pred)])
    elif mission == 'K2':
        X_meta_input = np.hstack([np.zeros_like(base_pred), np.zeros_like(base_pred), base_pred])

    # 4. Get the final stacked prediction
    final_prediction = meta_model.predict_proba(X_meta_input)[:, 1]
    
    return {
        "mission_source": mission,
        "base_prediction_prob": base_pred.flatten(),
        "final_stacked_prob": final_prediction
    }

# =========================
# 4. Main Execution Block (Example Usage)
# =========================
if __name__ == "__main__":
    # Load all models and feature lists once
    models_tuple = load_all_models()
    features_tuple = load_all_features()
    
    if not all(models_tuple) or not all(features_tuple):
        print("\nPrediction halted. Could not load all necessary files.")
    else:
        # --- EXAMPLE: Predict on the first 10 rows of the Kepler test set ---
        try:
            # Load one of the original files to simulate a new data upload
            new_data_df = pd.read_csv(KEPLER_FILE, comment='#')
            # Let's pretend we only want to predict on the first 10 candidates
            data_to_predict = new_data_df.head(10)

            # Get the predictions
            prepare()
            results = predict(data_to_predict)
            
            print("\n" + "="*50)
            print("PREDICTION RESULTS ON NEW KEPLER DATA")
            print("="*50)
            
            # Create a nice output DataFrame to display results
            output_df = pd.DataFrame({
                'Original_Disposition': data_to_predict['koi_disposition'],
                'Base_Model_Probability': results['base_prediction_prob'],
                'Final_Stacked_Probability': results['final_stacked_prob']
            })
            print(output_df)
            print("="*50)

        except FileNotFoundError:
            print(f"Could not find {KEPLER_FILE} to run the example prediction.")
        except Exception as e:
            print(f"An error occurred during the example prediction: {e}")
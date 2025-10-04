import pandas as pd
import numpy as np
import pickle
import os
import logging

# Set logging level
logging.basicConfig(level=logging.INFO)

# =========================
# Configuration & File Paths
# =========================
MODEL_SAVE_DIR = './trained_models' 

KEPLER_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'kepler_base_model.pkl')
TESS_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'tess_base_model.pkl')
K2_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'k2_base_model.pkl')
META_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'stacked_meta_model.pkl')
FEATURE_DICT_PATH = os.path.join(MODEL_SAVE_DIR, 'model_features.pkl')


# =========================
# 3. Core Mapping Definitions (CORRECTED)
# =========================

# REMOVED 'radius_ratio' from all maps to prevent data leakage.
# Pointing TESS/K2 error columns to the new normalized key.
USER_TO_MISSION_MAPS = {
    'Kepler': {
        'planetary_radius': 'pl_prad_re',            
        'orbital_period': 'pl_orbper_days',          
        'insolation_flux': 'koi_insol',              
        'transit_depth': 'koi_depth',                # Kepler uses PPM
        'transit_duration': 'pl_dur_hours',          
        'impact_parameter': 'koi_impact',            
        'stellar_temp': 'koi_teff',                  
        'stellar_radius': 'koi_srad',                
        'model_snr': 'koi_model_snr',                 # Crucial for Kepler
        'stellar_logg': 'koi_slogg'
    },
    'TESS': {
        'planetary_radius': 'pl_rade', 
        'orbital_period': 'pl_orbper', 
        'transit_duration': 'pl_trandurh',
        'transit_depth_fractional': 'pl_trandep',    # USES NORMALIZED KEY
        'transit_depth_err_fractional': 'pl_trandeperr1', # USES NORMALIZED KEY
        'insolation_flux': 'pl_insol',
        'stellar_temp': 'st_teff',
        'stellar_radius': 'st_rad',
        'stellar_mass': 'st_mass',                   
        'stellar_logg': 'st_logg'
    },
    'K2': {
        'planetary_radius': 'pl_prad_re',
        'orbital_period': 'pl_orbper_days',
        'transit_duration': 'pl_trandur',
        'transit_depth_fractional': 'pl_trandep',    # USES NORMALIZED KEY
        'transit_depth_err_fractional': 'pl_trandep_err1', # USES NORMALIZED KEY
        'impact_parameter': 'pl_imppar',
        'insolation_flux': 'pl_insol',
        'stellar_temp': 'st_teff_k',
        'stellar_radius': 'st_rad',
        'stellar_mass': 'st_mass',                   
        'stellar_logg': 'st_logg'
    }
}

# =========================
# 4. Utility Functions (CORRECTED)
# =========================

def load_all_models():
    """Loads all models from disk."""
    try:
        logging.info("Attempting to load saved models...")
        with open(KEPLER_MODEL_PATH, 'rb') as f: kepler_model = pickle.load(f)
        with open(TESS_MODEL_PATH, 'rb') as f: tess_model = pickle.load(f)
        with open(K2_MODEL_PATH, 'rb') as f: k2_model = pickle.load(f)
        with open(META_MODEL_PATH, 'rb') as f: meta_model = pickle.load(f)
        logging.info("All four models loaded successfully.")
        return kepler_model, tess_model, k2_model, meta_model
    except FileNotFoundError as e:
        logging.error(f"Error loading models. Did you run the training script? Missing file: {e.filename}")
        return None, None, None, None

def load_all_features():
    """Loads all feature lists from disk."""
    try:
        with open(FEATURE_DICT_PATH, 'rb') as f: feature_dict = pickle.load(f)
        return (feature_dict.get('kepler_features'), 
                feature_dict.get('tess_features'), 
                feature_dict.get('k2_features'))
    except FileNotFoundError as e:
        logging.error(f"Error loading feature lists. Missing file: {e.filename}")
        return None, None, None

# --- CORRECTED NORMALIZER ---
def normalize_user_inputs(user_inputs: dict) -> dict:
    """Handles all unit conversions before features are mapped."""
    # Convert transit depth from PPM to fractional for TESS/K2
    if 'transit_depth' in user_inputs and user_inputs['transit_depth'] is not None:
        user_inputs['transit_depth_fractional'] = user_inputs['transit_depth'] / 1_000_000.0
    
    # Convert transit depth error from PPM to fractional for TESS/K2
    if 'transit_depth_err' in user_inputs and user_inputs['transit_depth_err'] is not None:
        user_inputs['transit_depth_err_fractional'] = user_inputs['transit_depth_err'] / 1_000_000.0

    return user_inputs

def create_mapped_dataframe(user_inputs: dict, map_dict: dict) -> pd.DataFrame:
    """Creates a single-row DataFrame using user values and mission-specific column names."""
    mapped_data = {internal_key: [user_inputs[user_key]] for user_key, internal_key in map_dict.items() if user_key in user_inputs}
    return pd.DataFrame(mapped_data) if mapped_data else pd.DataFrame()

# --- CORRECTED FEATURE ENGINEERING ---
def engineer_features(df: pd.DataFrame, mission_type: str) -> pd.DataFrame:
    """Engineers new features, now with correct unit conversions."""
    if df.empty: return df
    logging.info(f"Engineering features for {mission_type} prediction...")
    epsilon = 1e-6
    
    # Conversion factor: ~109.2 Earth Radii per 1 Solar Radius
    EARTH_TO_SOLAR_RADIUS = 109.2
    
    # Standardize column names for calculation
    # Use .get() to avoid errors if a column is missing
    planet_rad_er = df.get('pl_prad_re', df.get('pl_rade', np.nan))
    star_rad_sr = df.get('koi_srad', df.get('st_rad', np.nan))

    # Calculate size ratio with correct units
    if pd.notna(planet_rad_er).any() and pd.notna(star_rad_sr).any():
        planet_rad_sr = planet_rad_er / EARTH_TO_SOLAR_RADIUS
        df['fe_size_ratio'] = planet_rad_sr / (star_rad_sr + epsilon)

    if mission_type == 'Kepler':
        df['fe_transit_shape'] = df.get('pl_dur_hours', np.nan) / (df.get('pl_orbper_days', np.nan) + epsilon)
        df['fe_certainty_depth'] = df.get('koi_depth', np.nan) / (df.get('koi_model_snr', np.nan) + epsilon)
        df['fe_temp_proxy'] = (df.get('koi_insol', 0).clip(lower=0) * df.get('koi_teff', 0).clip(lower=0))**0.25
        
    elif mission_type == 'TESS':
        df['fe_transit_shape'] = df.get('pl_trandurh', np.nan) / (df.get('pl_orbper', np.nan) + epsilon)
        df['fe_certainty_depth'] = df.get('pl_trandep', np.nan) / (df.get('pl_trandeperr1', np.nan) + epsilon)
        df['fe_temp_proxy'] = (df.get('pl_insol', 0).clip(lower=0) * df.get('st_teff', 0).clip(lower=0))**0.25

    elif mission_type == 'K2':
        df['fe_transit_shape'] = df.get('pl_trandur', np.nan) / (df.get('pl_orbper_days', np.nan) + epsilon)
        df['fe_temp_proxy'] = (df.get('pl_insol', 0).clip(lower=0) * df.get('st_teff_k', 0).clip(lower=0))**0.25

    return df

def create_aligned_dataframe(X_raw: pd.DataFrame, expected_cols: list, mission_type: str) -> pd.DataFrame:
    """Aligns DataFrame to the model's expected feature order."""
    X_aligned = pd.DataFrame(index=X_raw.index, columns=expected_cols)
    for col in X_raw.columns:
        if col in expected_cols: X_aligned[col] = X_raw[col]
    return X_aligned.fillna(np.nan)

# =====================================================================
# 5. FINAL GENERAL PREDICTION FUNCTION (No changes needed here)
# =====================================================================

def predict_general_candidate(user_inputs: dict, models, features, maps):
    """
    Normalizes inputs, maps them, engineers features, and returns a final stacked prediction.
    """
    normalized_inputs = normalize_user_inputs(user_inputs.copy())
    
    kepler_model, tess_model, k2_model, meta_model = models
    kepler_features, tess_features, k2_features = features
    
    # 1. Map, Engineer, and Align
    X_k_aligned = create_aligned_dataframe(engineer_features(create_mapped_dataframe(normalized_inputs, maps['Kepler']), 'Kepler'), kepler_features, 'Kepler')
    X_t_aligned = create_aligned_dataframe(engineer_features(create_mapped_dataframe(normalized_inputs, maps['TESS']), 'TESS'), tess_features, 'TESS')
    X_k2_aligned = create_aligned_dataframe(engineer_features(create_mapped_dataframe(normalized_inputs, maps['K2']), 'K2'), k2_features, 'K2')
    
    # 2. Generate Base Probabilities
    p_k = kepler_model.predict_proba(X_k_aligned)[:, 1][0]
    p_t = tess_model.predict_proba(X_t_aligned)[:, 1][0]
    p_k2 = k2_model.predict_proba(X_k2_aligned)[:, 1][0]
    logging.info(f"Base Probabilities: Kepler={p_k:.4f}, TESS={p_t:.4f}, K2={p_k2:.4f}")

    # 3. Meta-Learner Prediction (Sparse Input Method)
    meta_input_k = np.array([[p_k, 0, 0]])
    meta_input_t = np.array([[0, p_t, 0]])
    meta_input_k2 = np.array([[0, 0, p_k2]])
    pred_from_k = meta_model.predict_proba(meta_input_k)[:, 1][0]
    pred_from_t = meta_model.predict_proba(meta_input_t)[:, 1][0]
    pred_from_k2 = meta_model.predict_proba(meta_input_k2)[:, 1][0]
    final_stacked_proba = (pred_from_k + pred_from_t + pred_from_k2) / 3.0
    
    return p_k, p_t, p_k2, final_stacked_proba

# =========================
# 6. Main Execution Block
# =========================

if __name__ == "__main__":
    features_tuple = load_all_features()
    models_tuple = load_all_models()

    if not all(features_tuple) or not all(models_tuple):
        print("\nPrediction halted. Ensure models and features are loaded successfully.")
    else:
        # Example Input: Strong "Confirmed/Candidate" Planet (Hot Jupiter)
        # REMOVED 'radius_ratio' as it's now calculated internally
        web_input_strong_candidate = {
            'planetary_radius': 12.0, 'orbital_period': 3.5,
            'insolation_flux': 850.0, 'transit_depth': 12000, 'transit_duration': 3.1,
            'impact_parameter': 0.2, 'stellar_temp': 5800.0, 'stellar_radius': 1.0,
            'model_snr': 350.0, 'stellar_mass': 1.0, 'stellar_logg': 4.4,
            'transit_depth_err': 50.0
        }

        print("\n" + "="*50)
        print("GENERAL CANDIDATE PREDICTION (V4 - Corrected Units)")
        print("="*50)
        print("Input Parameters (Strong Candidate Example):")
        for k, v in web_input_strong_candidate.items():
            print(f"  {k}: {v}")
        
        p_k, p_t, p_k2, final_proba = predict_general_candidate(
            web_input_strong_candidate, models_tuple, features_tuple, USER_TO_MISSION_MAPS
        )
        
        if final_proba is not None:
            print(f"\n[Input Summary] {len(web_input_strong_candidate)} features provided.")
            print("-" * 40)
            print("Base Model Probabilities (P(Confirmed)):")
            print(f"  Kepler Base: {p_k:.4f}")
            print(f"  TESS Base:   {p_t:.4f}")
            print(f"  K2 Base:     {p_k2:.4f}")
            print("\nFinal Stacked Output:")
            print(f"  Final Stacked Probability: {final_proba:.4f}")

        print("\n" + "="*50)
        print("Prediction process complete.")
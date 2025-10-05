import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import Literal
import lightkurve as lk
from math import sqrt

logging.basicConfig(level=logging.INFO)

# =========================
# Configuration & File Paths
# =========================
MODEL_SAVE_DIR = './trained_reduced_models' 

KEPLER_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'kepler_reduced_model.pkl')
TESS_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'tess_reduced_model.pkl')
K2_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'k2_reduced_model.pkl')
META_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'meta_reduced_model.pkl')
FEATURE_DICT_PATH = os.path.join(MODEL_SAVE_DIR, 'reduced_model_features.pkl')

# =========================
# 3. Core Mapping Definitions (Final & Corrected)
# =========================
USER_TO_MISSION_MAPS = {
    'Kepler': {
        'planetary_radius': 'pl_prad_re', 'orbital_period': 'pl_orbper_days',
        'insolation_flux': 'koi_insol', 'transit_depth': 'koi_depth',
        'transit_duration': 'pl_dur_hours', 'impact_parameter': 'koi_impact',
        'stellar_temp': 'koi_teff', 'stellar_radius': 'koi_srad',
        'model_snr': 'koi_model_snr', 'stellar_logg': 'koi_slogg',
        'transit_depth_err': 'koi_depth_err1'
    },
    'TESS': {
        'planetary_radius': 'pl_rade', 'orbital_period': 'pl_orbper',
        'transit_duration': 'pl_trandurh', 'transit_depth': 'pl_trandep',
        'transit_depth_err': 'pl_trandeperr1', 'insolation_flux': 'pl_insol',
        'stellar_temp': 'st_teff', 'stellar_radius': 'st_rad', 'stellar_logg': 'st_logg'
    },
    'K2': {
        'planetary_radius': 'pl_prad_re', 'orbital_period': 'pl_orbper_days',
        'transit_duration': 'pl_trandur', 'transit_depth_fractional': 'pl_trandep',
        'transit_depth_err_fractional': 'pl_trandeperr1', 'impact_parameter': 'pl_imppar',
        'insolation_flux': 'pl_insol', 'stellar_temp': 'st_teff_k',
        'stellar_radius': 'st_rad', 'stellar_mass': 'st_mass', 'stellar_logg': 'st_logg'
    }
}

# =========================
# 4. Utility Functions (Final & Corrected)
# =========================
def load_models():
    """Loads all four reduced models."""
    try:
        with open(KEPLER_MODEL_PATH, 'rb') as f: k_model = pickle.load(f)
        with open(TESS_MODEL_PATH, 'rb') as f: t_model = pickle.load(f)
        with open(K2_MODEL_PATH, 'rb') as f: k2_model = pickle.load(f)
        with open(META_MODEL_PATH, 'rb') as f: meta_model = pickle.load(f)
        logging.info("All four models (3 base + 1 meta) loaded successfully.")
        return k_model, t_model, k2_model, meta_model
    except FileNotFoundError as e:
        logging.error(f"Error loading models: {e.filename}")
        return None, None, None, None

def load_features():
    """Loads the reduced feature lists."""
    try:
        with open(FEATURE_DICT_PATH, 'rb') as f: feature_dict = pickle.load(f)
        return (feature_dict.get('kepler_features'), 
                feature_dict.get('tess_features'), 
                feature_dict.get('k2_features'))
    except FileNotFoundError: return None, None, None

def normalize_user_inputs(inputs):
    """Handles unit conversions for K2, which expects fractional depth."""
    if 'transit_depth' in inputs and inputs['transit_depth'] is not None:
        inputs['transit_depth_fractional'] = inputs['transit_depth'] / 1_000_000.0
    if 'transit_depth_err' in inputs and inputs['transit_depth_err'] is not None:
        inputs['transit_depth_err_fractional'] = inputs['transit_depth_err'] / 1_000_000.0
    return inputs

def create_mapped_dataframe(inputs, map_dict):
    """Creates a single-row DataFrame from user inputs."""
    mapped = {key: [inputs[user_key]] for user_key, key in map_dict.items() if user_key in inputs}
    return pd.DataFrame(mapped)

def engineer_features(df, mission):
    """Engineers new features, consistent with the training script."""
    if df.empty: return df
    epsilon = 1e-6; EARTH_TO_SOLAR_RADIUS = 109.2
    planet_rad_er = df.get('pl_prad_re', df.get('pl_rade')).iloc[0]
    star_rad_sr = df.get('koi_srad', df.get('st_rad')).iloc[0]
    if pd.notna(planet_rad_er) and pd.notna(star_rad_sr):
        df['fe_size_ratio'] = (planet_rad_er / EARTH_TO_SOLAR_RADIUS) / (star_rad_sr + epsilon)
    if mission == 'Kepler':
        df['fe_transit_shape'] = df.get('pl_dur_hours', np.nan) / (df.get('pl_orbper_days', np.nan) + epsilon)
        df['fe_certainty_depth'] = df.get('koi_depth', np.nan) / (df.get('koi_model_snr', np.nan) + epsilon)
        df['fe_temp_proxy'] = (df.get('koi_insol', 0) * df.get('koi_teff', 0))**0.25
    elif mission == 'TESS':
        df['fe_transit_shape'] = df.get('pl_trandurh', np.nan) / (df.get('pl_orbper', np.nan) + epsilon)
        df['fe_certainty_depth'] = df.get('pl_trandep', np.nan) / (df.get('pl_trandeperr1', np.nan) + epsilon)
        df['fe_temp_proxy'] = (df.get('pl_insol', 0) * df.get('st_teff', 0))**0.25
    elif mission == 'K2':
        df['fe_transit_shape'] = df.get('pl_trandur', np.nan) / (df.get('pl_orbper_days', np.nan) + epsilon)
        df['fe_certainty_depth'] = df.get('pl_trandep', np.nan) / (df.get('pl_trandeperr1', np.nan) + epsilon)
        df['fe_temp_proxy'] = (df.get('pl_insol', 0) * df.get('st_teff_k', 0))**0.25
    return df

def create_aligned_dataframe(X_raw, expected_cols):
    """Aligns DataFrame to the model's expected feature order."""
    X_aligned = pd.DataFrame(index=[0], columns=expected_cols)
    for col in X_raw.columns:
        if col in expected_cols: X_aligned[col] = X_raw[col].iloc[0]
    return X_aligned.fillna(np.nan)

# =====================================================================
# 5. Prediction Function with Stacking Logic
# =====================================================================
def predict_from_manual(user_inputs, models, features):
    """
    Takes user inputs, gets base predictions, and feeds them to the meta-model
    for a final, stacked prediction.
    """
    norm_inputs = normalize_user_inputs(user_inputs.copy())
    kepler_model, tess_model, k2_model, meta_model = models
    k_feats, t_feats, k2_feats = features

    # --- 1. Get Base Probabilities ---
    X_k = create_aligned_dataframe(engineer_features(create_mapped_dataframe(norm_inputs, USER_TO_MISSION_MAPS['Kepler']), 'Kepler'), k_feats)
    p_k = kepler_model.predict_proba(X_k)[:, 1][0]
    
    X_t = create_aligned_dataframe(engineer_features(create_mapped_dataframe(norm_inputs, USER_TO_MISSION_MAPS['TESS']), 'TESS'), t_feats)
    p_t = tess_model.predict_proba(X_t)[:, 1][0]
    
    X_k2 = create_aligned_dataframe(engineer_features(create_mapped_dataframe(norm_inputs, USER_TO_MISSION_MAPS['K2']), 'K2'), k2_feats)
    p_k2 = k2_model.predict_proba(X_k2)[:, 1][0]
    
    # --- 2. Create the input vector for the meta-model ---
    # This is a "dense" vector with all three predictions
    X_meta_input = np.array([p_k, p_t, p_k2]).reshape(1, -1)

    # --- 3. Get the Final Stacked Prediction from the Meta-Model ---
    final_stacked_proba = meta_model.predict_proba(X_meta_input)[:, 1][0]
    
    return p_k, p_t, p_k2, final_stacked_proba


def load_lightcurve_any(file_path: str):
    """
    Universal loader for Kepler, K2, or TESS light curves.
    Returns a LightCurve object regardless of file type or format.
    """
    logging.info(f"Opening FITS light curve file: {file_path}")
    lcfile = lk.open(file_path)

    # Handle multi-sector or multi-quarter collections
    if isinstance(lcfile, lk.LightCurveCollection):
        lc = lcfile.stitch()
        logging.info(f"Loaded LightCurveCollection with {len(lcfile)} segments.")
        return lc

    # Handle container-style objects
    for attr in ["PDCSAP_FLUX", "SAP_FLUX", "LC"]:
        if hasattr(lcfile, attr):
            lc = getattr(lcfile, attr)
            logging.info(f"Loaded {attr} from {type(lcfile).__name__}")
            return lc

    # Direct LightCurve object
    if isinstance(lcfile, lk.LightCurve):
        logging.info("Loaded direct LightCurve object.")
        return lcfile

    raise TypeError(f"Unsupported light curve format for file: {file_path}")


def extract_features_from_lightcurve(file_path: str) -> dict:
    """
    Extracts mission-agnostic transit and stellar features usable for ML exoplanet detection.
    Works for TESS, Kepler, or K2 lightcurves.
    """
    try:
        lc = load_lightcurve_any(file_path)

        # ✅ Basic preprocessing
        processed_lc = lc.remove_nans().normalize()
        flat_lc = processed_lc.flatten(window_length=801)

        # ✅ BLS Transit Search
        logging.info("Running BLS periodogram for transit search...")
        bls = flat_lc.to_periodogram(method="bls", minimum_period=0.5, maximum_period=30)

        period = bls.period_at_max_power.value
        duration = bls.duration_at_max_power.value
        depth_fractional = bls.depth_at_max_power.value
        transit_time = bls.transit_time_at_max_power.value
        snr = getattr(bls, "max_power", np.nan)
        snr = snr.value if hasattr(snr, "value") else snr

        if not np.isfinite(period) or depth_fractional <= 0:
            logging.warning("No significant periodic transit found.")
            return None

        # ✅ Metadata extraction
        meta = getattr(lc, "meta", {})
        mission = meta.get("MISSION", "Unknown")

        # Stellar parameters (fallbacks for different missions)
        stellar_temp = meta.get("TEFF") or meta.get("TSTAR") or meta.get("ST_TEFF")
        stellar_radius = meta.get("RADIUS") or meta.get("RSTAR") or meta.get("ST_RAD")
        stellar_mass = meta.get("MASS") or meta.get("MSTAR") or meta.get("ST_MASS")
        stellar_logg = meta.get("LOGG") or meta.get("ST_LOGG")
        stellar_density = meta.get("DENSITY") or meta.get("ST_DENSITY")
        stellar_mag_tess = meta.get("TESSMAG") or meta.get("KPMAG") or meta.get("KEPMAG")

        # ✅ Derived parameters
        radius_ratio = sqrt(depth_fractional) if depth_fractional > 0 else None
        transit_depth_ppm = depth_fractional * 1_000_000.0
        transit_duration_hr = duration * 24.0

        planetary_radius = None
        if stellar_radius and radius_ratio:
            planetary_radius = stellar_radius * radius_ratio  # in stellar radii
            # convert to Earth radii (1 R☉ = 109.2 R⊕)
            planetary_radius *= 109.2

        # Basic insolation flux approximation (optional)
        insolation_flux = None
        if stellar_temp and stellar_radius and period:
            # simplified luminosity-to-flux scaling (arbitrary normalization)
            insolation_flux = (stellar_radius ** 2) * (stellar_temp / 5778.0) ** 4 / (period ** (4 / 3))

        # Impact parameter is not directly extractable from BLS; set None
        # Same for eccentricity and transit_depth_err
        extracted_features = {
            'radius_ratio': radius_ratio,
            'planetary_radius': planetary_radius,
            'orbital_period': period,
            'insolation_flux': insolation_flux,
            'transit_depth': transit_depth_ppm,
            'transit_duration': transit_duration_hr,
            'impact_parameter': None,
            'transit_midpoint': transit_time,
            'stellar_temp': stellar_temp,
            'stellar_radius': stellar_radius,
            'stellar_density': stellar_density,
            'model_snr': snr,
            'stellar_mag_tess': stellar_mag_tess,
            'orbital_eccentricity': None,
            'stellar_mass': stellar_mass,
            'stellar_logg': stellar_logg,
            'stellar_metallicity': meta.get("FEH") or meta.get("ST_METFE") or None,
            'transit_depth_err': None,
        }

        logging.info(f"[{mission}] Extracted features: {extracted_features}")
        return extracted_features

    except Exception as e:
        logging.error(f"Feature extraction failed for {file_path}: {e}")
        return None
    
# =========================
# 6. Main Execution Block
# =========================

models_tuple = None
features_tuple = None
def prepare():
    global models_tuple, features_tuple
    models_tuple = load_models()
    features_tuple = load_features()
    
def predict(type: Literal['manual', 'raw'], inputs: dict) -> dict:
    """
    Main prediction router. Handles manual, raw light curve, and other input types.
    """
    # Load all models and feature lists once
    global models_tuple, features_tuple

    if not all(models_tuple) or not all(features_tuple):
        logging.error("Prediction halted. Could not load all models or feature lists.")
        return None

    p_k, p_t, p_k2, final_proba = None, None, None, None
    extracted_features = None

    if type == 'manual':
        # Use the user-provided dictionary directly
        p_k, p_t, p_k2, final_proba = predict_from_manual(inputs, models_tuple, features_tuple)
        extracted_features = inputs # For manual input, the "extracted" features are just the inputs themselves

    elif type == 'raw':
        # For raw input, the 'inputs' dict should contain a 'file_path' key
        if 'file_path' not in inputs:
            logging.error("For 'raw' type, the 'inputs' dictionary must contain a 'file_path' key.")
            return None
        
        # 1. Extract features from the light curve file
        extracted_features = extract_features_from_lightcurve(inputs['file_path'])
        
        # 2. If features were found, pass them to the manual prediction function
        if extracted_features:
            p_k, p_t, p_k2, final_proba = predict_from_manual(extracted_features, models_tuple, features_tuple)

    # You can add elif blocks for 'csv' and 'existing' here later

    # Return a comprehensive dictionary with results
    if final_proba is not None:
        return {
            "input_type": type,
            "extracted_features": extracted_features,
            "prediction": {
                "kepler_base_prob": p_k,
                "tess_base_prob": p_t,
                "k2_base_prob": p_k2,
                "final_stacked_prob": final_proba
            }
        }
    else:
        return {
            "input_type": type,
            "status": "Prediction failed. Could not extract features or process input."
        }

if __name__ == "__main__":
    models_tuple = load_models()
    features_tuple = load_features()

    if not all(models_tuple) or not all(features_tuple):
        print("\nPrediction halted. Could not load all models or feature lists.")
    else:
        # Example Input: False Positive
    #     web_input_strong_candidate = {
    # 'radius_ratio': 0.009,                 
    # 'planetary_radius': 1.0,               
    # 'orbital_period': 365.0,               
    # 'insolation_flux': 1.0,                
    # 'transit_depth': 84,                   
    # 'transit_duration': 13.0,              
    # 'impact_parameter': 0.1,               
    # 'transit_midpoint': 2459000.0,
    # 'stellar_temp': 5778.0,                
    # 'stellar_radius': 1.0,                 
    # 'stellar_density': 1.0,                
    # 'model_snr': 15.0,                     
    # 'stellar_mag_tess': 9.0,
    # 'orbital_eccentricity': 0.0167,        
    # 'stellar_mass': 1.0,                   
    # 'stellar_logg': 4.44,                  
    # 'stellar_metallicity': 0.0,
    # 'transit_depth_err': 10.0 
    #     }
    #     p_k, p_t, p_k2, final_proba = predict_from_manual(
    #         web_input_strong_candidate, models_tuple, features_tuple
    #     )
        res = predict(type='raw', inputs={"file_path": "./false.fits"})
        
        print("-" * 50)
        print("MANUAL INPUT PREDICTION (REDUCED STACKED ENSEMBLE)")
        print("-" * 50)
        print("\nBase Model Probabilities (P(Confirmed)):")
        print(res)
        # print(f"  Kepler Reduced: {p_k:.4f}")
        # print(f"  TESS Reduced:   {p_t:.4f}")
        # print(f"  K2 Reduced:     {p_k2:.4f}")
        # print("\nFinal Stacked Output:")
        # print(f"  Meta-Model Final Probability: {final_proba:.4f}")
        print("-" * 50)
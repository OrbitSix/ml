import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import Literal
import lightkurve as lk

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


def extract_features_from_lightcurve(file_path: str) -> dict:
    """
    Loads a light curve file, searches for a transit signal, and extracts
    the necessary features for the prediction model.

    Args:
        file_path: The path to the light curve file (e.g., a .fits file).

    Returns:
        A dictionary of extracted features, or None if an error occurs
        or no significant signal is found.
    """
    try:
        logging.info(f"Loading light curve from: {file_path}")
        # 1. Load the light curve data using lightkurve
        # The `open` command can intelligently handle Kepler, K2, and TESS FITS files.
        lc = lk.open(file_path).get_lightcurve()
        
        # 2. Basic data processing
        # Remove any NaN (Not a Number) values and normalize the flux.
        processed_lc = lc.remove_nans().normalize()
        
        # 3. Flatten the light curve to remove long-term stellar variability.
        # This helps isolate the transit signals.
        flat_lc = processed_lc.flatten(window_length=801) # window_length should be odd and longer than the transit
        
        # 4. Search for a periodic signal using the Box Least Squares (BLS) algorithm.
        logging.info("Searching for periodic transit signal using BLS...")
        periodogram = flat_lc.to_periodogram("bls", minimum_period=0.5, maximum_period=30)
        
        # Find the most significant signal in the periodogram
        best_fit = periodogram.compute_stats(period=periodogram.period_at_max_power)

        # 5. Extract the key transit parameters from the BLS results.
        # BLS provides the most likely period, duration, depth, and transit midpoint.
        period = best_fit['period'][0].value
        duration_days = best_fit['duration'][0].value
        depth = best_fit['depth'][0]
        transit_midpoint = best_fit['transit_time'][0].value
        
        # Check if the signal is significant. A depth of 0 means nothing was found.
        if depth == 0 or not np.isfinite(period):
            logging.warning("No significant periodic signal found in the light curve.")
            return None

        logging.info(f"Signal found! Period: {period:.4f} days, Duration: {duration_days*24:.2f} hours")

        # 6. Extract stellar parameters from the FITS file's metadata header.
        # Lightkurve conveniently stores this in the .meta attribute.
        meta = processed_lc.meta
        
        # Use .get() for safety in case a key is missing
        stellar_temp = meta.get('TEFF')
        stellar_radius = meta.get('RADIUS', meta.get('ST_RAD')) # Check for common alternative keys
        stellar_logg = meta.get('LOGG')
        
        # We can't easily get all features (e.g., impact_parameter, insolation),
        # but our reduced model is trained to handle these missing values.

        # 7. Assemble the final feature dictionary in the format our model expects.
        # CRITICAL: Convert units to match what the model was trained on.
        extracted_features = {
            'orbital_period': period,
            'transit_duration': duration_days * 24.0,  # Convert duration from days to hours
            'transit_depth': depth * 1_000_000.0, # Convert fractional depth to PPM
            'stellar_temp': stellar_temp,
            'stellar_radius': stellar_radius,
            'stellar_logg': stellar_logg,
            'planetary_radius': None, # We estimate this later if needed, but can't get from BLS alone
            'model_snr': periodogram.max_power.value, # Use the BLS power as a proxy for SNR
            # The following will be missing, which is expected by the reduced model:
            'insolation_flux': None,
            'impact_parameter': None,
            'stellar_mass': None,
            'transit_depth_err': None
        }

        logging.info("Successfully extracted features from light curve:")
        print(extracted_features)
        return extracted_features

    except Exception as e:
        logging.error(f"An error occurred during light curve processing: {e}")
        return None
    
# =========================
# 6. Main Execution Block
# =========================


def predict(type: Literal['manual', 'csv', 'existing', 'raw'], inputs: dict) -> dict:
    """
    Main prediction router. Handles manual, raw light curve, and other input types.
    """
    # Load all models and feature lists once
    models_tuple = load_models()
    features_tuple = load_features()

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
        web_input_strong_candidate = {
            'radius_ratio': 0.40,                   # Very large ratio, more like a star than a planet
            'planetary_radius': 45.0,               # Implausibly large for a planet (45 Earth radii)
            'orbital_period': 1.2,                  # Very short period is common for close binaries
            'insolation_flux': 1200.0,              # High flux
            'transit_depth': 150000,                # 15% depth, physically impossible for a planet
            'transit_duration': 2.0,                # Short duration
            'impact_parameter': 0.9,                # Grazing transit, suggests a V-shaped light curve
            'transit_midpoint': 2455100.2,
            'stellar_temp': 4500.0,                 # Cooler M-dwarf or secondary star
            'stellar_radius': 0.8,                  # Smaller primary star
            'stellar_density': 2.5,                 # Higher density
            'model_snr': 500.0,                     # Signal is very strong, but un-planet-like
            'stellar_mag_tess': 12.0,
            'orbital_eccentricity': 0.0,            # Circular orbit
            'stellar_mass': 0.75,                   # Less massive star
            'stellar_logg': 4.6,                    # Higher surface gravity
            'stellar_metallicity': -0.1,
            'transit_depth_err': 200.0              # Low error, so the huge depth is reliable
        }
        p_k, p_t, p_k2, final_proba = predict_from_manual(
            web_input_strong_candidate, models_tuple, features_tuple
        )
        
        print("-" * 50)
        print("MANUAL INPUT PREDICTION (REDUCED STACKED ENSEMBLE)")
        print("-" * 50)
        print("\nBase Model Probabilities (P(Confirmed)):")
        print(f"  Kepler Reduced: {p_k:.4f}")
        print(f"  TESS Reduced:   {p_t:.4f}")
        print(f"  K2 Reduced:     {p_k2:.4f}")
        print("\nFinal Stacked Output:")
        print(f"  Meta-Model Final Probability: {final_proba:.4f}")
        print("-" * 50)
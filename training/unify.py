import pandas as pd
import numpy as np

# --- 1. Define Column Mappings for Unification ---
SENTINEL_VALUE = -9999.0  # For floats (e.g., radius, period)
SENTINEL_FLAG = -1        # For binary flags or scores

# --- UNIFIED COLUMN MAPPING SCHEME ---

# Mapping for Dataset 1 (Kepler Candidate Data)
KEPLER_MAPPING = {
    # Core Unified Columns
    'kepid': 'source_id',
    'kepoi_name': 'candidate_id',
    'kepler_name': 'pl_name',
    'koi_disposition': 'disposition',
    'koi_period': 'pl_orbper_days',
    'koi_period_err1': 'pl_orbper_days_err1',
    'koi_period_err2': 'pl_orbper_days_err2',
    'koi_time0': 'pl_tranmid_bjd',
    'koi_duration': 'pl_trandur_hrs',
    'koi_depth': 'pl_depth_ppm',
    'koi_prad': 'pl_prad_re',
    'koi_insol': 'pl_insol_flux',
    'koi_teq': 'pl_eqt_k',
    'koi_steff': 'st_teff_k',
    'koi_slogg': 'st_logg_cgs',
    'koi_srad': 'st_rad_rsun',
    'ra': 'ra_deg',
    'dec': 'dec_deg',
    
    # Kepler Unique Features (Numerical/Flags)
    'koi_kepmag': 'st_mag_kepler', 
    'koi_score': 'disposition_score',
    'koi_fpflag_nt': 'fp_flag_nt',
    'koi_impact': 'pl_impact',
    'koi_fpflag_ss': 'k_fpflag_ss',
    'koi_fpflag_co': 'k_fpflag_co',
    'koi_fpflag_ec': 'k_fpflag_ec',
    'koi_ror': 'k_ror', 'koi_ror_err1': 'k_ror_err1', 'koi_ror_err2': 'k_ror_err2',
    'koi_srho': 'k_srho', 'koi_srho_err1': 'k_srho_err1', 'koi_srho_err2': 'k_srho_err2',
    'koi_smass': 'k_smass', 'koi_smass_err1': 'k_smass_err1', 'koi_smass_err2': 'k_smass_err2',
    'koi_sage': 'k_sage', 'koi_sage_err1': 'k_sage_err1', 'koi_sage_err2': 'k_sage_err2',
    'koi_max_sngle_ev': 'k_max_sngle_ev', 'koi_max_mult_ev': 'k_max_mult_ev',
    'koi_gmag': 'k_gmag', 
    
    # Kepler Unique Features (Categorical/String)
    'koi_vet_stat': 'k_vet_stat', 'koi_comment': 'k_comment', 'koi_fittype': 'k_fittype', 
    'koi_tce_delivname': 'k_tce_delivname', 'koi_datalink_dvr': 'k_datalink_dvr',
}

# Mapping for Dataset 2 (TESS Candidate Data)
TESS_MAPPING = {
    # Core Unified Columns
    'tid': 'source_id',
    'toi': 'candidate_id',
    'tfopwg_disp': 'disposition',
    'pl_orbper': 'pl_orbper_days',
    'pl_orbpererr1': 'pl_orbper_days_err1',
    'pl_orbpererr2': 'pl_orbper_days_err2',
    'pl_tranmid': 'pl_tranmid_bjd',
    'pl_trandurh': 'pl_trandur_hrs',
    'pl_trandep': 'pl_depth_ppm',
    'pl_rade': 'pl_prad_re',
    'pl_insol': 'pl_insol_flux',
    'pl_eqt': 'pl_eqt_k',
    'st_teff': 'st_teff_k',
    'st_logg': 'st_logg_cgs',
    'st_rad': 'st_rad_rsun',
    'ra': 'ra_deg',
    'dec': 'dec_deg',

    # TESS Unique Features (Numerical/Flags)
    'st_tmag': 'st_mag_tess', 
    'st_pmra': 't_pmra', 'st_pmraerr1': 't_pmra_err1', 'st_pmraerr2': 't_pmra_err2', 
    'st_pmdec': 't_pmdec', 'st_pmdecerr1': 't_pmdec_err1', 'st_pmdecerr2': 't_pmdec_err2', 
    'st_dist': 't_dist', 'st_disterr1': 't_dist_err1', 'st_dist_err2': 't_dist_err2',
    
    # TESS Unique Features (Categorical/String)
    'toipfx': 't_toipfx', 'ctoi_alias': 't_ctoi_alias', 'toi_created': 't_toi_created', 'rowupdate': 't_rowupdate',
}

# Mapping for Dataset 3 (K2 Archive Catalog)
K2_MAPPING = {
    'pl_name': 'pl_name', # Matches
    'disposition': 'disposition', # Matches
    'hostname': 'k2_hostname',
    'k2_name': 'k2_id',
    'tic_id': 'k2_tic_id',
    'gaia_id': 'k2_gaia_id',
    'pl_orbper': 'pl_orbper_days', 
    'pl_orbpererr1': 'pl_orbper_days_err1',
    'pl_orbpererr2': 'pl_orbper_days_err2',
    'pl_rade': 'pl_prad_re',
    'pl_radeerr1': 'pl_prad_re_err1',
    'pl_radeerr2': 'pl_prad_re_err2',
    'pl_trandep': 'pl_depth_ppm',
    'pl_trandeperr1': 'pl_depth_ppm_err1',
    'pl_trandeperr2': 'pl_depth_ppm_err2',
    'pl_trandur': 'pl_trandur_hrs',
    'pl_trandurerr1': 'pl_trandur_hrs_err1',
    'pl_trandurerr2': 'pl_trandur_hrs_err2',
    'pl_insol': 'pl_insol_flux',
    'pl_eqt': 'pl_eqt_k',
    'st_teff': 'st_teff_k',
    'st_tefferr1': 'st_teff_k_err1',
    'st_tefferr2': 'st_teff_k_err2',
    'st_logg': 'st_logg_cgs',
    'st_loggerr1': 'st_logg_cgs_err1',
    'st_loggerr2': 'st_logg_cgs_err2',
    'st_rad': 'st_rad_rsun',
    'st_raderr1': 'st_rad_rsun_err1',
    'st_raderr2': 'st_rad_rsun_err2',
    'ra': 'ra_deg',
    'dec': 'dec_deg',
    
    # K2 Unique Features (Numerical/Flags) - prefixed 'a_' for Archive/K2
    'sy_snum': 'a_sy_snum', 'sy_pnum': 'a_sy_pnum', 'sy_mnum': 'a_sy_mnum', # System counts
    'pl_masse': 'a_pl_masse', 'pl_masseerr1': 'a_pl_masse_err1', 'pl_masseerr2': 'a_pl_masse_err2', # Planet Mass
    'pl_dens': 'a_pl_dens', 'pl_denserr1': 'a_pl_dens_err1', 'pl_denserr2': 'a_pl_dens_err2', # Planet Density
    'pl_orbeccen': 'a_pl_orbeccen', 'pl_orbeccenerr1': 'a_pl_orbeccen_err1', 'pl_orbeccenerr2': 'a_pl_orbeccen_err2', # Eccentricity
    'pl_orbincl': 'a_pl_orbincl', 'pl_orbinclerr1': 'a_pl_orbincl_err1', 'pl_orbinclerr2': 'a_pl_orbincl_err2', # Inclination
    'st_mass': 'a_st_mass', 'st_masserr1': 'a_st_mass_err1', 'st_masserr2': 'a_st_mass_err2', # Stellar Mass (validated)
    'st_age': 'a_st_age', 'st_ageerr1': 'a_st_age_err1', 'st_ageerr2': 'a_st_age_err2', # Stellar Age
    'st_met': 'a_st_met', 'st_meterr1': 'a_st_met_err1', 'st_meterr2': 'a_st_met_err2', # Metallicity
    'st_dens': 'a_st_dens', 'st_denserr1': 'a_st_dens_err1', 'st_denserr2': 'a_st_dens_err2', # Stellar Density
    
    # K2 Unique Features (Categorical/String/Flags)
    'discoverymethod': 'a_discoverymethod', 'disc_year': 'a_disc_year', 'disc_facility': 'a_disc_facility',
    'rv_flag': 'a_rv_flag', 'tran_flag': 'a_tran_flag', 'ima_flag': 'a_ima_flag', # Detection flags
    'st_spectype': 'a_st_spectype',
    'k2_campaigns_num': 'a_k2_campaigns_num', # Specific K2 identifier
}

# --- DEFINE UNIQUE FEATURES AND THEIR SENTINELS (Updated for 3 sources) ---

# Kepler Unique Features (Numerical/Flags)
KEPLER_UNIQUE_FEATS = {
    'st_mag_kepler': SENTINEL_VALUE, 'disposition_score': SENTINEL_FLAG, 'fp_flag_nt': SENTINEL_FLAG,
    'pl_impact': SENTINEL_VALUE, 'k_fpflag_ss': SENTINEL_FLAG, 'k_fpflag_co': SENTINEL_FLAG,
    'k_fpflag_ec': SENTINEL_FLAG, 'k_ror': SENTINEL_VALUE, 'k_ror_err1': SENTINEL_VALUE,
    'k_ror_err2': SENTINEL_VALUE, 'k_srho': SENTINEL_VALUE, 'k_srho_err1': SENTINEL_VALUE,
    'k_srho_err2': SENTINEL_VALUE, 'k_smass': SENTINEL_VALUE, 'k_smass_err1': SENTINEL_VALUE,
    'k_smass_err2': SENTINEL_VALUE, 'k_sage': SENTINEL_VALUE, 'k_sage_err1': SENTINEL_VALUE,
    'k_sage_err2': SENTINEL_VALUE, 'k_max_sngle_ev': SENTINEL_VALUE, 'k_max_mult_ev': SENTINEL_VALUE,
    'k_gmag': SENTINEL_VALUE,
}
# TESS Unique Features (Numerical/Flags)
TESS_UNIQUE_FEATS = {
    'st_mag_tess': SENTINEL_VALUE, 't_pmra': SENTINEL_VALUE, 't_pmra_err1': SENTINEL_VALUE,
    't_pmra_err2': SENTINEL_VALUE, 't_pmdec': SENTINEL_VALUE, 't_pmdec_err1': SENTINEL_VALUE,
    't_pmdec_err2': SENTINEL_VALUE, 't_dist': SENTINEL_VALUE, 't_dist_err1': SENTINEL_VALUE,
    't_dist_err2': SENTINEL_VALUE,
}
# K2 Unique Features (Numerical/Flags) - based on a_ prefix
K2_UNIQUE_FEATS = {
    'a_sy_snum': SENTINEL_FLAG, 'a_sy_pnum': SENTINEL_FLAG, 'a_sy_mnum': SENTINEL_FLAG,
    'a_pl_masse': SENTINEL_VALUE, 'a_pl_masse_err1': SENTINEL_VALUE, 'a_pl_masse_err2': SENTINEL_VALUE,
    'a_pl_dens': SENTINEL_VALUE, 'a_pl_dens_err1': SENTINEL_VALUE, 'a_pl_dens_err2': SENTINEL_VALUE,
    'a_pl_orbeccen': SENTINEL_VALUE, 'a_pl_orbeccen_err1': SENTINEL_VALUE, 'a_pl_orbeccen_err2': SENTINEL_VALUE,
    'a_pl_orbincl': SENTINEL_VALUE, 'a_pl_orbincl_err1': SENTINEL_VALUE, 'a_pl_orbincl_err2': SENTINEL_VALUE,
    'a_st_mass': SENTINEL_VALUE, 'a_st_mass_err1': SENTINEL_VALUE, 'a_st_mass_err2': SENTINEL_VALUE,
    'a_st_age': SENTINEL_VALUE, 'a_st_age_err1': SENTINEL_VALUE, 'a_st_age_err2': SENTINEL_VALUE,
    'a_st_met': SENTINEL_VALUE, 'a_st_met_err1': SENTINEL_VALUE, 'a_st_met_err2': SENTINEL_VALUE,
    'a_st_dens': SENTINEL_VALUE, 'a_st_dens_err1': SENTINEL_VALUE, 'a_st_dens_err2': SENTINEL_VALUE,
    'a_k2_campaigns_num': SENTINEL_FLAG,
}

# --- DEFINE CATEGORICAL/STRING COLUMNS ---
CATEGORICAL_KEPLER_COLS = ['k_vet_stat', 'k_comment', 'k_fittype', 'k_tce_delivname', 'k_datalink_dvr']
CATEGORICAL_TESS_COLS = ['t_toipfx', 't_ctoi_alias', 't_toi_created', 't_rowupdate']
CATEGORICAL_K2_COLS = [
    'k2_hostname', 'k2_id', 'k2_tic_id', 'k2_gaia_id', 'a_discoverymethod',
    'a_disc_year', 'a_disc_facility', 'a_rv_flag', 'a_tran_flag', 'a_ima_flag', 'a_st_spectype'
]

# --- DEFINE COMMON COLUMNS (SHARED FEATURES) ---
COMMON_COLS = [
    'source_id', 'candidate_id', 'pl_name', 'ra_deg', 'dec_deg',
    'disposition',
    # Shared Orbital/Physical Parameters
    'pl_orbper_days', 'pl_orbper_days_err1', 'pl_orbper_days_err2',
    'pl_tranmid_bjd', 'pl_trandur_hrs', 'pl_depth_ppm', 'pl_prad_re',
    'pl_insol_flux', 'pl_eqt_k', 'st_teff_k', 'st_logg_cgs', 'st_rad_rsun',
    # Shared Error Terms
    'pl_prad_re_err1', 'pl_prad_re_err2', 'pl_depth_ppm_err1', 'pl_depth_ppm_err2',
    'pl_trandur_hrs_err1', 'pl_trandur_hrs_err2', 'st_teff_k_err1', 'st_teff_k_err2', 
    'st_logg_cgs_err1', 'st_logg_cgs_err2', 'st_rad_rsun_err1', 'st_rad_rsun_err2', 
]

# Helper function to generate the final list of unified columns
def generate_unified_columns(unique_feats_list, cat_cols_list):
    unified = COMMON_COLS.copy()
    
    # Add all unique features and their indicators
    for unique_feats in unique_feats_list:
        for col in unique_feats.keys():
            unified.append(col)
            unified.append(f'is_{col}_missing')
            
    # Add Categorical/String columns (no indicators)
    for cat_cols in cat_cols_list:
        unified.extend(cat_cols)
        
    return unified

UNIQUE_FEATS_LIST = [KEPLER_UNIQUE_FEATS, TESS_UNIQUE_FEATS, K2_UNIQUE_FEATS]
CATEGORICAL_COLS_LIST = [CATEGORICAL_KEPLER_COLS, CATEGORICAL_TESS_COLS, CATEGORICAL_K2_COLS]
UNIFIED_COLUMNS = generate_unified_columns(UNIQUE_FEATS_LIST, CATEGORICAL_COLS_LIST)


def standardize_disposition(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Standardizes the 'disposition' column to a common set of labels (CONFIRMED, CANDIDATE, FALSE POSITIVE)."""
    if source == 'Kepler':
        mapping = {'CONFIRMED': 'CONFIRMED', 'CANDIDATE': 'CANDIDATE', 'FALSE POSITIVE': 'FALSE POSITIVE'}
    elif source == 'TESS':
        # CP and KP are Confirmed Planet, PC is Candidate. APC/FA/FP are False Positives.
        mapping = {'CP': 'CONFIRMED', 'KP': 'CONFIRMED', 'PC': 'CANDIDATE', 'FP': 'FALSE POSITIVE', 'APC': 'FALSE POSITIVE', 'FA': 'FALSE POSITIVE'}
    elif source == 'K2':
        # K2 Archive catalogs often use a simple string for the final, validated disposition.
        mapping = {'CONFIRMED': 'CONFIRMED', 'CANDIDATE': 'CANDIDATE', 'FALSE POSITIVE': 'FALSE POSITIVE', 
                   'NOT DISPOSITIONED': 'CANDIDATE', 'REJECTED': 'FALSE POSITIVE'}
    else:
        return df

    df['disposition'] = df['disposition'].str.upper().map(mapping).fillna(df['disposition'])
    return df

def load_and_clean_data(file_path: str, mapping: dict, source: str, all_unique_feats: list, all_cat_cols: list) -> pd.DataFrame:
    """Loads a CSV, renames columns, standardizes the target, and adds missing indicators."""
    print(f"Loading and processing {source} data from: {file_path}")
    df = pd.read_csv(file_path, comment='#')  # Handle potential comment lines

    # 1. Rename columns
    df = df.rename(columns=mapping)

    # 2. Add a 'mission_source' column
    df['mission_source'] = source

    # 3. Standardize disposition
    df = standardize_disposition(df, source)

    # 4. Prepare for unification by creating indicator flags and filling NaNs
    
    # Identify unique features for ALL three missions
    kepler_unique_feats = KEPLER_UNIQUE_FEATS
    tess_unique_feats = TESS_UNIQUE_FEATS
    k2_unique_feats = K2_UNIQUE_FEATS
    
    # List of all unique features (numerical/flag) across all missions
    all_numeric_unique_feats = {**kepler_unique_feats, **tess_unique_feats, **k2_unique_feats}
    
    # List of all unique categorical features across all missions
    all_categorical_unique_cols = CATEGORICAL_KEPLER_COLS + CATEGORICAL_TESS_COLS + CATEGORICAL_K2_COLS
    
    # Process all unique numeric/flag features
    for col, sentinel in all_numeric_unique_feats.items():
        # Check if the column is NOT present in the current DataFrame (i.e., it's a feature unique to another mission)
        if col not in df.columns:
            # Add the column with NaN and set its indicator flag to 1 (Missing)
            df[col] = np.nan
            df[f'is_{col}_missing'] = 1
        else:
            # The column is present, so set its indicator flag to 0 (Not Missing)
            df[f'is_{col}_missing'] = 0

    # Process all unique categorical/string features
    for col in all_categorical_unique_cols:
        if col not in df.columns:
            df[col] = np.nan
        
    # 5. Filter and order the final columns
    
    # Add common columns that might be missing for some reason
    for col in COMMON_COLS:
        if col not in df.columns:
            df[col] = np.nan
            
    # Select and order the final columns
    df = df[['mission_source'] + UNIFIED_COLUMNS]

    print(f"Finished processing {source} data. Shape: {df.shape}")
    return df

def unify_and_save_data(kepler_file: str, tess_file: str, k2_file: str, output_file: str):
    """Main function to unify and save the three datasets."""
    
    # Process all three DataFrames
    kepler_df = load_and_clean_data(kepler_file, KEPLER_MAPPING, 'Kepler', UNIQUE_FEATS_LIST, CATEGORICAL_COLS_LIST)
    tess_df = load_and_clean_data(tess_file, TESS_MAPPING, 'TESS', UNIQUE_FEATS_LIST, CATEGORICAL_COLS_LIST)
    k2_df = load_and_clean_data(k2_file, K2_MAPPING, 'K2', UNIQUE_FEATS_LIST, CATEGORICAL_COLS_LIST)

    # Concatenate the three DataFrames
    unified_df = pd.concat([kepler_df, tess_df, k2_df], ignore_index=True)
    
    # Final step: Impute the sentinel values on the full unified dataset
    
    # Collect all numerical/flag columns that need imputation
    all_unique_feats_dict = {**KEPLER_UNIQUE_FEATS, **TESS_UNIQUE_FEATS, **K2_UNIQUE_FEATS}
    
    cols_to_impute_value = [col for col, sentinel in all_unique_feats_dict.items() if sentinel == SENTINEL_VALUE]
    cols_to_impute_flag = [col for col, sentinel in all_unique_feats_dict.items() if sentinel == SENTINEL_FLAG]
    
    # Impute numerical columns
    for col in COMMON_COLS + cols_to_impute_value:
        # Impute common columns and numerical unique columns with the numerical sentinel value
        unified_df[col] = unified_df[col].fillna(SENTINEL_VALUE)
    
    # Impute flag/integer columns
    for col in cols_to_impute_flag:
        # Note: If a column should be integer-like, you might want to convert the sentinel to int
        unified_df[col] = unified_df[col].fillna(SENTINEL_FLAG).astype(float) 
    
    # Note: Categorical/String columns remain as NaN, which is standard for future label encoding/one-hot.

    print(f"\n--- Unification Complete ---")
    print(f"Total rows in unified dataset: {len(unified_df)}")
    print(f"Number of Unified Features (plus source and indicators): {len(unified_df.columns)}")

    # Save the unified DataFrame to a new CSV file
    unified_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved unified dataset to: {output_file}")


if __name__ == '__main__':
    # Ensure these paths match your file locations
    KEPLER_FILE = 'keplar.csv'
    TESS_FILE = 'tess.csv'
    # K2 FILE PATH:
    K2_FILE = 'k2.csv' 
    OUTPUT_FILE = 'unified_exoplanet_catalog_v2.csv'

    try:
        # Note: You still need to run the updated 'generate_mock_data.py' (below) to create the input files if you haven't yet.
        unify_and_save_data(KEPLER_FILE, TESS_FILE, K2_FILE, OUTPUT_FILE)
    except FileNotFoundError as e:
        print(f"\nERROR: File not found. Please ensure all three files exist: '{KEPLER_FILE}', '{TESS_FILE}', and '{K2_FILE}'.")
        print("Hint: Run the 'generate_mock_data.py' script first if you are just testing.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during unification: {e}")

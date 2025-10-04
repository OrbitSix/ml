import pandas as pd
import lightkurve as lk
from astropy.timeseries import LombScargle
from tqdm.auto import tqdm
import numpy as np
import os

# --- CONFIG ---
QUICK_TEST_MODE = True   # 🔹 set False to run full dataset
SAMPLE_SIZE = 200        # 🔹 number of rows to sample in quick test mode


# --- Helper Function ---
def get_periodogram_features(row, quick=False):
    """
    Computes Lomb-Scargle periodogram features for a single target star.
    If quick=True, uses reduced frequency grid for faster testing.
    """
    if pd.isna(row['tid']):
        return pd.Series([np.nan, np.nan, np.nan, np.nan],
                         index=['ls_period', 'ls_power', 'period_discrepancy_ratio', 'fap_ls'])
        
    target_id = f"TIC {int(row['tid'])}"
    try:
        lc_collection = lk.search_lightcurve(
            target_id, mission='TESS', author='SPOC'
        ).download_all(quality_bitmask='default')

        if not lc_collection:
            return pd.Series([np.nan, np.nan, np.nan, np.nan],
                             index=['ls_period', 'ls_power', 'period_discrepancy_ratio', 'fap_ls'])

        lc = lc_collection.stitch().remove_nans().remove_outliers()
        time_val, flux_val = lc.time.value, lc.flux.value

        # --- Periodogram ---
        if quick:
            # faster: narrower freq window + fewer samples
            min_freq, max_freq = 1 / 20.0, 1 / 0.5
            samples_per_peak = 3
        else:
            # full: broader search + more resolution
            min_freq, max_freq = 1 / 35.0, 1 / 0.5
            samples_per_peak = 10

        frequency, power = LombScargle(time_val, flux_val).autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=samples_per_peak
        )

        best_power = np.max(power)
        best_freq = frequency[np.argmax(power)]
        ls_period = 1 / best_freq
        fap = LombScargle(time_val, flux_val).false_alarm_probability(best_power)

        pipeline_period = row['pl_orbper']
        period_discrepancy_ratio = (
            pipeline_period / ls_period
            if ls_period > 0 and pd.notna(pipeline_period) else np.nan
        )

        return pd.Series([ls_period, best_power, period_discrepancy_ratio, fap],
                         index=['ls_period', 'ls_power', 'period_discrepancy_ratio', 'fap_ls'])

    except Exception:
        return pd.Series([np.nan, np.nan, np.nan, np.nan],
                         index=['ls_period', 'ls_power', 'period_discrepancy_ratio', 'fap_ls'])


# --- Main Orchestration Function ---
def process_full_dataset_and_save(full_df, output_csv_path, quick=False):
    """
    Processes the TESS dataframe to add Lomb-Scargle features.
    Supports quick test mode with subset + lightweight config.
    """
    df_to_process = full_df.copy()

    # --- QUICK TEST SUBSAMPLING ---
    if quick:
        df_to_process = df_to_process.sample(n=min(SAMPLE_SIZE, len(df_to_process)), random_state=42)
        print(f"QUICK TEST MODE: running on {len(df_to_process)} randomly sampled rows")

    # --- RESUME CAPABILITY ---
    if os.path.exists(output_csv_path) and not quick:
        print(f"INFO: Found existing file at '{output_csv_path}'. Attempting to resume.")
        try:
            df_processed = pd.read_csv(output_csv_path, index_col=0)
            processed_indices = df_processed.index
            unprocessed_indices = df_to_process.index.difference(processed_indices)

            if len(unprocessed_indices) == 0:
                print("INFO: All rows have already been processed. Loading final data.")
                return df_processed

            print(f"Resuming... {len(processed_indices)} rows processed, {len(unprocessed_indices)} remaining.")
            df_to_process = df_to_process.loc[unprocessed_indices]

        except Exception as e:
            print(f"WARNING: Could not load existing file. Reprocessing all data. Error: {e}")
            df_processed = None
    else:
        if not quick:
            print("INFO: No existing file found. Starting fresh.")
        df_processed = None

    tqdm.pandas(desc=f"Calculating Periodogram Features for {len(df_to_process)} targets")
    new_features = df_to_process.progress_apply(lambda row: get_periodogram_features(row, quick=quick), axis=1)

    df_newly_processed = pd.concat([df_to_process, new_features], axis=1)

    final_df = pd.concat([df_processed, df_newly_processed]) if df_processed is not None else df_newly_processed
    final_df = final_df.sort_index()

    if not quick:  # don’t save in quick mode
        print(f"\nProcessing complete. Saving updated dataset to '{output_csv_path}'...")
        try:
            final_df.to_csv(output_csv_path, index=True)
            print("Save successful!")
        except Exception as e:
            print(f"ERROR: Failed to save the file. Error: {e}")
    else:
        print("\nQUICK TEST MODE: results not saved, returning in-memory dataframe")

    return final_df


# ==============================================================================
# --- HOW TO RUN ---
# ==============================================================================
ORIGINAL_TESS_FILE = '../datasets/tess.csv'
FEATURED_TESS_FILE = '../datasets/tess_full_with_ls_features.csv'

try:
    print(f"Loading original TESS data from '{ORIGINAL_TESS_FILE}'...")
    original_tess_df = pd.read_csv(ORIGINAL_TESS_FILE, comment='#')
    print(f"Successfully loaded {len(original_tess_df)} rows.")
except FileNotFoundError:
    print(f"ERROR: The file '{ORIGINAL_TESS_FILE}' was not found.")
    exit()

featured_tess_df = process_full_dataset_and_save(original_tess_df,
                                                 FEATURED_TESS_FILE,
                                                 quick=QUICK_TEST_MODE)

print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETE!")
if QUICK_TEST_MODE:
    print("Quick test results are in-memory only (not saved).")
else:
    print(f"Enriched dataset saved at: {FEATURED_TESS_FILE}")
print("="*60)

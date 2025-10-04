import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import numpy as np
import logging
import requests
import json
import time
import joblib # NEW: Import joblib for model serialization
import os # NEW: Import os for path handling
from llm import system_prompt

# Set logging level
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =========================
# LLM API CONFIGURATION
# !!! IMPORTANT !!!
# Replace the placeholder API key with your actual Gemini API key.
# =========================
GEMINI_API_KEY = "" 
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
LLM_SAMPLES_TO_VERIFY = 50  # Limit the LLM verification to the first 50 test samples for efficiency

# Define the required JSON output schema for the LLM
LLM_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {
            "type": "INTEGER",
            "description": "Final classification verdict. Use 1 for 'Confirmed Exoplanet' or 'Candidate', and 0 for 'False Positive' or 'Refuted'."
        },
        "reasoning": {
            "type": "STRING",
            "description": "A detailed, technical explanation (3-4 sentences) justifying the verdict based on the provided planetary features and the ML prediction probability."
        }
    },
    "required": ["verdict", "reasoning"]
}

# =========================
# Configuration & File Paths
# =========================
KEPLER_FILE = '../datasets/keplar.csv' 
TESS_FILE = '../datasets/tess.csv'      
K2_FILE = '../datasets/k2.csv'             

# --- NEW: Paths for saving artifacts ---
MODEL_DIR = 'saved_models'
DATA_DIR = 'saved_data'
MASTER_TEST_DATA_PATH = os.path.join(DATA_DIR, 'master_test_data.csv')
TRUE_LABELS_PATH = os.path.join(DATA_DIR, 'y_meta_test.npy')
# ---------------------------------------

RANDOM_SEED = 42
N_SPLITS = 5

TUNED_PARAMS = {
    'feature_fraction': 0.85,
    'bagging_fraction': 0.80,
    'bagging_freq': 1,
    'min_child_samples': 20,
}

LGBM_BASE_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'n_estimators': 500, 'num_leaves': 40,
    'verbose': -1, 'seed': RANDOM_SEED, 'n_jobs': -1 , **TUNED_PARAMS 
}

LGBM_TESS_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.015, 'n_estimators': 1000, 'num_leaves': 61,
    'verbose': -1, 'seed': RANDOM_SEED, 'n_jobs': -1 , **TUNED_PARAMS 
}

LGBM_META_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.05, 'n_estimators': 200, 'num_leaves': 16,
    'verbose': -1, 'seed': RANDOM_SEED, 'n_jobs': -1, **TUNED_PARAMS 
}

# =========================
# 1. Feature Mapping Definitions (Unchanged)
# =========================

def preprocess_kepler(df):
    KEPLER_MAPPING = {'koi_period': 'pl_orbper_days', 'koi_duration': 'pl_dur_hours', 
                      'koi_prad': 'pl_prad_re', 'koi_srho': 'k_srho', 'koi_teq': 'pl_eqt'}
    df = df.rename(columns=KEPLER_MAPPING)
    target_col = 'koi_disposition'
    KEPLER_DROP = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 
                   'koi_fpflag_ec', 'koi_pdisposition', 'koi_vet_stat', 'kepid', 
                   'koi_name', 'kepler_name', 'koi_vet_date', 'koi_disp_prov', 
                   'koi_comment', 'koi_fittype', 'koi_limbdark_mod', 'koi_parm_prov', 
                   'koi_tce_delivname', 'koi_quarters', 'koi_trans_mod', 
                   'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sparprov']
    y = df[target_col].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}).astype(int)
    df_clean = df.drop(columns=[target_col] + KEPLER_DROP, errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    return X, y

def preprocess_tess(df):
    TESS_MAPPING = {'pl_orbper': 'pl_orbper_days', 'pl_trandurh': 'pl_dur_hours', 
                    'pl_rade': 'pl_prad_re', 'st_tmag': 'st_mag_tess', 'pl_eqt': 'pl_eqt'}
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

def preprocess_k2(df):
    K2_MAPPING = {'pl_orbper': 'pl_orbper_days', 'pl_rade': 'pl_prad_re', 
                  'st_teff': 'st_teff_k', 'st_mass': 'st_mass', 
                  'sy_pnum': 'a_sy_pnum', 'pl_eqt': 'pl_eqt'}
    df = df.rename(columns=K2_MAPPING)
    target_col = 'disposition'
    K2_DROP = [
        'pl_controv_flag', 'discoverymethod', 'disc_year', 'disc_refname', 
        'disc_pubdate', 'disc_locale', 'disc_facility', 'disc_telescope', 
        'disc_instrument', 'rv_flag', 'pul_flag', 'ptv_flag', 'tran_flag', 
        'ast_flag', 'obm_flag', 'micro_flag', 'etv_flag', 'ima_flag', 'dkin_flag', 
        'default_flag', 'pl_orbperlim', 'pl_orbsmaxlim', 'pl_radelim', 
        'pl_radjlim', 'pl_masselim', 'pl_massjlim', 'pl_msinielim', 
        'pl_msinijlim', 'pl_cmasselim', 'pl_cmassjlim', 'pl_bmasselim', 
        'pl_bmassjlim', 'pl_denslim', 'pl_orbeccenlim', 'pl_insollim', 
        'pl_eqtlim', 'pl_eqtsymerr', 'pl_orbincllim', 'pl_tranmidlim', 
        'pl_impparlim', 'pl_trandeplim', 'pl_trandurlim', 'pl_ratdorlim', 
        'pl_ratrorlim', 'pl_occdeplim', 'pl_orbtperlim', 'pl_orblperlim', 
        'pl_rvamplim', 'pl_projobliqlim', 'pl_trueobliqlim', 'st_tefflim', 
        'st_radlim', 'st_masslim', 'st_metlim', 'st_lumlim', 'st_logglim', 
        'st_agelim', 'st_denslim', 'st_vsinlim', 'st_rotplim', 'st_radvlim',
        'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 
        'epic_candname', 'hd_name', 'hip_name', 'tic_id', 'gaia_id', 
        'disp_refname', 'st_refname', 'sy_refname', 'soltype', 'pl_refname', 
        'pl_bmassprov', 'pl_tsystemref', 'rastr', 'decstr', 'st_spectype', 
        'rowupdate', 'pl_pubdate', 'releasedate', 'k2_campaigns'
    ]
    y = df[target_col].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0, 'REFUTED': 0}).astype(int)
    df_clean = df.drop(columns=[target_col] + K2_DROP, errors='ignore') 
    X = df_clean.select_dtypes(include=np.number)
    return X, y


# =========================
# 2. Load and Split Data 
# =========================

try:
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

def train_and_generate_oof(X_train, y_train, model_name, n_splits=N_SPLITS, params=LGBM_BASE_PARAMS):
    logging.info(f"\n--- Training Base Model: {model_name} ---")
    oof_preds = np.zeros(y_train.shape[0])
    base_model = lgb.LGBMClassifier(**params)
    pipeline = Pipeline([
        ('oversampler', RandomOverSampler(random_state=RANDOM_SEED)),
        ('classifier', base_model)
    ])
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


# =========================
# 4. Meta-Learner Preparation & Training
# =========================
# --- Training Data Construction ---
oof_k_r = oof_k.reshape(-1, 1)
oof_t_r = oof_t.reshape(-1, 1)
oof_k2_r = oof_k2.reshape(-1, 1)

X_k_meta_train = np.hstack([oof_k_r, np.zeros_like(oof_k_r), np.zeros_like(oof_k_r)])
X_t_meta_train = np.hstack([np.zeros_like(oof_t_r), oof_t_r, np.zeros_like(oof_t_r)])
X_k2_meta_train = np.hstack([np.zeros_like(oof_k2_r), np.zeros_like(oof_k2_r), oof_k2_r])

X_meta_train = np.vstack([X_k_meta_train, X_t_meta_train, X_k2_meta_train])
y_meta_train = np.concatenate([y_k_train, y_t_train, y_k2_train])

# --- Test Data Construction (Base Model Predictions) ---
p_k_test = kepler_model.predict_proba(X_k_test)[:, 1].reshape(-1, 1)
p_t_test = tess_model.predict_proba(X_t_test)[:, 1].reshape(-1, 1)
p_k2_test = k2_model.predict_proba(X_k2_test)[:, 1].reshape(-1, 1)

X_meta_test = np.vstack([
    np.hstack([p_k_test, np.zeros_like(p_k_test), np.zeros_like(p_k_test)]),
    np.hstack([np.zeros_like(p_t_test), p_t_test, np.zeros_like(p_t_test)]),
    np.hstack([np.zeros_like(p_k2_test), np.zeros_like(p_k2_test), p_k2_test])
])
y_meta_test = np.concatenate([y_k_test, y_t_test, y_k2_test])

# --- Train Meta-Learner ---
logging.info("\n--- Training Meta-Learner: LightGBM Classifier ---")
meta_model = lgb.LGBMClassifier(**LGBM_META_PARAMS) 
meta_model.fit(X_meta_train, y_meta_train)

# Final probability prediction on the combined test set (for ML evaluation)
final_stacked_preds = meta_model.predict_proba(X_meta_test)[:, 1]
final_stacked_classes = (final_stacked_preds > 0.5).astype(int)


# =========================
# 5. NEW: Model and Data Persistence
# =========================

def save_artifacts(kepler_model, k2_model, tess_model, meta_model, master_test_df, y_meta_test):
    """Saves all trained models and the prepared test dataset for later LLM verification."""
    
    # 1. Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 2. Save Models using joblib
    joblib.dump(kepler_model, os.path.join(MODEL_DIR, 'kepler_model.pkl'))
    joblib.dump(k2_model, os.path.join(MODEL_DIR, 'k2_model.pkl'))
    joblib.dump(tess_model, os.path.join(MODEL_DIR, 'tess_model.pkl'))
    joblib.dump(meta_model, os.path.join(MODEL_DIR, 'meta_model.pkl'))
    logging.info(f"ML models saved successfully to folder: '{MODEL_DIR}'.")
    
    # 3. Save Master Test Data (DataFrame with features and probabilities)
    master_test_df.to_csv(MASTER_TEST_DATA_PATH, index=False)
    logging.info(f"Master test data saved successfully to file: '{MASTER_TEST_DATA_PATH}'.")

    # 4. Save True Labels (NumPy array)
    np.save(TRUE_LABELS_PATH, y_meta_test)
    logging.info(f"True labels saved successfully to file: '{TRUE_LABELS_PATH}'.")


# =========================
# 6. LLM Integration: Data Consolidation & Persistence Call
# =========================

# --- Feature Set for LLM Prompt (Mapped Columns) ---
PROMPT_FEATURES = ['pl_orbper_days', 'pl_prad_re', 'pl_eqt', 'st_teff']

def create_master_test_df():
    """
    Combines original test features, true labels, mission tag, and stacked predictions.
    
    UPDATED: Now includes ALL original columns from the test sets, relying on
    pd.concat to align features across missions (filling NaNs where columns don't overlap).
    """
    
    def prepare_df(X, y, mission, p_base, p_stacked):
        # Keep ALL columns from the original test set X
        df = pd.DataFrame(X).copy() 
        
        # Append the new derived columns
        df['True_Label'] = y.values
        df['Stacked_Prob'] = p_stacked.flatten()
        df['Base_Prob'] = p_base.flatten()
        df['Mission'] = mission
        return df.reset_index(drop=True)

    # 1. Split final_stacked_preds back into mission segments
    len_k = len(y_k_test)
    len_t = len(y_t_test)
    
    p_s_k = final_stacked_preds[:len_k]
    p_s_t = final_stacked_preds[len_k:len_k + len_t]
    p_s_k2 = final_stacked_preds[len_k + len_t:]

    # 2. Combine 
    df_k = prepare_df(X_k_test, y_k_test, 'Kepler', p_k_test, p_s_k)
    df_t = prepare_df(X_t_test, y_t_test, 'TESS', p_t_test, p_s_t)
    df_k2 = prepare_df(X_k2_test, y_k2_test, 'K2', p_k2_test, p_s_k2)
    
    # pd.concat performs outer join, ensuring all unique columns are present (NaN where missing)
    master_df = pd.concat([df_k, df_t, df_k2], ignore_index=True)
    return master_df

MASTER_TEST_DF = create_master_test_df() 

# --- Call save function here ---
save_artifacts(kepler_model, k2_model, tess_model, meta_model, MASTER_TEST_DF, y_meta_test)


# =========================
# 7. LLM Integration: API Call and Verdict Logic
# =========================

def call_llm_for_verdict(sample_data: pd.Series, max_retries: int = 5) -> tuple[int | None, str]:
    """
    Calls the Gemini API to get a structured verdict and reasoning for a single exoplanet candidate.
    """
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY is not set. Skipping LLM call.")
        return None, "API Key missing."


    features_str = sample_data.to_json()

    user_query = f"""
    --- ML Model's Prediction ---
    {sample_data['Stacked_Prob']:.4f}
    --- Candidate Data ---
    {features_str}
    ---
    """
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": LLM_RESPONSE_SCHEMA,
            "thinkingConfig": {
                "thinkingBudget": -1
            }
        }
    }

    # 3. Make API call with exponential backoff
    for attempt in range(max_retries):
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=20  # Timeout after 20 seconds
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
            
            if json_text:
                # LLM returns a JSON string, parse it
                llm_output = json.loads(json_text)
                verdict = llm_output.get('verdict')
                reasoning = llm_output.get('reasoning')
                
                # Basic validation
                if verdict in [0, 1] and isinstance(reasoning, str) and len(reasoning) > 10:
                    return int(verdict), reasoning
                else:
                    logging.warning(f"Attempt {attempt+1}: LLM response validation failed. Raw: {json_text[:100]}")
                    
            else:
                logging.warning(f"Attempt {attempt+1}: LLM returned no text content.")

        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt+1}: API Request failed: {e}")
        except json.JSONDecodeError:
            logging.error(f"Attempt {attempt+1}: Failed to decode JSON from LLM response.")
        
        # Exponential Backoff
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            time.sleep(wait_time)

    return None, "LLM verification failed after all retries."

# =========================
# 8. LLM Integration: Execution and Evaluation
# =========================

llm_verdicts = []
llm_true_labels = []

logging.info(f"\n--- Starting LLM Verification on first {LLM_SAMPLES_TO_VERIFY} Test Samples ---")

for i in range(min(LLM_SAMPLES_TO_VERIFY, len(MASTER_TEST_DF))):
    sample = MASTER_TEST_DF.iloc[i]
    verdict, reasoning = call_llm_for_verdict(sample)
    
    if verdict is not None:
        llm_verdicts.append(verdict)
        llm_true_labels.append(sample['True_Label'])
        logging.info(f"Sample {i+1}: Stacked Prob={sample['Stacked_Prob']:.4f} | LLM Verdict={verdict} (True={sample['True_Label']})")
        logging.info(reasoning)
    else:
        logging.warning(f"Sample {i+1}: Skipping due to LLM failure.")

# Convert to numpy arrays for evaluation
llm_verdicts = np.array(llm_verdicts)
llm_true_labels = np.array(llm_true_labels)


# =========================
# 9. Display Final Results
# =========================

print("\n#####################################################")
print("### STACKED ENSEMBLE FINAL PERFORMANCE REPORT (V3) ###")
print(f"Total Test Samples Evaluated: {len(y_meta_test)}")
print("-" * 50)
print(f"ML Stacked ROC-AUC Score: {roc_auc_score(y_meta_test, final_stacked_preds):.4f}")
print(f"ML Stacked ACCURACY Score: {accuracy_score(y_meta_test, final_stacked_classes):.4f}")
print("-" * 50)

if len(llm_verdicts) > 0:
    llm_accuracy = accuracy_score(llm_true_labels, llm_verdicts)
    
    print(f"### LLM-VERIFIED EVALUATION ({len(llm_verdicts)} Samples) ###")
    print(f"LLM Verification ACCURACY Score: {llm_accuracy:.4f}")
    print("-" * 50)
    print("Classification Report (LLM Verdicts):")
    print(classification_report(llm_true_labels, llm_verdicts))
else:
    print("LLM Verification was skipped or failed due to API Key/Connectivity issues.")

print("#####################################################")

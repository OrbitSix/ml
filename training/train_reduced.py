import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score 
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import numpy as np
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)

# =========================
# Configuration & File Paths
# =========================
KEPLER_FILE = '../datasets/keplar.csv'
TESS_FILE = '../datasets/tess.csv'
K2_FILE = '../datasets/k2.csv'
SAVE_DIR = '../trained_reduced_models/'
os.makedirs(SAVE_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
LGBM_BASE_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.015, 'n_estimators': 500, 'num_leaves': 40,
    'verbose': -1, 'seed': RANDOM_SEED, 'n_jobs': -1,
    'min_child_samples': 50, 
    'reg_alpha': 2.0, 'reg_lambda': 2.0
}
LGBM_META_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.05, 'n_estimators': 200, 'num_leaves': 16,
    'verbose': -1, 'seed': RANDOM_SEED, 'n_jobs': -1
}

# ====================================================================
# 1. REDUCED CORE FEATURE SETS (FINAL)
# ====================================================================
KEPLER_REDUCED_FEATURES = ['pl_prad_re', 'pl_orbper_days', 'koi_insol', 'koi_depth', 'pl_dur_hours', 'koi_impact', 'koi_steff', 'koi_srad', 'koi_model_snr', 'koi_slogg', 'fe_size_ratio', 'fe_transit_shape', 'fe_certainty_depth', 'fe_temp_proxy']
TESS_REDUCED_FEATURES = ['pl_rade', 'pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_trandeperr1', 'pl_insol', 'st_teff', 'st_rad', 'st_logg', 'fe_size_ratio', 'fe_transit_shape', 'fe_certainty_depth', 'fe_temp_proxy']
K2_REDUCED_FEATURES = ['pl_prad_re', 'pl_orbper_days', 'pl_trandur', 'pl_trandep', 'pl_trandeperr1', 'pl_imppar', 'pl_insol', 'st_teff_k', 'st_rad', 'st_mass', 'st_logg', 'fe_size_ratio', 'fe_transit_shape', 'fe_certainty_depth', 'fe_temp_proxy']

# ====================================================================
# 2. Preprocessing with CORRECT Feature Engineering (FINAL)
# ====================================================================
def preprocess_and_engineer(df, mission):
    # (This function is unchanged from the last correct version)
    epsilon = 1e-6; EARTH_TO_SOLAR_RADIUS = 109.2; df_clean = df.copy()
    cols_to_convert_tess = ['pl_trandeperr1', 'pl_trandeperr2', 'pl_insol', 'st_teff', 'pl_trandep', 'pl_rade', 'st_rad', 'pl_trandurh', 'pl_orbper', 'st_logg']
    if mission == 'Kepler':
        df_clean = df_clean.rename(columns={'koi_period': 'pl_orbper_days', 'koi_duration': 'pl_dur_hours', 'koi_prad': 'pl_prad_re'})
        planet_rad_sr = df_clean['pl_prad_re'] / EARTH_TO_SOLAR_RADIUS; df_clean['fe_size_ratio'] = planet_rad_sr / (df_clean['koi_srad'] + epsilon)
        df_clean['fe_transit_shape'] = df_clean['pl_dur_hours'] / (df_clean['pl_orbper_days'] + epsilon)
        df_clean['fe_certainty_depth'] = df_clean['koi_depth'] / (df_clean['koi_model_snr'] + epsilon)
        df_clean['fe_temp_proxy'] = (df_clean['koi_insol'].clip(lower=0) * df_clean['koi_steff'].clip(lower=0))**0.25
        y = df_clean['koi_disposition'].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}).astype(int); X = df_clean.select_dtypes(include=np.number)
        final_features = [col for col in KEPLER_REDUCED_FEATURES if col in X.columns]
    elif mission == 'TESS':
        for col in cols_to_convert_tess:
            if col in df_clean.columns: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean['pl_trandeperr1'] = df_clean['pl_trandeperr1'].fillna(df_clean['pl_trandeperr2'])
        planet_rad_sr = df_clean['pl_rade'] / EARTH_TO_SOLAR_RADIUS; df_clean['fe_size_ratio'] = planet_rad_sr / (df_clean['st_rad'] + epsilon)
        df_clean['fe_transit_shape'] = df_clean['pl_trandurh'] / (df_clean['pl_orbper'] + epsilon)
        df_clean['fe_certainty_depth'] = df_clean['pl_trandep'] / (df_clean['pl_trandeperr1'] + epsilon)
        df_clean['fe_temp_proxy'] = (df_clean['pl_insol'].clip(lower=0) * df_clean['st_teff'].clip(lower=0))**0.25
        y = df_clean['tfopwg_disp'].replace({'PC': 1, 'KP': 1, 'CP': 1, 'FA': 0, 'FP': 0, 'APC': 0}).astype(int); X = df_clean.select_dtypes(include=np.number)
        final_features = [col for col in TESS_REDUCED_FEATURES if col in X.columns]
    elif mission == 'K2':
        df_clean = df_clean.rename(columns={'pl_orbper': 'pl_orbper_days', 'pl_rade': 'pl_prad_re', 'st_teff': 'st_teff_k'})
        df_clean['pl_trandep'] = df_clean['pl_trandep'] / 100.0; df_clean['pl_trandeperr1'] = df_clean['pl_trandeperr1'] / 100.0
        planet_rad_sr = df_clean['pl_prad_re'] / EARTH_TO_SOLAR_RADIUS; df_clean['fe_size_ratio'] = planet_rad_sr / (df_clean['st_rad'] + epsilon)
        df_clean['fe_transit_shape'] = df_clean['pl_trandur'] / (df_clean['pl_orbper_days'] + epsilon)
        df_clean['fe_certainty_depth'] = df_clean['pl_trandep'] / (df_clean['pl_trandeperr1'] + epsilon)
        df_clean['fe_temp_proxy'] = (df_clean['pl_insol'].clip(lower=0) * df_clean['st_teff_k'].clip(lower=0))**0.25
        y = df_clean['disposition'].replace({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0, 'REFUTED': 0}).astype(int); X = df_clean.select_dtypes(include=np.number)
        final_features = [col for col in K2_REDUCED_FEATURES if col in X.columns]
    X_final = X[final_features].copy(); logging.info(f"{mission} Features Final Set: {len(X_final.columns)} features.")
    return X_final.dropna(axis=1, how='all'), y, final_features

# =========================
# 3. Model Training (with OOF)
# =========================

def train_and_generate_oof(X_train, y_train, X_test, model_name):
    """
    Trains a base model using cross-validation, generates Out-of-Fold (OOF) predictions
    for the training set, and a final prediction for the test set.
    """
    logging.info(f"\n--- Training Base Model: {model_name} ---")
    
    oof_preds = np.zeros(y_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, oof_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_oof = X_train.iloc[train_idx], X_train.iloc[oof_idx]
        y_fold_train, y_fold_oof = y_train.iloc[train_idx], y_train.iloc[oof_idx]

        pipeline = Pipeline([
            ('oversampler', RandomOverSampler(random_state=RANDOM_SEED)),
            ('classifier', lgb.LGBMClassifier(**LGBM_BASE_PARAMS))
        ])
        pipeline.fit(X_fold_train, y_fold_train)
        
        oof_preds[oof_idx] = pipeline.predict_proba(X_fold_oof)[:, 1]
    
    # Train final model on all training data to predict on test data
    final_pipeline = Pipeline([
        ('oversampler', RandomOverSampler(random_state=RANDOM_SEED)),
        ('classifier', lgb.LGBMClassifier(**LGBM_BASE_PARAMS))
    ])
    final_pipeline.fit(X_train, y_train)
    test_preds = final_pipeline.predict_proba(X_test)[:, 1]
    
    return oof_preds, test_preds, final_pipeline

# =========================
# 4. Main Execution (FINAL CORRECTED VERSION)
# =========================
if __name__ == "__main__":
    # --- Step 1: Load and split data ---
    X_k_all, y_k_all, features_k = preprocess_and_engineer(pd.read_csv(KEPLER_FILE, comment='#'), 'Kepler')
    X_k_train, X_k_test, y_k_train, y_k_test = train_test_split(X_k_all, y_k_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_k_all)
    X_t_all, y_t_all, features_t = preprocess_and_engineer(pd.read_csv(TESS_FILE, comment='#'), 'TESS')
    X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_t_all, y_t_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_t_all)
    X_k2_all, y_k2_all, features_k2 = preprocess_and_engineer(pd.read_csv(K2_FILE, comment='#'), 'K2')
    X_k2_train, X_k2_test, y_k2_train, y_k2_test = train_test_split(X_k2_all, y_k2_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_k2_all)

    # --- Step 2: Train base models and get OOF and test predictions ---
    oof_k, test_k, kepler_model = train_and_generate_oof(X_k_train, y_k_train, X_k_test, "Kepler Reduced")
    oof_t, test_t, tess_model = train_and_generate_oof(X_t_train, y_t_train, X_t_test, "TESS Reduced")
    oof_k2, test_k2, k2_model = train_and_generate_oof(X_k2_train, y_k2_train, X_k2_test, "K2 Reduced")

    # --- Step 3: Prepare data for the meta-model using the CORRECT "sparse" method ---
    # Reshape predictions to be column vectors
    oof_k_r = oof_k.reshape(-1, 1)
    oof_t_r = oof_t.reshape(-1, 1)
    oof_k2_r = oof_k2.reshape(-1, 1)

    # Create sparse meta-features for the training set
    X_k_meta_train = np.hstack([oof_k_r, np.zeros_like(oof_k_r), np.zeros_like(oof_k_r)])
    X_t_meta_train = np.hstack([np.zeros_like(oof_t_r), oof_t_r, np.zeros_like(oof_t_r)])
    X_k2_meta_train = np.hstack([np.zeros_like(oof_k2_r), np.zeros_like(oof_k2_r), oof_k2_r])

    # Vertically stack the meta-features and the true labels
    X_meta_train = np.vstack([X_k_meta_train, X_t_meta_train, X_k2_meta_train])
    y_meta_train = pd.concat([y_k_train, y_t_train, y_k2_train])

    # Do the same for the test set predictions
    test_k_r = test_k.reshape(-1, 1)
    test_t_r = test_t.reshape(-1, 1)
    test_k2_r = test_k2.reshape(-1, 1)
    X_k_meta_test = np.hstack([test_k_r, np.zeros_like(test_k_r), np.zeros_like(test_k_r)])
    X_t_meta_test = np.hstack([np.zeros_like(test_t_r), test_t_r, np.zeros_like(test_t_r)])
    X_k2_meta_test = np.hstack([np.zeros_like(test_k2_r), np.zeros_like(test_k2_r), test_k2_r])
    X_meta_test = np.vstack([X_k_meta_test, X_t_meta_test, X_k2_meta_test])
    y_meta_test = pd.concat([y_k_test, y_t_test, y_k2_test])

    # --- Step 4: Train and evaluate the meta-model ---
    logging.info("\n--- Training and Evaluating Meta-Learner ---")
    meta_model = lgb.LGBMClassifier(**LGBM_META_PARAMS)
    meta_model.fit(X_meta_train, y_meta_train)
    
    final_preds = meta_model.predict_proba(X_meta_test)[:, 1]
    final_class_preds = (final_preds > 0.5).astype(int)

    # --- Step 5: Generate the final performance report ---
    auc = roc_auc_score(y_meta_test, final_preds); accuracy = accuracy_score(y_meta_test, final_class_preds)
    report = classification_report(y_meta_test, final_class_preds, target_names=['False Positive', 'Confirmed/Candidate'])
    print("\n\n" + "#"*70); print("###   FINAL STACKED ENSEMBLE PERFORMANCE ON COMBINED TEST SET   ###"); print("#"*70)
    print(f"Total Test Samples: {len(y_meta_test)}")
    print(f"\nROC-AUC Score of Stacked System: {auc:.4f}"); print(f"Accuracy Score of Stacked System: {accuracy:.4f}")
    print("\nCombined Classification Report:"); print(report); print("#"*70)

    # --- Step 6: Train final models on ALL data for deployment ---
    logging.info("\n--- Training FINAL production models on ALL data ---")
    k_prod_model = Pipeline([('o', RandomOverSampler(random_state=RANDOM_SEED)), ('c', lgb.LGBMClassifier(**LGBM_BASE_PARAMS))]).fit(X_k_all, y_k_all)
    t_prod_model = Pipeline([('o', RandomOverSampler(random_state=RANDOM_SEED)), ('c', lgb.LGBMClassifier(**LGBM_BASE_PARAMS))]).fit(X_t_all, y_t_all)
    k2_prod_model = Pipeline([('o', RandomOverSampler(random_state=RANDOM_SEED)), ('c', lgb.LGBMClassifier(**LGBM_BASE_PARAMS))]).fit(X_k2_all, y_k2_all)
    
    # We do not train a final meta-model in this setup for saving. The three base models are what you need.
    # The stacking logic is applied at prediction time.

    # --- Step 7: Save the production-ready models ---
    logging.info("\n--- Saving All Production Models and Feature Lists ---")
    models_to_save = {
        'kepler_reduced_model.pkl': k_prod_model,
        'tess_reduced_model.pkl': t_prod_model,
        'k2_reduced_model.pkl': k2_prod_model,
        'meta_reduced_model.pkl': meta_model # Save the evaluated meta-model
    }
    for filename, model in models_to_save.items():
        path = os.path.join(SAVE_DIR, filename)
        with open(path, 'wb') as f: pickle.dump(model, f)
        logging.info(f"Saved: {path}")
    features_path = os.path.join(SAVE_DIR, 'reduced_model_features.pkl')
    with open(features_path, 'wb') as f: pickle.dump({'kepler_features': features_k, 'tess_features': features_t, 'k2_features': features_k2}, f)
    logging.info(f"Saved: {features_path}")
    print("\nReduced model training and evaluation complete.")
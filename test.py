# pip install pandas lightgbm scikit-learn

import pandas as pd
import lightgbm as lgb

# =========================
# Load trained LightGBM model
# =========================
model = lgb.Booster(model_file='exoplanet_nasa_lgb.txt')

# =========================
# Load CSV (same dataset or subset)
# =========================
df = pd.read_csv('cleaned_keplar.csv', comment='#')

# Define label column
label_col = 'koi_pdisposition'

# Map to binary (only needed for evaluation, optional here)
df[label_col] = df[label_col].replace({
    'CONFIRMED': 1,
    'CANDIDATE': 1,
    'FALSE POSITIVE': 0
})

# Drop rows without label (optional for testing)
df = df.dropna(subset=[label_col])

# Select numeric features (same as used for training)
exclude_cols = [
    label_col, 'kepid', 'kepoi_name', 'kepler_name', 
    'koi_disposition', 'koi_pdisposition', 'koi_score',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'rowid'
]
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
features = [c for c in numeric_cols if c not in exclude_cols]

# Fill missing values
X_test = df[features].fillna(df[features].median())

# =========================
# Predict probabilities
# =========================
y_pred_prob = model.predict(X_test)

# Convert to binary predictions with 0.5 threshold
y_pred = (y_pred_prob > 0.6).astype(int)

# =========================
# Show predictions for first 10 candidates
# =========================
for i in range(20):
    print(f"Candidate {i+1}: Probability planet = {y_pred_prob[i]:.3f}, Predicted label = {y_pred[i]}")

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import logging
import requests
import json
import time
import os
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
LLM_SAMPLES_TO_VERIFY = 50  # Matches the limit used during the saving process

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
            "description": "Reasoning behind the verdict"
        }
    },
    "required": ["verdict", "reasoning"]
}

# =========================
# Configuration & File Paths (Must match paths used in llm_enhanced_stacking.py)
# =========================
DATA_DIR = 'saved_data'
MASTER_TEST_DATA_PATH = os.path.join(DATA_DIR, 'master_test_data.csv')
TRUE_LABELS_PATH = os.path.join(DATA_DIR, 'y_meta_test.npy')


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
                timeout=60  # Timeout after 20 seconds
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

def run_llm_test():
    """Loads saved data, runs LLM verification, and prints performance report."""
    
    # 1. Load Data
    if not os.path.exists(MASTER_TEST_DATA_PATH):
        logging.error(f"Required file not found: {MASTER_TEST_DATA_PATH}. Please run 'llm_enhanced_stacking.py' first.")
        return
    
    try:
        master_test_df = pd.read_csv(MASTER_TEST_DATA_PATH)
    except Exception as e:
        logging.error(f"Error loading master test data: {e}")
        return

    logging.info(f"Loaded {len(master_test_df)} samples from master test data.")
    
    # Select the subset for LLM verification
    test_subset = master_test_df.head(LLM_SAMPLES_TO_VERIFY)

    llm_verdicts = []
    llm_true_labels = []

    logging.info(f"\n--- Starting LLM Verification on {len(test_subset)} Test Samples ---")

    # 2. Execute LLM Verification
    for i in range(len(test_subset)):
        sample = test_subset.iloc[i]
        
        # Display progress
        print(f"[{i+1}/{len(test_subset)}] Processing Sample (True Label: {sample['True_Label']}, ML Prob: {sample['Stacked_Prob']:.4f})...", end='\r')
        
        verdict, reasoning = call_llm_for_verdict(sample)
        
        if verdict is not None:
            llm_verdicts.append(verdict)
            llm_true_labels.append(sample['True_Label'])
        else:
            logging.warning(f"Sample {i+1}: Skipping due to LLM failure or validation error.")

    # 3. Evaluate and Report
    print("\n" + "="*60)
    print("### LLM VETTING PERFORMANCE REPORT ###")
    print(f"Total Samples Verified by LLM: {len(llm_verdicts)}")
    
    if len(llm_verdicts) > 0:
        llm_verdicts = np.array(llm_verdicts)
        llm_true_labels = np.array(llm_true_labels)
        
        llm_accuracy = accuracy_score(llm_true_labels, llm_verdicts)
        
        print("-" * 60)
        print(f"LLM Verification ACCURACY Score: {llm_accuracy:.4f}")
        print("-" * 60)
        print("Classification Report (LLM Verdicts):")
        print(classification_report(llm_true_labels, llm_verdicts, zero_division=0))
    else:
        print("No samples were successfully verified by the LLM.")
    
    print("="*60)


if __name__ == "__main__":
    run_llm_test()

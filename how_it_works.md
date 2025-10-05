# How It Works

Our solution addresses the challenge of identifying exoplanets by employing a **flexible, dual-model Machine Learning architecture** built upon a sophisticated **Stacked Ensemble** framework.  
This approach leverages the specialized predictive strengths of multiple models—each trained on data from a unique NASA mission—to produce highly robust and accurate final classifications.

The system dynamically selects the appropriate model based on the type of data you provide, ensuring optimal performance for every use case.

---

## 1. A Dual-Model Architecture for Maximum Flexibility

Most research works so far focus on homogeneous dataset and definitive input data for prediction. To handle diverse analysis scenarios with heterogeneous datasets, our platform intelligently deploys one of two pre-trained models:

### A. The Full-Feature Model 💯

This is our **primary, highest-accuracy model**, designed for data-rich inputs. It utilizes a **comprehensive set of dozens of observational features** to achieve maximum predictive power.

* **When it's used:** For **CSV file uploads** and when selecting a candidate from the built-in **mission datasets (Kepler, TESS, K2)**, where a complete feature set is available.

### B. The Reduced-Feature Model ⚡

This is a **streamlined, lightweight version** of our architecture, specifically engineered for scenarios where only essential data is available. It is trained on a curated subset of ~14 of the most impactful features (e.g., planet radius, orbital period, transit depth, stellar temperature).

* **When it's used:** For **manual user input** and **raw light curve file uploads**, where providing a full feature set is impractical.

Both models share the same **Stacked Ensemble** design, ensuring consistent analytical logic and strong predictive performance across different input types.

---

## 2. Data Foundation and Pre-processing

To build robust, unbiased models, we utilized open-source data from three foundational NASA exoplanet missions: **Kepler, TESS, and K2**.

### A. Feature Integrity and Leakage Prevention

Exoplanet datasets often contain metrics, scores, or flags (like `koi_score`, `tfopwg_disp`, or other confidence metrics) derived from prior manual vetting.  
Including such columns would cause **data leakage**, leading the model to mimic human judgment rather than learning to identify exoplanets from physical characteristics alone.

To prevent this, we systematically dropped all columns related to:

* **Identifiers and Metadata** — e.g., star names, IDs, observation dates.  
* **Prior vetting outcomes or disposition scores** — including confidence metrics, flags, or uncertainty indicators.

The remaining features represent only the **raw astrophysical properties** of the star and candidate (e.g., orbital period, stellar radius, temperature, transit depth).

### B. Addressing Class Imbalance

Exoplanet detection is a highly imbalanced classification problem — confirmed exoplanets are vastly outnumbered by false positives.  
To ensure balanced learning, we incorporated **Random Over Sampling (ROS)** within the training pipeline for each base model. This guarantees an equal mix of Exoplanet and False Positive samples, reducing bias toward the majority class.

---

## 3. The Stacked Ensemble Architecture

Our core architecture is a **two-level Stacked Ensemble**, designed for both the Full-Feature and Reduced-Feature models.

### A. Level 1: Specialized Base Models

We trained **three high-performance Light Gradient Boosting Machine (LightGBM)** classifiers, each dedicated to one NASA mission dataset:

| Model                 | Mission Data | Parameter Tuning                                   |
| --------------------- | ------------ | -------------------------------------------------- |
| **Kepler Base Model** | Kepler       | Standard LGBM (500 estimators, 0.03 learning rate) |
| **K2 Base Model**     | K2           | Standard LGBM (500 estimators, 0.03 learning rate) |
| **TESS Base Model**   | TESS         | Custom LGBM (1,000 estimators, 0.015 learning rate) |

By specializing the base models, we allow each to learn the **unique noise characteristics and feature distributions** of its mission’s data.

### B. Training using Cross-Validation (OOF Prediction)

To train the Meta-Learner without overfitting, we applied **Stratified K-Fold Cross-Validation** (5-Fold).  
Each base model was trained multiple times, generating **Out-of-Fold (OOF) predictions**—predictions made on data unseen by that model.  
These OOF scores were concatenated to create a robust input for the second-level Meta-Learner.

### C. Level 2: The LightGBM Meta-Learner

The OOF prediction probabilities from the three base models become the input features for the final **Meta-Learner**, another LightGBM classifier.  
It learns a **non-linear weighting scheme** to optimally combine the base model outputs, dynamically evaluating which base prediction is most trustworthy for each sample.

This design achieved the highest accuracy among all tested configurations.

---

## 4. Model Optimization and Experimental Evolution

Our final Stacked Ensemble was the product of iterative experimentation and benchmarking.

### A. Initial Benchmark: Unified Model

A single LightGBM trained on all missions yielded:

| Metric | Unified Model Performance |
| ------- | ------------------------- |
| **ROC-AUC** | 0.9376 |
| **Accuracy** | 0.87 |

While solid, it struggled to generalize across the distinct mission domains.

### B. Iteration 2: Logistic Regression Stack

We then trained mission-specific models and combined them with a **Logistic Regression Meta-Learner**, improving separability:

| Metric | LR Stacked Ensemble |
| ------- | ------------------ |
| **ROC-AUC** | 0.9511 |

### C. Final Approach: LightGBM Stacked Ensemble (V3)

Replacing the linear meta-layer with a **non-linear LightGBM Meta-Learner** further boosted performance to **ROC-AUC 0.9543**, confirming the advantage of dynamic weighting and mission specialization.

---

## 5. Performance Metrics and Results

Both models were evaluated on a held-out test set of **4,248 candidate samples**.

### A. Intermediate (OOF) Base Model Performance

| Model | OOF ROC-AUC |
| ------ | ------------ |
| **K2 Base Model** | 0.9926 |
| **Kepler Base Model** | 0.9728 |
| **TESS Base Model** | 0.8849 |

### B. Final Stacked Ensemble (Full-Feature Model)

| Metric | Result | Interpretation |
| ------- | ------- | -------------- |
| **ROC-AUC** | 0.9543 | High separability and strong classification confidence |
| **Accuracy** | 0.8981 | Nearly 90% of test samples correctly classified |

**Class-Level Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| ------ | ---------- | ------- | -------- | -------- |
| **0 (False Positive)** | 0.87 | 0.81 | 0.84 | 1381 |
| **1 (Exoplanet/Candidate)** | 0.91 | 0.94 | 0.93 | 2867 |
| **Weighted Average** | 0.90 | 0.90 | 0.90 | 4248 |

High **Recall (0.94)** means the system correctly identifies **94% of true exoplanets**, minimizing missed detections.

### C. Reduced-Feature Model Performance

Despite fewer inputs, it maintains excellent predictive power:

| Metric | Result | Interpretation |
| ------- | ------- | -------------- |
| **ROC-AUC** | 0.9151 | Very good separability using minimal data |
| **Accuracy** | 0.8545 | Over 85% accuracy on limited features |

| Class | Precision | Recall | F1-Score |
| ------ | ---------- | ------- | -------- |
| **0 (False Positive)** | 0.79 | 0.75 | 0.77 |
| **1 (Exoplanet/Candidate)** | 0.88 | 0.91 | 0.89 |
| **Weighted Average** | 0.85 | 0.85 | 0.85 |

---

## 6. Explainability and User Insight (The LLM Layer)

After classification, our system bridges the gap between machine output and scientific reasoning using a **Large Language Model (LLM)**.

### A. Feature-to-Explanation Pipeline

The model’s key feature values, prediction probabilities, and classification result are passed into a structured LLM prompt.

### B. Automated Scientific Reasoning

The LLM acts as a **domain expert**, converting raw ML outputs into natural-language explanations that include:

* **Key Evidence:** Highlighting which stellar or planetary features drove the classification (e.g., “A shallow transit depth and long orbital period indicate an exoplanet-like signature.”)  
* **False Positive Reasoning:** If the result suggests a false positive, it explains likely causes (e.g., “consistent with an eclipsing binary”).  
* **Mission Context:** Adds relevant background about the source mission (Kepler, TESS, or K2).  
* **Mitigating Factors:** Explains why certain possible confounders were discounted.

This LLM layer ensures **transparency, interpretability, and user trust**, turning a numeric classification into a meaningful scientific narrative.

---

## 7. Summary of Key Benefits

* **High Accuracy:** Final ROC-AUC of **0.9543** and accuracy of **90%**, competitive with or exceeding many deep learning approaches.  
* **Lightweight and Efficient:** Built with Classical ML (LightGBM) rather than Deep Learning for rapid inference and training, requiring no specialized hardware—ideal for hackathon time constraint and limited processing power at our level.  
* **Flexible and Scalable:** Modular architecture supports future mission data (e.g., **PLATO**, **ARIEL**) by simply adding new base models to the stack.  
* **User-Transparent:** The LLM layer translates predictions into scientifically grounded, human-readable insights.

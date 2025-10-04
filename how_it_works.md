# How It Works

Our solution addresses the challenge of manually identifying exoplanets by employing a sophisticated, multi-stage Machine Learning architecture known as a **Stacked Ensemble**. This approach leverages the specialized predictive strengths of three models, each trained on data from a unique NASA mission, and combines their expertise to produce highly robust and accurate final classifications.

---

## 1. Data Foundation and Pre-processing

To build a reliable model, we utilized open-source data from three foundational NASA exoplanet missions: **Kepler, TESS, and K2**.

### A. Feature Integrity and Leakage Prevention

Exoplanet datasets often contain metrics, scores, or flags (such as `koi_score` or `tfopwg_disp` confidence metrics) that are the result of previous manual human vetting or analysis. Including these metrics would cause **data leakage**, resulting in a model that merely mimics past human decisions rather than independently identifying exoplanets from raw observations.

We implemented a feature selection process for each dataset, systematically dropping all columns related to:

* **Identifiers and Metadata** (e.g., star names, id, observation dates).
* **Any column indicating a prior vetting outcome**, disposition score, or uncertainty limit flags.

The resulting feature set for each model focuses exclusively on the fundamental, raw physical characteristics (e.g., orbital period, planetary radius, stellar temperature, etc.) of the star and its candidate.

### B. Addressing Class Imbalance

Exoplanet detection is a highly imbalanced classification problem, as confirmed exoplanets are significantly outnumbered by false positives in the training data. To prevent the models from being biased toward the majority "False Positive" class, we incorporated **Random Over Sampling (ROS)** within the training pipeline for each base model. This ensures the models receive a balanced mix of positive (Exoplanet/Candidate) and negative (False Positive) examples during the training process.

---

## 2. The Stacked Ensemble Architecture

Our core design is the **two-level Stacked Ensemble**:

### A. Level 1: Specialized Base Models

We trained three high-performance **Light Gradient Boosting Machine (LightGBM)** classifiers, one dedicated to each mission dataset:

| Model                 | Mission Data | Parameter Tuning                                    |
| --------------------- | ------------ | --------------------------------------------------- |
| **Kepler Base Model** | Kepler       | Standard LGBM (500 estimators, 0.03 learning rate)  |
| **K2 Base Model**     | K2           | Standard LGBM (500 estimators, 0.03 learning rate)  |
| **TESS Base Model**   | TESS         | Custom LGBM (1,000 estimators, 0.015 learning rate) |

By specializing the base models, we allow each one to deeply learn the unique noise characteristics and observational feature patterns specific to its mission, maximizing its predictive power.

### B. Training using Cross-Validation (OOF Prediction)

To train the final layer (the Meta-Learner) without introducing overfitting, we employed **Stratified K-Fold Cross-Validation** to generate **Out-of-Fold (OOF) predictions**:

* Each base model was trained five times (5-Fold CV).
* In each fold, the model predicted on the held-out portion of the training data (the OOF data).
* The final OOF prediction scores were concatenated, ensuring that each data point in the training set received a prediction from a model that had never seen that specific data point during its training.

### C. Level 2: The LightGBM Meta-Learner

The OOF probabilities (prediction scores) from the three base models become the new input features for the final model—the **Meta-Learner**.

We utilize a final non-linear **LightGBM Classifier** as the Meta-Learner. Its task is to learn a non-linear, optimal weighting scheme for the three base model probabilities. This dynamic evaluation ensures the system utilizes the most reliable base prediction based on the specific characteristics of the candidate being analyzed. The output of the Meta-Learner is the final, high-confidence probability score.

---

## 3. Performance Metrics and Results

Our final model was rigorously evaluated on a held-out test set of **4,248 candidate samples** to confirm its ability to generalize accurately.

### A. Intermediate Model Performance (OOF AUC)

The Area Under the Receiver Operating Characteristic (**ROC-AUC**) score measures the model's ability to distinguish between positive (Exoplanet) and negative (False Positive) classes.

| Model                 | OOF AUC Score |
| --------------------- | ------------- |
| **K2 Base Model**     | 0.9926        |
| **Kepler Base Model** | 0.9728        |
| **TESS Base Model**   | 0.8849        |

### B. Final Stacked Ensemble Performance

The Meta-Learner successfully combined these specialized probabilities to achieve excellent overall performance:

| Metric                           | Result | Interpretation                                                                     |
| -------------------------------- | ------ | ---------------------------------------------------------------------------------- |
| **ROC-AUC Score**  | 0.9543 | Indicates a high level of separability and confidence in candidate classification. |
| **ACCURACY Score** | 0.8981 | Nearly 90% of all test samples were correctly classified.                          |

### C. Detailed Classification Report

The classification report provides key metrics, with *Class 1* representing **Confirmed Exoplanets/Candidates** and *Class 0* representing **False Positives**:

| Class                       | Precision | Recall | F1-Score | Support |
| --------------------------- | --------- | ------ | -------- | ------- |
| **0 (False Positive)**      | 0.87      | 0.81   | 0.84     | 1381    |
| **1 (Exoplanet/Candidate)** | 0.91      | 0.94   | 0.93     | 2867    |
| **Weighted Average**        | 0.90      | 0.90   | 0.90     | 4248    |

The high **Recall for Class 1 (0.94)** is a critical result, meaning our model successfully identifies **94% of all true exoplanets/candidates** in the test set. This effectiveness in minimizing missed detections makes the model a highly reliable first-pass filter for astronomical data.

---

## 4. Model Optimization and Approach

Our final Stacked Ensemble methodology was the most successful result of an iterative development process that benchmarked several alternative strategies:

### A. Initial Benchmarks

We began by benchmarking several algorithms on individual mission datasets and quickly identified **LightGBM** as the superior, most efficient classifier for this binary task.

### B. Initial Approach: Unified Dataset

Training a single LightGBM model on a unified dataset (combining all three missions) yielded the following baseline performance:

| Metric            | Unified Model Performance |
| ----------------- | ------------------------- |
| **ROC-AUC Score** | 0.9376                    |
| **Accuracy**      | 0.87                      |

This was a strong starting point, but the single model struggled to optimally handle the unique noise and data structure differences inherent across the three missions.

### C. Second Iteration: Logistic Regression Stacking

To capitalize on specialization, we switched to a stacking approach, where we trained three models on three different datasets and then combined the base model predictions using a simple **Logistic Regression Meta-Learner** (linear combination only):

| Metric                          | LR Stacked Ensemble Performance |
| ------------------------------- | ------------------------------- |
| **Final Stacked ROC-AUC Score** | 0.9511                          |

This confirmed that specialization was critical, but the linear combination was still suboptimal.

### D. Final Approach: LightGBM Stacked Ensemble (V3)

By replacing the linear Meta-Learner with a **non-linear LightGBM Meta-Learner**, we allowed the stack to dynamically evaluate the weights of each prediction and account for their non-linear relationships.

This resulted in the **highest accuracy and AUC** among all our experiments (as detailed in Section 3).

---

## 5. Explainability and User Insight (The LLM Layer)

Following the Stacked Ensemble's classification, the system executes a final, crucial step to bridge the gap between algorithmic prediction and human understanding: generating an automated, comprehensive explanation for the result. This step is powered by a Large Language Model (LLM).

### A. Feature Importance and Prompt Engineering

To generate a meaningful explanation, key feature values, along with the final prediction confidence score and the classification outcome, are passed to the LLM via a specialized prompt.

### B. Automated Scientific Reasoning

The LLM is engineered to act as a **domain expert**. It translates the numerical decision points into natural language scientific reasoning, focusing on:

* **Key Evidence:** Highlighting which stellar or planetary properties drove the classification (e.g., "The small transit depth, consistent with Earth-sized objects, and the long orbital period strongly supported a true exoplanet classification"). If it's a False Positive, LLM also explains which type of False Positive case it is and why.
* **Mitigating Factors:** Explaining why the model discounted potential false positive indicators.
* **Mission Context:** Integrating knowledge of the source mission (Kepler, TESS, or K2) to provide context relevant to that mission's noise characteristics or survey methodology.

This human-readable explanation serves two critical functions: it provides the user (such as an astronomer or citizen scientist) with a better **grasp of the data and the ML prediction**, and it establishes **trust and transparency** in the final automated classification.

---

## 6. Summary of Key Benefits

* **High Accuracy**: Final ROC-AUC score of **0.9543** and accuracy of **90%**, surpasses simpler baselines and competitive with published researches, some of which used DL.
* **Light-weight & Efficient**: Chose **LightGBM over Deep Learning** due to hackathon time constraints and limited resources, ensuring rapid training and inference without specialized hardware.
* **Flexible and Scalable**: Architecture can easily incorporate future mission datasets (e.g., **PLATO, ARIEL**) by adding new specialized base models to the Level 1 stack.
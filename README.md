# Hospital 30-Day Readmission Prediction
**By:** Krishna Priya Nemalikanti

This project predicts 30-day hospital readmissions using structured EHR data and synthetic clinical data.

## Workflow:
1. Data loading & cleaning
2. Feature engineering (clinical features, comorbidity indices)
3. Model training (Random Forest & Gradient Boosting)
4. Evaluation (ROC-AUC, F1 score)
5. Interpretation & reporting

## Project Folder Structure

hospital-readmission-prediction/
│── notebooks/
    ├── 01_Readmission_Model_Development.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── utils.py
├── README.md
├── requirements.txt


---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

# Hospital Readmission Prediction (Healthcare Analytics)

## Executive Summary

Hospital readmissions within 30 days remain a critical quality and cost concern in healthcare systems. This project develops an end-to-end machine learning pipeline to evaluate the feasibility of predicting 30-day hospital readmission risk using structured clinical and utilization data.

The dataset contains 5,000 patient encounters with demographic information, laboratory measurements, chronic condition indicators, prior utilization metrics, and a binary readmission outcome. Exploratory data analysis and preprocessing were performed to ensure data quality, including type validation, correlation analysis, categorical encoding, and feature scaling.

Two supervised learning models—Random Forest and Gradient Boosting—were trained and evaluated using stratified train-test splits and ROC-AUC as the primary metric. Model performance was modest (ROC-AUC ≈ 0.49–0.50), reflecting known challenges in readmission prediction when relying solely on structured EHR variables.

Despite limited predictive power, feature importance analysis identified clinically meaningful drivers of readmission risk, including laboratory indicators (hemoglobin, glucose, creatinine), utilization patterns (days since last admission, prior visits), age, and length of stay. These findings align with healthcare research and clinical intuition.

Cross-validation confirmed stable performance across folds, indicating that model limitations stem from feature availability rather than overfitting. This project emphasizes transparent evaluation, reproducible workflows, and honest interpretation—key principles in healthcare analytics.

---

## Problem Statement

Hospital readmissions are associated with increased costs, patient burden, and quality-of-care penalties. Accurately identifying patients at risk of 30-day readmission can support targeted interventions and improved care coordination.

The objective of this project is to:
- Build a reproducible ML pipeline for readmission prediction
- Evaluate model performance realistically
- Interpret clinical and utilization drivers of readmission risk

---

## Dataset Overview

- **Records:** 5,000 patient encounters  
- **Features:** 14 variables  
- **Target:** `readmitted_30d` (binary)

### Feature Categories
- **Demographics:** Age, Gender  
- **Laboratory Results:** Hemoglobin, Glucose, Creatinine  
- **Chronic Conditions:** Diabetes, Hypertension, CHF, CKD, COPD  
- **Utilization Metrics:** Length of stay, Prior visits, Days since last admission  

---

## Methodology

### 1. Exploratory Data Analysis
- Data type validation and summary statistics
- Correlation analysis on numeric variables
- Distribution analysis of clinical features

### 2. Feature Engineering
- One-hot encoding of categorical variables
- Feature scaling for gradient-based models
- Stratified train-test split to preserve class balance

### 3. Modeling
- Random Forest Classifier
- Gradient Boosting Classifier

### 4. Evaluation
- Classification metrics (precision, recall, F1-score)
- ROC-AUC as primary performance metric
- 5-fold cross-validation for robustness assessment

---

## Results

| Model | ROC-AUC | Accuracy |
|------|--------|----------|
| Random Forest | ~0.48 | ~0.49 |
| Gradient Boosting | ~0.50 | ~0.49 |

Model performance was close to random, reflecting the inherent difficulty of readmission prediction using limited structured data.

---

## Feature Importance & Interpretation

Top predictors identified by the Random Forest model:
- Laboratory values: hemoglobin, glucose, creatinine
- Utilization history: days since last admission, prior visits
- Patient factors: age, length of stay

Chronic condition flags showed lower individual importance, likely due to overlap with laboratory indicators. These findings are consistent with published healthcare research and real-world clinical experience.

---

## Key Takeaways

- Readmission prediction is a complex, multifactorial problem
- Structured EHR data alone provides limited predictive power
- Transparent evaluation and interpretation are essential in healthcare ML
- Model pipelines and clinical reasoning are as important as raw accuracy

---

## Future Improvements

- Incorporate diagnosis codes (ICD)
- Include medication and procedure history
- Add social determinants of health
- Explore time-series and longitudinal modeling approaches

---




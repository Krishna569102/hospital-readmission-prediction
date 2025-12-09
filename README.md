# Hospital 30-Day Readmission Prediction
**Author:** Krishna Priya Nemalikanti

This project predicts 30-day hospital readmissions using structured EHR data and synthetic clinical data.

## Workflow:
1. Data loading & cleaning
2. Feature engineering (clinical features, comorbidity indices)
3. Model training (Random Forest & Gradient Boosting)
4. Evaluation (ROC-AUC, F1 score)
5. Interpretation & reporting

## Folder Structure

hospital-readmission-prediction/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── utils.py
│
└── notebooks/
    ├── 01_Readmission_Model_Development.ipynb


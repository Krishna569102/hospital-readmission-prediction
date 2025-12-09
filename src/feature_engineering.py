import pandas as pd
import numpy as np

def create_los_bucket(df, col="length_of_stay"):
    df["los_bucket"] = pd.cut(
        df[col],
        bins=[0, 2, 5, 10, 30, np.inf],
        labels=["0-2", "3-5", "6-10", "11-30", "30+"]
    )
    return df

def create_age_group(df, age_col="age"):
    df["age_group"] = pd.cut(
        df[age_col],
        bins=[0, 18, 40, 60, 75, np.inf],
        labels=["0-18", "19-40", "41-60", "61-75", "75+"]
    )
    return df

def flag_abnormal_labs(df):
    df["creatinine_high"] = (df["lab_creatinine"] > 1.2).astype(int)
    df["hb_low"] = (df["lab_hgb"] < 12).astype(int)
    df["glucose_high"] = (df["lab_glucose"] > 140).astype(int)
    return df

def calculate_comorbidity_index(df):
    df["comorbidity_count"] = (
        df[["diabetes_flag","hypertension_flag","copd_flag","chf_flag","ckd_flag"]].sum(axis=1)
    )
    return df

def create_prior_utilization_features(df):
    df["high_prior_visits"] = (df["prior_visits"] > 3).astype(int)
    df["recent_admission"] = (df["days_since_last_admission"] < 30).astype(int)
    return df

def one_hot_encode_categoricals(df, drop_first=True):
    categorical_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df

def engineer_features(df):
    df = create_los_bucket(df)
    df = create_age_group(df)
    df = flag_abnormal_labs(df)
    df = calculate_comorbidity_index(df)
    df = create_prior_utilization_features(df)
    df = one_hot_encode_categoricals(df)
    return df

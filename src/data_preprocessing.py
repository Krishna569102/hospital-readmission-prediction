import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include="object").columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df

def remove_outliers(df, cols, method="iqr"):
    for col in cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.where(df[col].between(lower, upper), df[col], np.nan)
    return df

def preprocess(df):
    df = clean_column_names(df)
    df = handle_missing_values(df)
    df = remove_outliers(df, ["length_of_stay", "lab_creatinine"])
    return df

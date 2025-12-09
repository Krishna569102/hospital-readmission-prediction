import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_dataframe(df, path):
    df.to_csv(path, index=False)
    print(f"Saved dataframe: {path}")

def save_plot(fig, filename, folder="reports"):
    ensure_dir(folder)
    full_path = os.path.join(folder, filename)
    fig.savefig(full_path, bbox_inches="tight")
    print(f"Saved plot: {full_path}")

def summarize_split(X_train, X_test, y_train, y_test):
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training target rate:", y_train.mean())
    print("Testing target rate:", y_test.mean())

def log(msg):
    print(f"[INFO] {msg}")

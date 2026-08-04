"""
Loads and validates the raw Kaggle Telco Customer Churn CSV.
Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""
import os
import sys
import pandas as pd

from src import config

# Local dataset path (used only on this development machine, if it exists)
LOCAL_DATASET_PATH = r"D:\Customer Churn Prediction\data\raw\Telco-Customer-Churn Datasets.csv"


def _resolve_default_path() -> str:
    """
    Prefer the local dev machine's path if it exists; otherwise fall back
    to the standard relative path (data/raw/...), which is what the repo
    ships with and what Streamlit Cloud / any other server will have.
    This means the same code works unmodified on your PC AND after deploy —
    no need to remember to switch the path back before pushing to GitHub.
    """
    if os.path.exists(LOCAL_DATASET_PATH):
        return LOCAL_DATASET_PATH
    return config.RAW_DATA_PATH


def load_raw_data(path: str = None) -> pd.DataFrame:
    path = path or _resolve_default_path()

    if not os.path.exists(path):
        sys.exit(
            f"\n[ERROR] Raw dataset not found at: {path}\n"
            "Download it from Kaggle first:\n"
            "  kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip\n"
            "or manually place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/\n"
        )

    df = pd.read_csv(path)

    expected_cols = {config.TARGET_COL, config.ID_COL, *config.NUMERIC_COLS, *config.CATEGORICAL_COLS}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    print(f"[data_loader] Loaded {df.shape[0]} rows, {df.shape[1]} columns from {os.path.basename(path)}")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(df[config.TARGET_COL].value_counts(normalize=True))
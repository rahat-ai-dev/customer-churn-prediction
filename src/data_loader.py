"""
Loads and validates the raw Kaggle Telco Customer Churn CSV.
Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""
import os
import sys
import pandas as pd

from src import config


def load_raw_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
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

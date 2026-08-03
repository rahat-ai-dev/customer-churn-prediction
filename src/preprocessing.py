"""
Cleaning + encoding + scaling + train/test split.

Known quirks of this specific Kaggle dataset that we handle explicitly:
  - `TotalCharges` is loaded as an object dtype because ~11 rows contain
    blank strings for brand-new customers (tenure == 0). We coerce to
    numeric and impute those with 0.
  - `SeniorCitizen` is already 0/1 int but semantically categorical.
  - `customerID` is a unique identifier, dropped before modeling.
"""
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges has blank strings for customers with 0 tenure -> coerce + impute
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Normalize target to binary int
    df[config.TARGET_COL] = df[config.TARGET_COL].map({"Yes": 1, "No": 0})

    # SeniorCitizen is 0/1 already; treat as categorical string for the OHE step
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df = df.drop(columns=[config.ID_COL])
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: scale numeric, one-hot encode categorical."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_COLS),
            ("cat", categorical_pipeline, config.CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def split_and_transform(df: pd.DataFrame):
    """Split raw-cleaned df into train/test, fit preprocessor on train only."""
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train_df = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names, index=X_test.index)

    # persist artifacts needed at inference time
    joblib.dump(preprocessor, config.PREPROCESSOR_ARTIFACT_PATH)
    with open(config.FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    return X_train_df, X_test_df, y_train, y_test


def run_preprocessing_pipeline(df: pd.DataFrame):
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = split_and_transform(df_clean)

    X_train.to_csv(config.X_TRAIN_PATH, index=False)
    X_test.to_csv(config.X_TEST_PATH, index=False)
    y_train.to_csv(config.Y_TRAIN_PATH, index=False)
    y_test.to_csv(config.Y_TEST_PATH, index=False)

    print(f"[preprocessing] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[preprocessing] Churn rate — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df = load_raw_data()
    run_preprocessing_pipeline(df)

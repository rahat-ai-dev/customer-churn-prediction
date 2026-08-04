"""
Loads the saved preprocessor + ensemble model and runs inference on new,
raw (unprocessed) customer records — e.g. from an API request.
"""
import json
import os
import sys

import joblib
import pandas as pd

# allow running this file directly (`python src/predict.py`) as well as
# as a module (`python -m src.predict`)
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.feature_engineering import add_engineered_features

_model = None
_preprocessor = None
_feature_names = None


def _load_artifacts():
    global _model, _preprocessor, _feature_names
    if _model is None:
        _model = joblib.load(config.MODEL_ARTIFACT_PATH)
    if _preprocessor is None:
        _preprocessor = joblib.load(config.PREPROCESSOR_ARTIFACT_PATH)
    if _feature_names is None:
        with open(config.FEATURE_NAMES_PATH) as f:
            _feature_names = json.load(f)
    return _model, _preprocessor, _feature_names


def predict_churn(raw_records: list[dict]) -> list[dict]:
    """
    raw_records: list of dicts with the SAME raw columns as the original
    Kaggle CSV (minus customerID/Churn), e.g.:
        {"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", ...,
         "tenure": 12, "MonthlyCharges": 70.35, "TotalCharges": 845.5}
    Returns list of {"churn_probability": float, "churn_prediction": "Yes"/"No"}
    """
    model, preprocessor, feature_names = _load_artifacts()

    df = pd.DataFrame(raw_records)

    # reuse the same cleaning path, but skip target mapping (no target at inference)
    if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df = add_engineered_features(df)

    X = preprocessor.transform(df)
    X_df = pd.DataFrame(X, columns=feature_names)

    probs = model.predict_proba(X_df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return [
        {"churn_probability": round(float(p), 4), "churn_prediction": "Yes" if pred == 1 else "No"}
        for p, pred in zip(probs, preds)
    ]


if __name__ == "__main__":
    sample = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    }
    print(predict_churn([sample]))

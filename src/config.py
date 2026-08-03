"""
Central configuration: paths, constants, hyperparameters.
Keeping these in one place means every other module just does
`from src.config import X` instead of hardcoding strings everywhere.
"""
import os

# ---------------------------------------------------------------- paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

X_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")

MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "churn_ensemble.pkl")
PREPROCESSOR_ARTIFACT_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
METRICS_REPORT_PATH = os.path.join(MODELS_DIR, "metrics_report.json")

# ------------------------------------------------------------- constants ---
TARGET_COL = "Churn"
ID_COL = "customerID"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns as they appear in the raw Kaggle Telco Churn CSV
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# ---------------------------------------------------------- model params ---
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

LGBM_PARAMS = dict(
    n_estimators=400,
    max_depth=-1,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

LOGREG_PARAMS = dict(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

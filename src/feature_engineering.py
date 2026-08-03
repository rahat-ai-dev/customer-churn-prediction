"""
Domain-driven feature engineering, applied after cleaning and before
train/test split + encoding. These features consistently show up as
high-importance in churn models on this dataset.
"""
import pandas as pd

from src import config

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Tenure buckets — churn risk is highly non-linear w.r.t. raw tenure
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"],
    ).astype(str)

    # 2. Count of add-on services subscribed (proxy for "stickiness")
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)

    # 3. Average monthly spend over tenure (catches recent price hikes)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # 4. Contract risk flag — month-to-month is the single strongest churn driver
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # 5. Has any streaming service
    df["has_streaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    return df


def register_engineered_columns():
    """
    Call once to extend config's NUMERIC_COLS / CATEGORICAL_COLS so the
    preprocessing ColumnTransformer picks up the new features automatically.
    """
    new_numeric = ["num_services", "avg_monthly_spend", "is_month_to_month", "has_streaming"]
    new_categorical = ["tenure_bucket"]

    for col in new_numeric:
        if col not in config.NUMERIC_COLS:
            config.NUMERIC_COLS.append(col)
    for col in new_categorical:
        if col not in config.CATEGORICAL_COLS:
            config.CATEGORICAL_COLS.append(col)


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data

    df = load_raw_data()
    df = clean_data(df)
    df = add_engineered_features(df)
    print(df[["tenure_bucket", "num_services", "avg_monthly_spend", "is_month_to_month"]].head())

"""
Evaluation utilities: computes standard classification metrics for one or
many models and prints a clean comparison table.
"""
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src import config


def evaluate_model(model, X_test, y_test, name: str = "model") -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def compare_models(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results).drop(columns=["confusion_matrix"])
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return df


def print_report(results: list):
    df = compare_models(results)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON  (sorted by ROC-AUC)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    best = df.iloc[0]
    print(f"\nBest model: {best['model']}  (ROC-AUC = {best['roc_auc']})\n")


def save_report(results: list, path: str = config.METRICS_REPORT_PATH):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate] Metrics report saved to {path}")

"""
Trains all base learners + both ensemble strategies, evaluates each on the
held-out test set, and persists the best-performing model to disk.
"""
import joblib
import pandas as pd

from src import config
from src.evaluate import evaluate_model, print_report, save_report
from src.models import build_stacking_ensemble, build_voting_ensemble, get_base_models


def load_split_data():
    X_train = pd.read_csv(config.X_TRAIN_PATH)
    X_test = pd.read_csv(config.X_TEST_PATH)
    y_train = pd.read_csv(config.Y_TRAIN_PATH).squeeze("columns")
    y_test = pd.read_csv(config.Y_TEST_PATH).squeeze("columns")
    return X_train, X_test, y_train, y_test


def train_all_models(X_train, y_train):
    """Fits base models + both ensembles. Returns dict of fitted models."""
    fitted = {}

    base_models = get_base_models()
    for name, model in base_models.items():
        print(f"[train] Fitting {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model

    print("[train] Fitting voting ensemble...")
    voting = build_voting_ensemble(get_base_models())
    voting.fit(X_train, y_train)
    fitted["voting_ensemble"] = voting

    print("[train] Fitting stacking ensemble (5-fold CV internally, this takes longest)...")
    stacking = build_stacking_ensemble(get_base_models())
    stacking.fit(X_train, y_train)
    fitted["stacking_ensemble"] = stacking

    return fitted


def run_training_pipeline():
    X_train, X_test, y_train, y_test = load_split_data()

    fitted_models = train_all_models(X_train, y_train)

    results = []
    for name, model in fitted_models.items():
        metrics = evaluate_model(model, X_test, y_test, name=name)
        results.append(metrics)

    print_report(results)
    save_report(results)

    # pick + persist best by ROC-AUC
    best_result = max(results, key=lambda r: r["roc_auc"])
    best_model = fitted_models[best_result["model"]]

    joblib.dump(best_model, config.MODEL_ARTIFACT_PATH)
    print(f"[train] Saved best model ({best_result['model']}) to {config.MODEL_ARTIFACT_PATH}")

    return fitted_models, results


if __name__ == "__main__":
    run_training_pipeline()

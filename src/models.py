"""
Base learners + two ensemble strategies:
  - Soft Voting Classifier  (averages predicted probabilities)
  - Stacking Classifier     (meta-learner trained on base learners' outputs)
"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src import config


def get_base_models():
    """Returns dict of {name: fresh estimator} for the base learners."""
    return {
        "logistic_regression": LogisticRegression(**config.LOGREG_PARAMS),
        "random_forest": RandomForestClassifier(**config.RF_PARAMS),
        "xgboost": XGBClassifier(**config.XGB_PARAMS),
        "lightgbm": LGBMClassifier(**config.LGBM_PARAMS),
    }


def build_voting_ensemble(base_models: dict) -> VotingClassifier:
    """Soft voting: average predicted class probabilities across all base models."""
    estimators = [(name, model) for name, model in base_models.items()]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def build_stacking_ensemble(base_models: dict) -> StackingClassifier:
    """
    Stacking: base learners' out-of-fold predictions feed into a
    Logistic Regression meta-learner, which learns how to best combine them.
    """
    estimators = [(name, model) for name, model in base_models.items() if name != "logistic_regression"]
    meta_learner = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

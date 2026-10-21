"""Construction, training and validation-based selection of the seven classifiers."""
from __future__ import annotations

import re

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .evaluation import compute_metrics, positive_scores

MODEL_NAMES = [
    "Gradient Boosting",
    "Decision Tree",
    "Random Forest",
    "Support Vector Classifier",
    "Gaussian Naive Bayes",
    "K-Nearest Neighbors",
    "Logistic Regression",
]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_models(cfg: dict, seed: int = 42) -> dict:
    """Unfitted estimators. SVM/KNN/LR are Pipelines (StandardScaler fitted on train only)."""
    m = cfg["models"]
    return {
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed, **m["gradient_boosting"]),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, **m["decision_tree"]),
        "Random Forest": RandomForestClassifier(random_state=seed, **m["random_forest"]),
        "Support Vector Classifier": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(random_state=seed, **m["svc"]))]
        ),
        "Gaussian Naive Bayes": GaussianNB(**m["gaussian_nb"]),
        "K-Nearest Neighbors": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(**m["knn"]))]
        ),
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=seed, **m["logistic_regression"]))]
        ),
    }


def fit_models(models: dict, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> dict:
    import time

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        if verbose:
            print(f"  trained {name:<28s} ({time.time() - t0:5.1f}s)")
    return models


def validation_table(models: dict, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """name -> metrics dict on the validation split."""
    out = {}
    for name, model in models.items():
        out[name] = compute_metrics(y_val, model.predict(X_val), positive_scores(model, X_val))
    return out


def rank_models(val_metrics: dict) -> list[str]:
    """Names sorted by validation macro F1 (desc), validation accuracy (desc), then name (asc)."""
    return sorted(
        val_metrics,
        key=lambda n: (-val_metrics[n]["macro_f1"], -val_metrics[n]["accuracy"], n),
    )


def select_top_k(val_metrics: dict, k: int = 3) -> list[str]:
    return rank_models(val_metrics)[:k]


def train_and_select(X_train, y_train, X_val, y_val, cfg: dict, seed: int = 42, verbose: bool = True) -> dict:
    """Fit the 7 models on train, select the top-k and the primary ensemble on VALIDATION only.

    The test split is never an argument of this function.
    """
    from .ensemble import AVERAGING_NAME, MAJORITY_NAME, MajorityVotingEnsemble, ProbabilityAveragingEnsemble

    models = fit_models(build_models(cfg, seed), X_train, y_train, verbose=verbose)
    val_metrics = validation_table(models, X_val, y_val)
    ranking = rank_models(val_metrics)
    selected = ranking[: cfg.get("ensemble", {}).get("top_k", 3)]

    members = {n: models[n] for n in selected}
    ensembles = {
        MAJORITY_NAME: MajorityVotingEnsemble(members),
        AVERAGING_NAME: ProbabilityAveragingEnsemble(members),
    }
    for name, ens in ensembles.items():
        val_metrics[name] = compute_metrics(y_val, ens.predict(X_val), positive_scores(ens, X_val))

    primary = sorted(
        ensembles, key=lambda n: (-val_metrics[n]["macro_f1"], -val_metrics[n]["accuracy"], n)
    )[0]
    return {
        "models": models,
        "ensembles": ensembles,
        "val_metrics": val_metrics,
        "ranking": ranking,
        "selected": selected,
        "primary_ensemble": primary,
    }

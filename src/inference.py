"""Saving / loading of model artifacts and single-headline inference.

Inference uses the same ``preprocess`` and ``extract_features`` code as training, and the feature
order is checked against the order stored in ``models/metadata.json``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np

from . import LABELS
from .ensemble import AVERAGING_NAME, MAJORITY_NAME, MajorityVotingEnsemble, ProbabilityAveragingEnsemble
from .features import FEATURE_NAMES, extract_features
from .resources import resource_fingerprint
from .train_models import slugify

METADATA_FILE = "metadata.json"


def save_artifacts(models: dict, metadata: dict, models_dir: str | Path) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, models_dir / f"{slugify(name)}.joblib", compress=3)
    with open(models_dir / METADATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def load_artifacts(models_dir: str | Path) -> tuple[dict, dict, dict]:
    """Return (metadata, individual_models, ensembles). Raises FileNotFoundError with guidance."""
    models_dir = Path(models_dir)
    meta_path = models_dir / METADATA_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {meta_path}. Train first with: "
            "python train.py --data dataset/Sarcasm_Headlines_Dataset_v2.json"
        )
    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    if metadata["feature_names"] != FEATURE_NAMES:
        raise RuntimeError(
            "Feature order in the saved metadata differs from src.features.FEATURE_NAMES; "
            "retrain the models."
        )
    models = {}
    for name in metadata["model_names"]:
        path = models_dir / f"{slugify(name)}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}. Re-run train.py.")
        models[name] = joblib.load(path)
    members = {n: models[n] for n in metadata["selected_models"]}
    ensembles = {
        MAJORITY_NAME: MajorityVotingEnsemble(members),
        AVERAGING_NAME: ProbabilityAveragingEnsemble(members),
    }
    return metadata, models, ensembles


class SarcasmPredictor:
    """Loads the saved artifacts once and classifies headlines."""

    def __init__(self, models_dir: str | Path = "models"):
        self.metadata, self.models, self.ensembles = load_artifacts(models_dir)
        self.all_models = {**self.ensembles, **self.models}
        self.primary = self.metadata["primary_ensemble"]
        saved = self.metadata.get("resource_fingerprint")
        if saved and saved != resource_fingerprint():
            warnings.warn(
                "Linguistic resources differ from those used at training time; predictions may "
                "not match training-time features. Retrain to be safe."
            )

    @property
    def model_names(self) -> list[str]:
        return [self.primary] + [n for n in self.all_models if n != self.primary]

    def predict(self, headline: str, model_name: str | None = None) -> dict:
        if not isinstance(headline, str) or not headline.strip():
            raise ValueError("Please enter a non-empty headline.")
        name = model_name or self.primary
        if name not in self.all_models:
            raise KeyError(f"Unknown model '{name}'. Available: {list(self.all_models)}")
        model = self.all_models[name]
        feats = extract_features(headline)
        X = feats.reshape(1, -1)
        label = int(model.predict(X)[0])

        result = {
            "headline": headline.strip(),
            "model": name,
            "label_id": label,
            "label": LABELS[label],
            "features": dict(zip(FEATURE_NAMES, feats.tolist())),
            "probability_sarcastic": None,
            "confidence": None,
            "votes": None,
        }
        if isinstance(model, MajorityVotingEnsemble):
            member_preds = model.member_predictions(X)[:, 0]
            result["votes"] = {
                member: LABELS[int(p)] for member, p in zip(model.models, member_preds)
            }
            result["vote_share_sarcastic"] = float(member_preds.mean())
        elif hasattr(model, "predict_proba"):
            p = float(model.predict_proba(X)[0, 1])
            result["probability_sarcastic"] = p
            result["confidence"] = p if label == 1 else 1.0 - p
        return result

"""Ensembles over already-fitted classifiers (the three best on validation macro F1).

Both ensembles reuse the fitted individual models as-is (trained on the training split only);
nothing is refit on validation or test data.
"""
from __future__ import annotations

import numpy as np

MAJORITY_NAME = "Ensemble: Majority Voting"
AVERAGING_NAME = "Ensemble: Probability Averaging"


class _BaseEnsemble:
    classes_ = np.array([0, 1])

    def __init__(self, models: dict):
        if len(models) < 2:
            raise ValueError("An ensemble needs at least two fitted models.")
        self.models = dict(models)  # name -> fitted estimator

    def _mean_proba(self, X) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.models.values()], axis=0)


class ProbabilityAveragingEnsemble(_BaseEnsemble):
    """Average ``predict_proba`` of the members and pick the class with the highest mean.

    The averaged probabilities are NOT calibrated (tree ensembles, SVC/Platt scaling, naive
    Bayes etc. are calibrated differently), so they are a score, not a true likelihood.
    Exact 50/50 ties go to class 0 (``argmax`` picks the first maximum).
    """

    name = AVERAGING_NAME

    def predict_proba(self, X) -> np.ndarray:
        return self._mean_proba(X)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class MajorityVotingEnsemble(_BaseEnsemble):
    """Hard voting: each member contributes one class prediction; the most common class wins.

    Tie fallback (only possible with an even number of members; three members can never tie on
    a binary task): the class with the higher *mean predicted probability* wins, and a still
    exact tie goes to class 0. ``predict_proba`` returns the vote shares, which is a score for
    ROC analysis but not a probability estimate.
    """

    name = MAJORITY_NAME

    def member_predictions(self, X) -> np.ndarray:
        return np.vstack([m.predict(X) for m in self.models.values()])  # (n_models, n_samples)

    def predict_proba(self, X) -> np.ndarray:
        votes1 = self.member_predictions(X).mean(axis=0)
        return np.column_stack([1.0 - votes1, votes1])

    def predict(self, X) -> np.ndarray:
        shares = self.predict_proba(X)
        pred = (shares[:, 1] > 0.5).astype(int)
        ties = np.isclose(shares[:, 1], 0.5)
        if ties.any():
            pred[ties] = np.argmax(self._mean_proba(X)[ties], axis=1)
        return pred

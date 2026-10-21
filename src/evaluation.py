"""Metrics and plots computed from actual predictions."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from . import LABELS  # noqa: E402

CLASS_NAMES = [LABELS[0], LABELS[1]]


def compute_metrics(y_true, y_pred, scores=None) -> dict:
    """Accuracy, macro/weighted metrics, per-class metrics, confusion matrix, ROC-AUC.

    ``scores`` are the scores of the positive class (sarcastic) or None.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    roc_auc = None
    if scores is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, scores))
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p_m),
        "macro_recall": float(r_m),
        "macro_f1": float(f_m),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            LABELS[i]: {
                "precision": float(p_c[i]),
                "recall": float(r_c[i]),
                "f1": float(f_c[i]),
                "support": int(s_c[i]),
            }
            for i in (0, 1)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "roc_auc": roc_auc,
    }


def positive_scores(model, X) -> np.ndarray | None:
    """Score of the sarcastic class (probability, or vote share for majority voting)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


# ------------------------------------------------------------------- plots
def plot_confusion_matrix(cm, title: str, path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc_curves(curves: dict, path, title="ROC curves (final test set)") -> None:
    """``curves``: name -> (y_true, scores)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, scores) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_macro_f1_comparison(names, values, path, title="Test macro F1 by model", highlight=None) -> None:
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    colors = ["#d95f02" if n == highlight else "#4c72b0" for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(min(values) - 0.05 if min(values) > 0.1 else 0, min(1.0, max(values) + 0.06))
    ax.set_xlabel("Macro F1")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

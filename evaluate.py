"""Evaluate the saved artifacts on the untouched final test split.

Usage: python evaluate.py
Reads models/ and data/processed/features.csv created by train.py. Model selection was already
done on the validation split in train.py; nothing here changes it.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.config import load_config, project_path
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_macro_f1_comparison,
    plot_roc_curves,
    positive_scores,
)
from src.features import FEATURE_NAMES
from src.inference import load_artifacts
from src.train_models import slugify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="Path to config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    models_dir = project_path(cfg["paths"]["models_dir"])
    reports_dir = project_path(cfg["paths"]["reports_dir"])
    figures_dir = project_path(cfg["paths"]["figures_dir"])
    features_csv = project_path(cfg["paths"]["processed_dir"]) / "features.csv"
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        metadata, models, ensembles = load_artifacts(models_dir)
        if not features_csv.is_file():
            raise FileNotFoundError(f"{features_csv} not found. Run train.py first.")
        val_path = reports_dir / "validation_report.json"
        if not val_path.is_file():
            raise FileNotFoundError(f"{val_path} not found. Run train.py first.")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with open(val_path, encoding="utf-8") as fh:
        val_metrics = json.load(fh)["metrics"]
    data = pd.read_csv(features_csv, encoding="utf-8")
    test = data[data["split"] == "test"]
    if len(test) != metadata["split_counts"]["test"]["n"]:
        print("ERROR: test split size does not match the saved metadata; re-run train.py.", file=sys.stderr)
        return 1
    X_test = test[FEATURE_NAMES].to_numpy()
    y_test = test["is_sarcastic"].to_numpy()

    all_models = {**models, **ensembles}
    primary = metadata["primary_ensemble"]
    results, curves = {}, {}
    for name, model in all_models.items():
        pred = model.predict(X_test)
        scores = positive_scores(model, X_test)
        m = compute_metrics(y_test, pred, scores)
        m["type"] = "ensemble" if name in ensembles else "individual"
        m["selected_for_ensemble"] = name in metadata["selected_models"]
        if name == "Ensemble: Majority Voting":
            m["roc_auc_note"] = "AUC computed from vote shares (only 4 distinct score values); coarse."
        results[name] = m
        if scores is not None:
            curves[name] = (y_test, scores)
        plot_confusion_matrix(m["confusion_matrix"], f"{name}\n(test, n={len(y_test)})",
                              figures_dir / f"confusion_matrix_{slugify(name)}.png")

    plot_roc_curves(curves, figures_dir / "roc_curves_test.png")
    names = list(results)
    plot_macro_f1_comparison(names, [results[n]["macro_f1"] for n in names],
                             figures_dir / "test_macro_f1_comparison.png", highlight=primary)

    rows = []
    for n in names:
        t, v = results[n], val_metrics[n]
        rows.append({
            "model": n, "type": t["type"], "selected_for_ensemble": t["selected_for_ensemble"],
            "primary_ensemble": n == primary,
            "val_accuracy": v["accuracy"], "val_macro_f1": v["macro_f1"],
            "test_accuracy": t["accuracy"], "test_macro_precision": t["macro_precision"],
            "test_macro_recall": t["macro_recall"], "test_macro_f1": t["macro_f1"],
            "test_weighted_f1": t["weighted_f1"], "test_roc_auc": t["roc_auc"],
        })
    table = pd.DataFrame(rows).sort_values(["type", "test_macro_f1"], ascending=[True, False])
    table.to_csv(reports_dir / "model_comparison.csv", index=False)

    with open(reports_dir / "final_test_report.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "note": "Computed from actual predictions on the final test split. "
                        "Model/ensemble selection used validation data only.",
                "dataset_file": metadata["dataset_file"],
                "n_test": int(len(y_test)),
                "label_mapping": metadata["label_mapping"],
                "selected_models": metadata["selected_models"],
                "primary_ensemble_selected_on_validation": primary,
                "results": results,
            },
            fh, indent=2,
        )

    print(f"Final test results (n={len(y_test)}); primary ensemble (chosen on validation): {primary}")
    print(table[["model", "test_accuracy", "test_macro_f1", "test_roc_auc"]].to_string(index=False, float_format="%.4f"))
    print(f"\nReports written to {reports_dir}; figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

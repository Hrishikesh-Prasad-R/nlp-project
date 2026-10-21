"""Train the seven classifiers and the two ensembles.

Usage (from the project root):
    python train.py --data dataset/Sarcasm_Headlines_Dataset_v2.json

The final TEST split is never touched here: it is only used by ``evaluate.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.config import load_config, project_path, resolve_input_path
from src.data_loader import DatasetError, load_dataset, print_audit
from src.features import FEATURE_NAMES, extract_features_batch
from src.inference import save_artifacts
from src.resources import ensure_nltk_data, resource_fingerprint
from src.train_models import train_and_select


def make_splits(df: pd.DataFrame, cfg: dict, seed: int) -> pd.Series:
    """Stratified train/validation/test split (70/15/15 by default). Returns a Series of split names."""
    ratios = cfg["split"]
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")
    idx = np.arange(len(df))
    y = df["is_sarcastic"].to_numpy()
    train_idx, rest_idx = train_test_split(
        idx, test_size=ratios["validation"] + ratios["test"], stratify=y, random_state=seed
    )
    val_share = ratios["validation"] / (ratios["validation"] + ratios["test"])
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_share, stratify=y[rest_idx], random_state=seed
    )
    split = np.empty(len(df), dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "validation", "test"
    return pd.Series(split, index=df.index, name="split")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="Path to a Sarcasm Headlines JSON Lines file (default: config data.path)")
    ap.add_argument("--config", help="Path to config.yaml (default: config/config.yaml)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = cfg["seed"]
    data_path = resolve_input_path(args.data or cfg["data"]["path"])
    models_dir = project_path(cfg["paths"]["models_dir"])
    reports_dir = project_path(cfg["paths"]["reports_dir"])
    processed_dir = project_path(cfg["paths"]["processed_dir"])
    for d in (models_dir, reports_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load + audit ---------------------------------------------------------
    try:
        df, audit = load_dataset(data_path)
    except (FileNotFoundError, DatasetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print_audit(audit)
    with open(reports_dir / "data_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    # 2. Split (after deduplication) ---------------------------------------------
    df["split"] = make_splits(df, cfg, seed)
    split_counts = {
        s: {
            "n": int((df.split == s).sum()),
            "sarcastic": int(df.loc[df.split == s, "is_sarcastic"].sum()),
            "non_sarcastic": int(((df.split == s) & (df.is_sarcastic == 0)).sum()),
        }
        for s in ("train", "validation", "test")
    }
    print("\nSplit sizes:", {k: v["n"] for k, v in split_counts.items()})
    splits = {
        "seed": seed,
        "ratios": cfg["split"],
        "id_definition": "record_id = 1-based line number in the input file",
        **{f"{s}_record_ids": df.loc[df.split == s, "record_id"].tolist() for s in ("train", "validation", "test")},
    }
    with open(processed_dir / "splits.json", "w", encoding="utf-8") as fh:
        json.dump(splits, fh)

    # 3. Features -----------------------------------------------------------------
    ensure_nltk_data()
    t0 = time.time()
    X = extract_features_batch(df["headline"])
    print(f"Extracted {X.shape[1]} features for {X.shape[0]} headlines in {time.time() - t0:.1f}s")
    feat_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(X, columns=FEATURE_NAMES)], axis=1)
    features_csv = processed_dir / "features.csv"
    feat_df.to_csv(features_csv, index=False, encoding="utf-8")

    train_m = (df.split == "train").to_numpy()
    val_m = (df.split == "validation").to_numpy()
    y = df["is_sarcastic"].to_numpy()

    # 4. Train + select on validation ---------------------------------------------
    print("\nTraining on the training split:")
    res = train_and_select(X[train_m], y[train_m], X[val_m], y[val_m], cfg, seed=seed)

    print("\nValidation results (used for model selection):")
    rows = []
    for name, m in res["val_metrics"].items():
        rows.append({"model": name, "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"]})
        print(f"  {name:<34s} acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
    print("Selected top models :", res["selected"])
    print("Primary ensemble    :", res["primary_ensemble"], "(by validation macro F1)")

    # 5. Save artifacts and validation report ----------------------------------------
    metadata = {
        "feature_names": FEATURE_NAMES,
        "seed": seed,
        "label_mapping": {"0": "Non-sarcastic", "1": "Sarcastic"},
        "dataset_file": audit["input_file"],
        "dataset_sha256": sha256_file(data_path),
        "n_records": audit["final_usable_records"],
        "split_counts": split_counts,
        "model_names": list(res["models"]),
        "selected_models": res["selected"],
        "validation_ranking": res["ranking"],
        "primary_ensemble": res["primary_ensemble"],
        "hyperparameters": cfg["models"],
        "features_csv": f"{cfg['paths']['processed_dir']}/features.csv",
        "resource_fingerprint": resource_fingerprint(),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    save_artifacts(res["models"], metadata, models_dir)
    with open(reports_dir / "validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "note": "Validation metrics, used for model/ensemble selection. Test metrics are in final_test_report.json.",
                "selected_models": res["selected"],
                "primary_ensemble": res["primary_ensemble"],
                "metrics": res["val_metrics"],
            },
            fh,
            indent=2,
        )
    with open(reports_dir / "split_summary.json", "w", encoding="utf-8") as fh:
        json.dump(split_counts, fh, indent=2)
    pd.DataFrame(rows).to_csv(reports_dir / "validation_comparison.csv", index=False)

    print(f"\nSaved models + metadata to {models_dir}")
    print("Next: python evaluate.py   (evaluates the saved artifacts on the untouched test split)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

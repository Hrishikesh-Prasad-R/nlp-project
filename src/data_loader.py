"""Robust JSON Lines loader for the Sarcasm Headlines datasets.

The files are JSON Lines (one JSON object per line), NOT a single JSON array, so every
line is parsed independently with ``json.loads``. ``article_link`` is never used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be loaded or has no usable records."""


MAX_LISTED_PROBLEMS = 100  # cap on per-record problems stored in the audit report


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load, validate and deduplicate a Sarcasm Headlines JSONL file.

    Returns
    -------
    df : DataFrame with columns ``record_id`` (1-based source line number),
         ``headline`` (stripped, original casing) and ``is_sarcastic`` (0/1).
         Row order is the file order of the first valid occurrence.
    audit : dict of counts (see keys below); also contains the malformed line numbers.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Pass the correct file with --data "
            "(e.g. --data dataset/Sarcasm_Headlines_Dataset_v2.json)."
        )

    raw_records = 0
    blank_lines = 0
    malformed: list[dict] = []
    empty_headlines = 0
    valid: list[tuple[int, str, int]] = []

    try:
        with open(path, "r", encoding="utf-8-sig") as fh:  # utf-8-sig tolerates a BOM
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    blank_lines += 1
                    continue
                raw_records += 1
                problem = None
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed.append({"line": line_no, "reason": f"invalid JSON: {exc.msg}"})
                    continue
                if not isinstance(obj, dict):
                    problem = "record is not a JSON object"
                elif "headline" not in obj:
                    problem = "missing 'headline' field"
                elif "is_sarcastic" not in obj:
                    problem = "missing 'is_sarcastic' field"
                else:
                    label = obj["is_sarcastic"]
                    # bool is a subclass of int in Python; JSON true/false is not a valid label.
                    if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
                        problem = f"is_sarcastic must be integer 0 or 1, got {label!r}"
                    elif obj["headline"] is not None and not isinstance(obj["headline"], str):
                        problem = f"headline must be a string or null, got {type(obj['headline']).__name__}"
                if problem:
                    malformed.append({"line": line_no, "reason": problem})
                    continue
                headline = obj["headline"]
                if headline is None or not headline.strip():
                    empty_headlines += 1
                    continue
                valid.append((line_no, headline.strip(), int(obj["is_sarcastic"])))
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{path} is not valid UTF-8: {exc}") from exc

    # Exact-duplicate removal (case-sensitive, on the stripped headline); first occurrence kept.
    seen: dict[str, int] = {}
    kept: list[tuple[int, str, int]] = []
    duplicates = 0
    conflicting = 0
    for rec in valid:
        first_label = seen.get(rec[1])
        if first_label is None:
            seen[rec[1]] = rec[2]
            kept.append(rec)
        else:
            duplicates += 1
            if first_label != rec[2]:
                conflicting += 1

    if not kept:
        raise DatasetError(
            f"No usable records in {path} ({raw_records} raw records, {len(malformed)} malformed)."
        )

    df = pd.DataFrame(kept, columns=["record_id", "headline", "is_sarcastic"])
    n = len(df)
    n_sarc = int(df["is_sarcastic"].sum())
    audit = {
        "input_file": path.name,
        "input_path": os.path.relpath(path) if _is_relative_ok(path) else path.name,
        "total_raw_records": raw_records,
        "blank_lines_ignored": blank_lines,
        "malformed_records": len(malformed),
        "malformed_line_numbers": [m["line"] for m in malformed[:MAX_LISTED_PROBLEMS]],
        "malformed_details": malformed[:MAX_LISTED_PROBLEMS],
        "missing_or_empty_headlines": empty_headlines,
        "valid_records_before_deduplication": len(valid),
        "duplicate_headlines_removed": duplicates,
        "duplicates_with_conflicting_labels": conflicting,
        "final_usable_records": n,
        "sarcastic_count": n_sarc,
        "sarcastic_percent": round(100 * n_sarc / n, 2),
        "non_sarcastic_count": n - n_sarc,
        "non_sarcastic_percent": round(100 * (n - n_sarc) / n, 2),
    }
    return df, audit


def _is_relative_ok(path: Path) -> bool:
    try:
        os.path.relpath(path)
        return True
    except ValueError:  # different drive on Windows
        return False


def print_audit(audit: dict) -> None:
    print("Data audit")
    print("----------")
    print(f"  Input file                      : {audit['input_file']}")
    print(f"  Total raw records               : {audit['total_raw_records']}")
    print(f"  Malformed records               : {audit['malformed_records']}")
    print(f"  Missing/empty headlines         : {audit['missing_or_empty_headlines']}")
    print(f"  Exact duplicate headlines removed: {audit['duplicate_headlines_removed']}")
    print(f"  Final usable records            : {audit['final_usable_records']}")
    print(f"  Sarcastic                       : {audit['sarcastic_count']} ({audit['sarcastic_percent']}%)")
    print(f"  Non-sarcastic                   : {audit['non_sarcastic_count']} ({audit['non_sarcastic_percent']}%)")

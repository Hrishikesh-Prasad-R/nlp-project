import json
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT
from src.data_loader import DatasetError, load_dataset

REAL_V2 = PROJECT_ROOT / "dataset" / "Sarcasm_Headlines_Dataset_v2.json"


def write_lines(tmp_path, lines):
    p = tmp_path / "data.json"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def rec(headline, label, link="https://example.com/x"):
    return json.dumps({"is_sarcastic": label, "headline": headline, "article_link": link})


def test_loads_jsonl_with_audit(tmp_path):
    p = write_lines(tmp_path, [rec("  first headline  ", 1), rec("second one", 0), rec("third", 0)])
    df, audit = load_dataset(p)
    assert list(df["headline"]) == ["first headline", "second one", "third"]  # stripped
    assert list(df["is_sarcastic"]) == [1, 0, 0]
    assert list(df["record_id"]) == [1, 2, 3]
    assert audit["total_raw_records"] == 3
    assert audit["final_usable_records"] == 3
    assert audit["sarcastic_count"] == 1 and audit["non_sarcastic_count"] == 2
    assert audit["sarcastic_percent"] == pytest.approx(33.33)
    assert "article_link" not in df.columns


def test_malformed_records_reported_with_line_numbers(tmp_path):
    p = write_lines(tmp_path, [
        rec("ok", 1),
        "{not json",                                           # 2 invalid JSON
        json.dumps({"headline": "no label"}),                  # 3 missing label
        json.dumps({"is_sarcastic": 1}),                       # 4 missing headline
        json.dumps({"is_sarcastic": 2, "headline": "bad"}),    # 5 bad label
        json.dumps({"is_sarcastic": True, "headline": "b"}),   # 6 bool label rejected
        json.dumps({"is_sarcastic": "1", "headline": "s"}),    # 7 string label
        json.dumps({"is_sarcastic": 1.0, "headline": "f"}),    # 8 float label
        json.dumps([1, 2]),                                    # 9 not an object
        json.dumps({"is_sarcastic": 0, "headline": 5}),        # 10 non-string headline
        rec("also ok", 0),
    ])
    df, audit = load_dataset(p)
    assert audit["malformed_records"] == 9
    assert audit["malformed_line_numbers"] == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert all("reason" in d for d in audit["malformed_details"])
    assert list(df["headline"]) == ["ok", "also ok"]


def test_empty_headlines_dropped_and_counted(tmp_path):
    p = write_lines(tmp_path, [
        rec("keep", 0),
        json.dumps({"is_sarcastic": 1, "headline": None}),
        json.dumps({"is_sarcastic": 1, "headline": ""}),
        json.dumps({"is_sarcastic": 1, "headline": "   \t "}),
    ])
    df, audit = load_dataset(p)
    assert audit["missing_or_empty_headlines"] == 3
    assert audit["malformed_records"] == 0
    assert len(df) == 1


def test_exact_duplicates_keep_first_occurrence(tmp_path):
    p = write_lines(tmp_path, [rec("same", 1), rec("other", 0), rec(" same ", 0), rec("same", 1)])
    df, audit = load_dataset(p)
    assert audit["duplicate_headlines_removed"] == 2
    assert audit["duplicates_with_conflicting_labels"] == 1
    assert list(df["headline"]) == ["same", "other"]
    assert list(df["is_sarcastic"]) == [1, 0]  # first valid occurrence preserved
    assert list(df["record_id"]) == [1, 2]


def test_blank_lines_ignored(tmp_path):
    p = write_lines(tmp_path, [rec("a", 0), "", "   ", rec("b", 1)])
    df, audit = load_dataset(p)
    assert len(df) == 2 and audit["blank_lines_ignored"] == 2 and audit["total_raw_records"] == 2


def test_missing_file_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--data"):
        load_dataset(tmp_path / "nope.json")


def test_no_usable_records_fails(tmp_path):
    p = write_lines(tmp_path, ["garbage", json.dumps({"is_sarcastic": 5, "headline": "x"})])
    with pytest.raises(DatasetError):
        load_dataset(p)


def test_json_array_is_not_supported_as_jsonl(tmp_path):
    p = tmp_path / "array.json"
    p.write_text(json.dumps([{"is_sarcastic": 1, "headline": "x"}]), encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(p)


@pytest.mark.skipif(not REAL_V2.is_file(), reason="real v2 dataset not present")
def test_real_v2_dataset_loads():
    df, audit = load_dataset(REAL_V2)
    assert audit["malformed_records"] == 0
    assert audit["final_usable_records"] + audit["duplicate_headlines_removed"] == audit["total_raw_records"]
    assert set(df["is_sarcastic"].unique()) == {0, 1}
    assert df["headline"].is_unique

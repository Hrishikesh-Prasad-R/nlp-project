import json

import numpy as np
import pytest

from src.config import load_config
from src.ensemble import AVERAGING_NAME, MAJORITY_NAME, MajorityVotingEnsemble
from src.features import FEATURE_NAMES, extract_features, extract_features_batch
from src.inference import SarcasmPredictor, save_artifacts
from src.resources import resource_fingerprint
from src.train_models import MODEL_NAMES, rank_models, select_top_k, train_and_select

WORDS_A = "area man local woman nation scientists report study finds mayor town dog".split()
WORDS_B = "senate passes budget bill president signs trade deal court rules election results".split()


def synthetic_dataset(n=240, seed=0):
    rng = np.random.default_rng(seed)
    heads, labels = [], []
    for i in range(n):
        label = i % 2
        pool = WORDS_A if label else WORDS_B
        k = int(rng.integers(4, 9))
        heads.append(" ".join(rng.choice(pool, size=k)))
        labels.append(label)
    return heads, np.array(labels)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    cfg = load_config()
    cfg["models"]["gradient_boosting"]["n_estimators"] = 20
    cfg["models"]["random_forest"]["n_estimators"] = 20
    heads, y = synthetic_dataset()
    X = extract_features_batch(heads)
    tr, va = slice(0, 160), slice(160, 240)
    res = train_and_select(X[tr], y[tr], X[va], y[va], cfg, seed=42, verbose=False)
    models_dir = tmp_path_factory.mktemp("models")
    meta = {
        "feature_names": FEATURE_NAMES,
        "model_names": list(res["models"]),
        "selected_models": res["selected"],
        "primary_ensemble": res["primary_ensemble"],
        "resource_fingerprint": resource_fingerprint(),
    }
    save_artifacts(res["models"], meta, models_dir)
    return res, models_dir, heads


def test_all_seven_models_trained(trained):
    res, _, _ = trained
    assert set(res["models"]) == set(MODEL_NAMES) and len(MODEL_NAMES) == 7
    assert len(res["selected"]) == 3
    assert set(res["ensembles"]) == {MAJORITY_NAME, AVERAGING_NAME}
    assert res["primary_ensemble"] in res["ensembles"]


def test_selection_rule_uses_macro_f1_then_accuracy_then_name():
    m = lambda f1, acc: {"macro_f1": f1, "accuracy": acc}
    val = {
        "B model": m(0.80, 0.80), "A model": m(0.80, 0.80),   # full tie -> alphabetical
        "C model": m(0.80, 0.85),                               # tie on F1 -> higher accuracy first
        "D model": m(0.90, 0.10),                               # best macro F1 wins regardless
        "E model": m(0.50, 0.99),
    }
    assert rank_models(val) == ["D model", "C model", "A model", "B model", "E model"]
    assert select_top_k(val, 3) == ["D model", "C model", "A model"]


def test_svc_supports_probabilities(trained):
    res, _, _ = trained
    svc = res["models"]["Support Vector Classifier"]
    proba = svc.predict_proba(np.zeros((1, 13)))
    assert proba.shape == (1, 2) and proba.sum() == pytest.approx(1.0)


def test_scaler_fitted_on_train_only(trained):
    res, _, heads = trained
    X_train = extract_features_batch(heads[:160])
    scaler = res["models"]["Logistic Regression"].named_steps["scaler"]
    assert np.allclose(scaler.mean_, X_train.mean(axis=0))


def test_majority_vote_is_member_majority(trained):
    res, _, heads = trained
    X = extract_features_batch(heads[160:])
    ens = res["ensembles"][MAJORITY_NAME]
    votes = ens.member_predictions(X)
    assert np.array_equal(ens.predict(X), (votes.sum(axis=0) >= 2).astype(int))


def test_majority_tie_fallback_is_deterministic():
    class Const:
        def __init__(self, label, p):
            self.label, self.p = label, p

        def predict(self, X):
            return np.full(len(X), self.label)

        def predict_proba(self, X):
            return np.tile([1 - self.p, self.p], (len(X), 1))

    # 1 vote each way: tie -> mean probability decides (0.9 and 0.2 -> mean 0.55 -> class 1)
    ens = MajorityVotingEnsemble({"a": Const(1, 0.9), "b": Const(0, 0.2)})
    assert ens.predict(np.zeros((2, 13))).tolist() == [1, 1]
    ens2 = MajorityVotingEnsemble({"a": Const(1, 0.6), "b": Const(0, 0.1)})
    assert ens2.predict(np.zeros((1, 13))).tolist() == [0]


def test_averaging_uses_mean_probability(trained):
    res, _, heads = trained
    X = extract_features_batch(heads[160:])
    ens = res["ensembles"][AVERAGING_NAME]
    mean = np.mean([m.predict_proba(X) for m in ens.models.values()], axis=0)
    assert np.allclose(ens.predict_proba(X), mean)
    assert np.array_equal(ens.predict(X), mean.argmax(axis=1))


def test_inference_uses_training_features_and_order(trained):
    res, models_dir, heads = trained
    predictor = SarcasmPredictor(models_dir)
    text = heads[3]
    out = predictor.predict(text, "Random Forest")
    assert list(out["features"]) == FEATURE_NAMES
    assert np.array_equal(np.array(list(out["features"].values())), extract_features(text))
    expected = int(res["models"]["Random Forest"].predict(extract_features(text).reshape(1, -1))[0])
    assert out["label_id"] == expected


def test_predict_all_models_and_probability_support(trained):
    _, models_dir, heads = trained
    predictor = SarcasmPredictor(models_dir)
    assert predictor.model_names[0] == predictor.primary
    for name in predictor.model_names:
        out = predictor.predict(heads[0], name)
        assert out["label"] in ("Sarcastic", "Non-sarcastic")
        if name == MAJORITY_NAME:
            assert out["votes"] is not None and out["probability_sarcastic"] is None
        else:
            assert 0.0 <= out["probability_sarcastic"] <= 1.0
            assert 0.0 <= out["confidence"] <= 1.0


def test_new_headline_classified_and_deterministic(trained):
    _, models_dir, _ = trained
    predictor = SarcasmPredictor(models_dir)
    a = predictor.predict("Local man wins lottery, immediately loses ticket")
    b = predictor.predict("Local man wins lottery, immediately loses ticket")
    assert a == b


@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_empty_input_rejected(trained, bad):
    _, models_dir, _ = trained
    with pytest.raises(ValueError):
        SarcasmPredictor(models_dir).predict(bad)


def test_unknown_model_rejected(trained):
    _, models_dir, _ = trained
    with pytest.raises(KeyError):
        SarcasmPredictor(models_dir).predict("hello world", "Nope")


def test_missing_artifacts_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        SarcasmPredictor(tmp_path)


def test_feature_order_mismatch_detected(trained, tmp_path):
    _, models_dir, _ = trained
    import shutil

    shutil.copytree(models_dir, tmp_path / "m")
    meta_path = tmp_path / "m" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["feature_names"] = list(reversed(meta["feature_names"]))
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="Feature order"):
        SarcasmPredictor(tmp_path / "m")

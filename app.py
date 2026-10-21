"""English Sarcasm Detection - Streamlit app. Loads saved artifacts; never retrains."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import load_config, project_path
from src.features import FEATURE_NAMES
from src.inference import SarcasmPredictor

st.set_page_config(page_title="English Sarcasm Detection", page_icon="🗞️", layout="wide")

CFG = load_config()
MODELS_DIR = project_path(CFG["paths"]["models_dir"])
REPORTS_DIR = project_path(CFG["paths"]["reports_dir"])
FIGURES_DIR = project_path(CFG["paths"]["figures_dir"])
SAMPLE_CSV = REPORTS_DIR / "sample_headlines.csv"


@st.cache_resource(show_spinner="Loading saved models...")
def get_predictor() -> SarcasmPredictor:
    return SarcasmPredictor(MODELS_DIR)


def read_json(path):
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


st.title("English Sarcasm Detection")
st.caption("Traditional Machine Learning with Handcrafted Linguistic Features")

try:
    predictor = get_predictor()
except FileNotFoundError as exc:
    st.error(f"Saved models not found. {exc}")
    st.code("python train.py --data dataset/Sarcasm_Headlines_Dataset_v2.json\npython evaluate.py")
    st.stop()
except Exception as exc:  # shown in full here, because Streamlit Cloud hides uncaught error messages
    st.error(f"The app could not start: {type(exc).__name__}: {exc}")
    st.stop()

tab_predict, tab_compare, tab_data, tab_about = st.tabs(
    ["Headline prediction", "Model comparison", "Dataset overview", "About the model"]
)

# ------------------------------------------------------------------ A. prediction
with tab_predict:
    st.subheader("Classify a headline")
    model_name = st.selectbox(
        "Model",
        predictor.model_names,
        format_func=lambda n: f"{n} (primary, selected on validation)" if n == predictor.primary else n,
    )
    headline = st.text_area("Headline", height=100, placeholder="e.g. area man passes up opportunity to touch the fridge")
    if st.button("Predict", type="primary"):
        if not headline.strip():
            st.warning("Please enter a headline first.")
        else:
            res = predictor.predict(headline, model_name)
            if res["label_id"] == 1:
                st.error(f"Prediction: **{res['label']}**")
            else:
                st.success(f"Prediction: **{res['label']}**")
            if res["votes"] is not None:
                st.write(
                    "Votes: "
                    + ", ".join(f"{m}: {v}" for m, v in res["votes"].items())
                    + f"  \n(Majority voting gives no probability; vote share for sarcastic = {res['vote_share_sarcastic']:.2f})"
                )
            elif res["probability_sarcastic"] is not None:
                c1, c2 = st.columns(2)
                c1.metric("P(sarcastic)", f"{res['probability_sarcastic']:.1%}")
                c2.metric("Confidence in prediction", f"{res['confidence']:.1%}")
                st.caption("Probabilities are model scores and may not be calibrated.")
            st.markdown("**Extracted linguistic features**")
            st.dataframe(
                pd.DataFrame({"feature": FEATURE_NAMES, "value": [round(res["features"][f], 4) for f in FEATURE_NAMES]}),
                hide_index=True,
                width="stretch",
            )
    st.info(
        "Predictions can be wrong and sarcasm is subjective. This model only sees 13 surface "
        "statistics of the headline, not its meaning or context."
    )

# ------------------------------------------------------------------ B. comparison
with tab_compare:
    st.subheader("Model comparison")
    comp_path = REPORTS_DIR / "model_comparison.csv"
    if not comp_path.is_file():
        st.warning("Test results not found. Run `python evaluate.py` first.")
    else:
        comp = pd.read_csv(comp_path)
        st.success(f"Primary ensemble (selected on the **validation** split): **{predictor.primary}**")
        st.markdown("**Final test set** metrics (never used for any selection) next to the validation metrics used for selection.")
        show = comp[["model", "type", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_roc_auc"]].copy()
        show.columns = ["Model", "Type", "Validation accuracy", "Validation macro F1",
                        "Test accuracy", "Test macro F1", "Test ROC-AUC"]
        st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns[2:]}), hide_index=True, width="stretch")
        chart = FIGURES_DIR / "test_macro_f1_comparison.png"
        if chart.is_file():
            st.image(str(chart), caption="Test macro F1 by model (orange = primary ensemble)")
        roc = FIGURES_DIR / "roc_curves_test.png"
        if roc.is_file():
            with st.expander("ROC curves (test set)"):
                st.image(str(roc))
        st.caption(
            "The paper reports ~92-94% accuracy; this implementation does not reproduce that. "
            "See 'About the model' for the reasons."
        )

# ------------------------------------------------------------------ C. dataset
with tab_data:
    st.subheader("Dataset overview")
    audit = read_json(REPORTS_DIR / "data_audit.json")
    splits = read_json(REPORTS_DIR / "split_summary.json")
    if audit is None:
        st.warning("Data audit not found. Run train.py.")
    else:
        st.write(f"Dataset file: `{audit['input_file']}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Usable records", f"{audit['final_usable_records']:,}")
        c2.metric("Sarcastic", f"{audit['sarcastic_count']:,} ({audit['sarcastic_percent']}%)")
        c3.metric("Non-sarcastic", f"{audit['non_sarcastic_count']:,} ({audit['non_sarcastic_percent']}%)")
        st.caption(
            f"Raw records: {audit['total_raw_records']:,} | malformed: {audit['malformed_records']} | "
            f"empty headlines: {audit['missing_or_empty_headlines']} | exact duplicates removed: {audit['duplicate_headlines_removed']}"
        )
        st.bar_chart(pd.DataFrame(
            {"count": [audit["non_sarcastic_count"], audit["sarcastic_count"]]},
            index=["Non-sarcastic", "Sarcastic"],
        ))
    if splits:
        st.markdown("**Stratified split (seed 42)**")
        st.dataframe(
            pd.DataFrame(splits).T.rename(columns={"n": "Total", "sarcastic": "Sarcastic", "non_sarcastic": "Non-sarcastic"}),
            width="stretch",
        )
    if SAMPLE_CSV.is_file():
        st.markdown("**Sample headlines** (article links are never used or shown)")
        sample = pd.read_csv(SAMPLE_CSV, encoding="utf-8").head(10)
        sample["label"] = sample["is_sarcastic"].map({0: "Non-sarcastic", 1: "Sarcastic"})
        st.dataframe(sample[["headline", "label"]], hide_index=True, width="stretch")

# ------------------------------------------------------------------ D. about
with tab_about:
    st.subheader("About the model")
    st.markdown(
        """
**Handcrafted linguistic features.** Each headline is reduced to 13 numbers (following Table 2 of
Pradhan et al., 2024): word entropy, lexical diversity, Dale-Chall readability, Flesch Reading Ease,
and counts of stop words, unknown ("incorrect") words, difficult words, words longer than two letters,
two-letter words, single-letter words, verbs, adjectives and nouns. No bag-of-words or embeddings are used.

**Traditional ML classifiers.** Gradient Boosting, Decision Tree, Random Forest, Support Vector
Classifier, Gaussian Naive Bayes, K-Nearest Neighbors and Logistic Regression are trained on a 70% training split.
The validation split (15%) selects the three best by macro F1; the test split (15%) is used once for the final report.

**Majority voting.** The three selected models each vote Sarcastic / Non-sarcastic and the majority wins
(three voters on two classes cannot tie). The second ensemble averages the three models' probabilities
(these are not calibrated). The primary ensemble is whichever scored higher on validation macro F1.

**Dataset limitations.** Sarcastic headlines come from The Onion and non-sarcastic from HuffPost, so models can
learn source style (length, vocabulary, phrasing conventions) rather than sarcasm itself. Only headlines
are used, with no context, and sarcasm is partly subjective.

**Differences from the paper.** The paper splits long headlines into shorter fragments and reports ~92-94% accuracy; this
project keeps whole headlines (no fragments), uses a 70/15/15 split instead of 80/20, de-duplicates before
splitting, and makes its own choices for the vocabularies, tokenizer, POS tagger and the unspecified
formula details (e.g. the Dale-Chall adjustment, "lengthy words" implemented literally as > 2 characters).
Results are therefore not expected to match the paper, and they do not.
"""
    )

# English Sarcasm Detection — Handcrafted Linguistic Features + Ensemble ML

Implementation of the method of *"An Efficient Sarcasm Detection using Linguistic Features and Ensemble
Machine Learning"* (Pradhan et al., Procedia Computer Science 235, 2024) on the News Headlines Sarcasm
dataset (v2): 13 handcrafted linguistic features → seven scikit-learn classifiers → the best three
combined by **majority voting** and **probability averaging**, plus a Streamlit app.

This is a re-implementation from the paper's description. Many details are unspecified in the paper
(vocabularies, tokenizer, POS tagger, ...); every choice made here is documented below and in
[`src/features.py`](src/features.py). Exact reproduction of the paper's numbers is **not** claimed.

## Quick start

Requires Python 3.10+ (developed on 3.14). From the project root:

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows (use `source .venv/bin/activate` on Linux/macOS)
python -m pip install -r requirements.txt
python train.py --data dataset/Sarcasm_Headlines_Dataset_v2.json
python evaluate.py
streamlit run app.py
python -m pytest
```

* The datasets are JSON Lines (one JSON object per line, **not** a JSON array). Put them anywhere and
  point `--data` to them (path relative to the working directory or project root), or edit `data.path`
  in [`config/config.yaml`](config/config.yaml). The raw files are never modified.
* The first run downloads four small NLTK resources (stopwords, words, cmudict, perceptron tagger) into
  `nltk_data/` (internet needed once). The Dale-Chall easy-word list is vendored in `src/resources/`.
* `train.py` takes ~2–3 minutes (the SVC with probability estimates dominates).
* `train.py` never reads the test split; `evaluate.py` reads it for the first time.
* v2 is the main experiment. v1 (`dataset/Sarcasm_Headlines_Dataset.json`) can be run separately with
  `--data`; the two files are never concatenated. **Note:** running `train.py` overwrites `models/` and
  `reports/` — copy them first if you want to keep results for both versions.

## Deploy on Streamlit Community Cloud

The app only loads saved artifacts, so the trained models and report files are committed to the repo
(`models/*.joblib`, `models/metadata.json`, `reports/`). The raw dataset and `data/processed/` are not needed
to run the app.

1. Push the repo to GitHub (public, or private with Streamlit access granted).
2. Go to https://share.streamlit.io and sign in with GitHub, then click **Create app**.
3. Choose the repository `Hrishikesh-Prasad-R/nlp-project`, branch `main`, main file `app.py`.
4. Open **Advanced settings** and select **Python 3.12** (the pinned numpy, pandas and scikit-learn
   versions need Python 3.11 or newer; scikit-learn must stay at 1.8.0 so the saved models load).
5. Click **Deploy**. The first start takes a few minutes: dependencies are installed and four small NLTK
   resources are downloaded on first use.

To retrain with different settings, run `train.py` and `evaluate.py` locally and commit the regenerated
`models/` and `reports/` files.

## Project layout

```
app.py              Streamlit app (loads saved artifacts, never retrains)
train.py            audit -> split -> features -> train 7 models -> select on validation -> save
evaluate.py         evaluates saved artifacts on the untouched test split, writes reports/figures
config/config.yaml  data path, seed, split ratios, explicit hyperparameters
src/                data_loader, preprocessing, resources, features, train_models, ensemble,
                    evaluation, inference
tests/              pytest suite (loader, preprocessing, features, inference/selection/ensembles)
models/             *.joblib + metadata.json (generated; binaries are git-ignored)
reports/            data_audit.json, validation_report.json, model_comparison.csv,
                    final_test_report.json, figures/ (generated)
data/processed/     features.csv and splits.json (generated; record_id = source line number)
```

## Method

**Data loading** ([`src/data_loader.py`](src/data_loader.py)): UTF-8, one `json.loads` per line; both
`headline` and `is_sarcastic` required; label must be integer 0/1 (booleans, floats, strings rejected);
malformed lines are reported with line numbers and skipped; null/empty/whitespace headlines dropped and
counted; headlines stripped; exact duplicate headlines removed (first kept) **before** splitting.
`article_link` is never used. Audit goes to `reports/data_audit.json`.

**Preprocessing** ([`src/preprocessing.py`](src/preprocessing.py)): decode HTML entities → strip HTML tags →
remove URLs → remove emoji → normalise curly apostrophes → lower-case → collapse whitespace. Punctuation,
negations and stop words are kept; no stemming/lemmatisation. The original headline is kept for display.

**13 features** (fixed order; [`src/features.py`](src/features.py)): `entropy`, `lexical_diversity`,
`dale_chall_readability`, `flesch_reading_ease`, `stop_word_count`, `incorrect_word_count`,
`difficult_word_count`, `lengthy_word_count`, `two_letter_word_count`, `single_letter_word_count`,
`verb_count`, `adjective_count`, `noun_count`.

| Choice | Implementation |
|---|---|
| Tokenizer | regex on the preprocessed text; contractions (`don't`) and hyphenated words (`x-ray`) stay one token; purely numeric tokens are not "words" |
| Sentences | every headline = 1 sentence (average sentence length = word count) |
| Stop words | NLTK English stop-word list |
| "Incorrect" words | not in NLTK `words` ∪ CMUdict headwords (after contraction/hyphen/inflection handling). Proper nouns, slang and uncommon valid words count as incorrect |
| "Difficult" words | not in the 2,941-word Dale-Chall familiar list (vendored from `textstat` 0.7.13, MIT) |
| Dale-Chall | `0.1579·PDW + 0.0496·ASL (+3.6365 if PDW>5%)`. The paper's Table 2 omits the adjustment; the standard formula is used |
| Flesch | `206.835 − 1.015·(words/sentences) − 84.6·(syllables/words)`; CMUdict syllables, vowel-group heuristic fallback |
| Lengthy words | **`|w| > 2` literally, as in the paper's Table 2**, although "lengthy" suggests a larger threshold |
| POS tags | NLTK averaged perceptron tagger; verbs `VB*`, adjectives `JJ*`, nouns `NN*`. It sees unsplit contractions, so their tags are approximate |
| Empty / no-word input | all 13 features are 0.0 (always finite) |
| Versions | NLTK pinned to 3.10.4; NLTK data has no version number, so content hashes of all resources are saved in `models/metadata.json` and checked at load time |

**Models** ([`config/config.yaml`](config/config.yaml), seed 42, no hyperparameter search): Gradient Boosting,
Decision Tree, Random Forest, SVC (RBF, `probability=True`), Gaussian NB, KNN, Logistic Regression.
`StandardScaler` (fitted on the training split only) is part of the SVC/KNN/LR pipelines.

**Split**: stratified 70/15/15 (train/validation/test), seed 42, after deduplication. Record IDs of each
split are in `data/processed/splits.json`.

**Selection (validation only)**: the top three individual models by validation macro F1 (ties → validation
accuracy → alphabetical name). Ensemble members are the already-trained models (no refit).
*Majority voting*: one class vote per member; a tie (impossible with three binary voters) falls back to the
mean predicted probability, then class 0. *Probability averaging*: mean `predict_proba`; these averaged
probabilities are **not calibrated**. The primary ensemble is the one with higher validation macro F1
(then validation accuracy, then name).

**Evaluation** (`evaluate.py`, test split only): accuracy, macro precision/recall/F1, weighted F1,
per-class precision/recall/F1, confusion matrices, ROC-AUC (majority voting uses vote shares as scores, so its
ROC is coarse). Outputs: `reports/model_comparison.csv`, `reports/final_test_report.json`,
`reports/figures/*.png`. Everything is computed from actual predictions.

## Results

See the section "Actual results of the run documented here" at the bottom of this file, and the generated
files in `reports/`, which are always authoritative.

## Why results differ from the paper

The paper reports ≈92–94 % accuracy for the tree-based models and the ensembles. Likely reasons this
implementation differs:

1. **Data construction.** The paper splits longer headlines into shorter fragments (Sec. 3.1) before an
   80:20 split; this project keeps whole headlines, as the task requires. If fragments of one headline land
   in both train and test, the test set is no longer independent, which can inflate accuracy. This project
   de-duplicates exact headlines and splits at headline level.
2. **Unspecified feature details**: vocabularies, tokenizer, POS tagger, syllable counter, Dale-Chall adjustment,
   sentence handling.
3. **Split ratio and random seed** (70/15/15 + validation-based selection vs 80/20).
4. **Hyperparameters** are unspecified in the paper.

## Limitations

* **Source bias.** Sarcastic headlines come from *The Onion* and non-sarcastic ones from *HuffPost*. A model can
  exploit source-specific style and topic (length, vocabulary, formatting) and thereby look good without
  understanding sarcasm. Scores on this dataset do not guarantee performance on other text.
* Only 13 surface statistics are used; no semantics, sentiment or context.
* Sarcasm is subjective; labels reflect source, not human judgement of each headline.
* Incorrect-word and POS features are noisy for proper nouns, slang and contractions.
* Probability outputs may be uncalibrated.

## Tests

`python -m pytest` runs 71 tests covering the loader (malformed/duplicate/empty/blank records, real v2 file),
preprocessing, the 13 features (edge cases, formulas, determinism, finiteness) and inference (selection rule,
ensembles, scaler fit on train only, training/inference feature parity, error handling).

## Actual results of the run documented here

Dataset: `Sarcasm_Headlines_Dataset_v2.json` (v2 only). Audit: 28,619 raw records, 0 malformed, 0 empty
headlines, 116 exact duplicates removed → **28,503** usable (13,552 sarcastic = 47.55 %; 14,951 non-sarcastic
= 52.45 %). Split: train 19,952 / validation 4,275 / test 4,276.

Selected on validation (macro F1): Support Vector Classifier, Gradient Boosting, Random Forest. Primary
ensemble (validation): **Majority Voting** (val macro F1 0.6890 vs 0.6850 for averaging).

Final test set (n = 4,276), from `reports/model_comparison.csv`:

| Model | Test accuracy | Test macro F1 | Test ROC-AUC |
|---|---|---|---|
| Ensemble: Probability Averaging | 0.6943 | 0.6923 | 0.7593 |
| **Ensemble: Majority Voting (primary)** | 0.6920 | 0.6899 | 0.7163 |
| Gradient Boosting | 0.6892 | 0.6871 | 0.7548 |
| Support Vector Classifier | 0.6890 | 0.6870 | 0.7565 |
| Random Forest | 0.6756 | 0.6736 | 0.7368 |
| Decision Tree | 0.6623 | 0.6606 | 0.7058 |
| Logistic Regression | 0.6597 | 0.6575 | 0.7172 |
| K-Nearest Neighbors | 0.6473 | 0.6457 | 0.7080 |
| Gaussian Naive Bayes | 0.5921 | 0.5916 | 0.6453 |

Averaging scored slightly higher on test than the validation-selected majority voting; the primary ensemble
was *not* changed after seeing test data. The ~69 % accuracy is far below the paper's 92–94 %; see
"Why results differ from the paper". No result was tuned to approach the paper's numbers.

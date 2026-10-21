"""The paper's 13 handcrafted linguistic features (Section 3.2, Table 2).

Canonical entry points: :func:`extract_features` (one headline -> 13 floats, fixed order) and
:func:`extract_features_batch`. Training and inference both go through these functions.

Implementation choices (the paper does not specify them uniquely):

Tokenizer
    After :func:`preprocess`, tokens are matched by ``[^\\W_]+(?:['-][^\\W_]+)*``. Contractions
    (``don't``) and hyphenated words (``x-ray``) therefore stay ONE token. A *word token* is a
    token containing at least one letter; purely numeric tokens (``9``, ``2016``) are not word
    tokens (they are ignored by every word-based feature but are still shown to the POS tagger
    as context). Letter counts (features 8-10) use alphabetic characters only, so ``don't`` has
    4 letters and ``x-ray`` has 4.

Sentences
    A headline is treated as exactly ONE sentence, whatever its punctuation (headlines have no
    reliable terminal punctuation, and ``rep.``/``u.s.`` would be mis-split). Hence average
    sentence length = number of word tokens.

Empty input
    A headline without word tokens yields all 13 features = 0.0 (including the readability
    scores) so every output is finite.

Dale-Chall (feature 3)
    ``0.1579 * PDW + 0.0496 * ASL`` plus the standard adjustment ``+3.6365`` when PDW > 5,
    PDW = percentage of words missing from the easy-word list, ASL = words per sentence.
    NOTE: the paper's Table 2 lists only the first two terms; the standard published formula
    (with the adjustment) is used here.

Flesch Reading Ease (feature 4)
    ``206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)``. Syllables come from
    CMUdict (first pronunciation, number of stressed-vowel phonemes); words not in CMUdict use a
    vowel-group heuristic (min. 1 syllable per word). Hyphenated tokens are summed part by part.

Easy / known word lookups (features 6-7)
    Contractions are reduced to their base (``isn't``->``is``, ``trump's``->``trump``,
    ``they're``->``they``), hyphenated words are checked part by part (all parts must match),
    and regular inflections (-s, -es, -ies, -ed, -d, -ing, -ly) are stripped before lookup.
    "Incorrect words" are word tokens absent from the known vocabulary; proper nouns, slang,
    neologisms and valid uncommon words are counted as incorrect. "Difficult words" are word
    tokens absent from the Dale-Chall easy-word list (so numbers-with-letters, names and most
    long words count as difficult).

Lengthy words (feature 8)
    The paper's Table 2 defines it as ``|w| > 2`` (more than TWO characters), which does not
    match the name "lengthy". The literal definition is implemented for the paper-aligned
    experiment.

POS tags (features 11-13)
    NLTK averaged-perceptron tagger (Penn Treebank tagset): verbs = tags starting ``VB``,
    adjectives = ``JJ``, nouns = ``NN`` (NN, NNS, NNP, NNPS). The tagger sees the unsplit
    contraction tokens, which it was not trained on, so tags for contractions are approximate.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from . import resources
from .preprocessing import preprocess

FEATURE_NAMES: list[str] = [
    "entropy",
    "lexical_diversity",
    "dale_chall_readability",
    "flesch_reading_ease",
    "stop_word_count",
    "incorrect_word_count",
    "difficult_word_count",
    "lengthy_word_count",
    "two_letter_word_count",
    "single_letter_word_count",
    "verb_count",
    "adjective_count",
    "noun_count",
]
N_FEATURES = len(FEATURE_NAMES)

_TOKEN_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")
_CONTRACTION_SUFFIXES = ("n't", "'s", "'re", "'ve", "'ll", "'d", "'m")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


def tokenize(clean_text: str) -> list[str]:
    """All tokens of an already-preprocessed string (including purely numeric ones)."""
    return _TOKEN_RE.findall(clean_text)


def word_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if any(c.isalpha() for c in t)]


def _letter_count(token: str) -> int:
    return sum(c.isalpha() for c in token)


# ---------------------------------------------------------------- lexicon lookups
def _stem_candidates(word: str) -> list[str]:
    cands = [word]
    if word.endswith("ies") and len(word) > 4:
        cands.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        cands.append(word[:-2])
    if word.endswith("s") and len(word) > 3:
        cands.append(word[:-1])
    if word.endswith("ed") and len(word) > 3:
        cands += [word[:-2], word[:-1]]
        if len(word) > 4 and word[-3] == word[-4]:
            cands.append(word[:-3])
    if word.endswith("ing") and len(word) > 4:
        cands += [word[:-3], word[:-3] + "e"]
        if len(word) > 5 and word[-4] == word[-5]:
            cands.append(word[:-4])
    if word.endswith("ly") and len(word) > 4:
        cands.append(word[:-2])
    return cands


def _contraction_base(token: str) -> str:
    for suf in _CONTRACTION_SUFFIXES:
        if token.endswith(suf) and len(token) > len(suf):
            return token[: -len(suf)]
    return token.replace("'", "")


def _in_vocab(token: str, vocab: frozenset[str]) -> bool:
    for part in token.split("-"):
        part = _contraction_base(part) if "'" in part else part
        if not part:
            continue
        if not any(c in vocab for c in _stem_candidates(part)):
            return False
    return True


def _is_stop_word(token: str) -> bool:
    return token in resources.stop_words()


def _is_known(token: str) -> bool:
    return _is_stop_word(token) or _in_vocab(token, resources.known_vocabulary())


def _is_easy(token: str) -> bool:
    return _in_vocab(token, resources.easy_words())


# ---------------------------------------------------------------- syllables
def _heuristic_syllables(word: str) -> int:
    word = "".join(c for c in word if c.isalpha())
    if not word:
        return 0
    count = len(_VOWEL_GROUP_RE.findall(word))
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def count_syllables(token: str) -> int:
    """Syllables of a word token: CMUdict if present, otherwise a vowel-group heuristic."""
    cmu = resources.cmu_dict()
    for candidate in (token, token.replace("'", "")):
        if candidate in cmu:
            return max(1, sum(ph[-1].isdigit() for ph in cmu[candidate][0]))
    parts = [p for p in re.split(r"['-]", token) if p]
    if len(parts) > 1 and "-" in token:
        return sum(count_syllables(p) for p in parts)
    return _heuristic_syllables(token)


# ---------------------------------------------------------------- features
def _entropy(words: list[str]) -> float:
    if not words:
        return 0.0
    n = len(words)
    return float(-sum((c / n) * math.log2(c / n) for c in Counter(words).values()))


def extract_features(headline: str) -> np.ndarray:
    """Return the 13 features of ``headline`` as a float64 vector in :data:`FEATURE_NAMES` order."""
    tokens = tokenize(preprocess(headline))
    words = word_tokens(tokens)
    n = len(words)
    if n == 0:
        return np.zeros(N_FEATURES, dtype=np.float64)

    letters = [_letter_count(w) for w in words]
    n_difficult = sum(not _is_easy(w) for w in words)
    pdw = 100.0 * n_difficult / n
    asl = float(n)  # one sentence per headline
    dale_chall = 0.1579 * pdw + 0.0496 * asl + (3.6365 if pdw > 5 else 0.0)
    syllables = sum(count_syllables(w) for w in words)
    flesch = 206.835 - 1.015 * asl - 84.6 * (syllables / n)

    tags = [tag for _, tag in resources.pos_tagger().tag(tokens)]
    tag_of_word = [tag for tok, tag in zip(tokens, tags) if any(c.isalpha() for c in tok)]

    feats = [
        _entropy(words),
        len(set(words)) / n,
        dale_chall,
        flesch,
        sum(_is_stop_word(w) for w in words),
        sum(not _is_known(w) for w in words),
        n_difficult,
        sum(l > 2 for l in letters),
        sum(l == 2 for l in letters),
        sum(l == 1 for l in letters),
        sum(t.startswith("VB") for t in tag_of_word),
        sum(t.startswith("JJ") for t in tag_of_word),
        sum(t.startswith("NN") for t in tag_of_word),
    ]
    out = np.asarray(feats, dtype=np.float64)
    if not np.all(np.isfinite(out)):  # defensive: must never happen
        raise ValueError(f"Non-finite feature value for headline {headline!r}: {out}")
    return out


def extract_features_batch(headlines) -> np.ndarray:
    """Feature matrix of shape (n_headlines, 13)."""
    rows = [extract_features(h) for h in headlines]
    return np.vstack(rows) if rows else np.zeros((0, N_FEATURES))


def features_as_dict(headline: str) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, extract_features(headline).tolist()))

import math

import numpy as np
import pytest

from src.features import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_features,
    extract_features_batch,
    features_as_dict,
)

SAMPLES = [
    "",
    "   ",
    "\t\n",
    "...!?,;:",
    "wow",
    "a",
    "I am ok",
    "don't worry, it isn't over",
    "they're here and we've won",
    "9 deliciously different recipes",
    "2016 2017 2018",
    "asdfgh qwertyuiop zxcvbn",
    "x-ray vision for 24/7 well-known people",
    "dem rep. totally nails why congress is falling short on gender, racial equality",
    "the quick brown fox jumps over the lazy dog",
    "<b>bold</b> https://example.com \U0001F600 headline",
    "word word word word",
]


def feat(text):
    return features_as_dict(text)


def test_feature_order_is_fixed():
    assert FEATURE_NAMES == [
        "entropy", "lexical_diversity", "dale_chall_readability", "flesch_reading_ease",
        "stop_word_count", "incorrect_word_count", "difficult_word_count", "lengthy_word_count",
        "two_letter_word_count", "single_letter_word_count", "verb_count", "adjective_count", "noun_count",
    ]
    assert N_FEATURES == 13


@pytest.mark.parametrize("text", SAMPLES)
def test_exactly_13_finite_values(text):
    v = extract_features(text)
    assert v.shape == (13,)
    assert v.dtype == np.float64
    assert np.all(np.isfinite(v))


@pytest.mark.parametrize("text", ["", "   ", "...!?", "123 456"])
def test_no_word_tokens_gives_zeros(text):
    assert np.all(extract_features(text) == 0.0)


def test_deterministic():
    for text in SAMPLES:
        assert np.array_equal(extract_features(text), extract_features(text))
    a = extract_features_batch(SAMPLES)
    b = extract_features_batch(SAMPLES)
    assert a.shape == (len(SAMPLES), 13) and np.array_equal(a, b)


def test_entropy_and_lexical_diversity():
    assert feat("a b c d")["entropy"] == pytest.approx(2.0)          # 4 equiprobable tokens
    assert feat("cat cat cat cat")["entropy"] == pytest.approx(0.0)
    assert feat("cat dog cat dog")["entropy"] == pytest.approx(1.0)
    assert feat("cat dog cat dog")["lexical_diversity"] == pytest.approx(0.5)
    assert feat("one two three four")["lexical_diversity"] == pytest.approx(1.0)
    assert feat("word word word word")["lexical_diversity"] == pytest.approx(0.25)


def test_letter_length_counts():
    f = feat("I am ok to go home")  # i | am ok to go | home
    assert f["single_letter_word_count"] == 1
    assert f["two_letter_word_count"] == 4
    assert f["lengthy_word_count"] == 1  # paper's literal definition: > 2 characters
    assert feat("cat")["lengthy_word_count"] == 1
    assert feat("at")["lengthy_word_count"] == 0


def test_stop_words():
    assert feat("the cat is on the mat")["stop_word_count"] == 4
    assert feat("zebra giraffe")["stop_word_count"] == 0


def test_flesch_matches_formula():
    # single one-syllable word, one sentence: 206.835 - 1.015*1 - 84.6*1
    assert feat("cat")["flesch_reading_ease"] == pytest.approx(206.835 - 1.015 - 84.6)
    # two words, 1 + 2 syllables ("cat", "water"): mean 1.5
    assert feat("cat water")["flesch_reading_ease"] == pytest.approx(206.835 - 1.015 * 2 - 84.6 * 1.5)


def test_dale_chall_formula():
    # 'the' and 'cat' are easy words -> PDW 0 -> no adjustment
    f = feat("the cat")
    assert f["difficult_word_count"] == 0
    assert f["dale_chall_readability"] == pytest.approx(0.0496 * 2)
    # an unknown word -> PDW 100 > 5 -> adjustment applied
    g = feat("zzzqx")
    assert g["difficult_word_count"] == 1
    assert g["dale_chall_readability"] == pytest.approx(0.1579 * 100 + 0.0496 * 1 + 3.6365)


def test_unknown_words_counted_incorrect():
    assert feat("asdfgh qwertyuiop")["incorrect_word_count"] == 2
    assert feat("the cat sat")["incorrect_word_count"] == 0


def test_contractions_are_single_tokens_and_known():
    f = feat("don't")
    assert f["lexical_diversity"] == 1.0 and f["entropy"] == 0.0  # one token
    assert f["incorrect_word_count"] == 0
    assert f["stop_word_count"] == 1
    assert f["lengthy_word_count"] == 1  # 4 letters
    assert feat("isn't")["incorrect_word_count"] == 0


def test_hyphenated_words_single_token():
    f = feat("well-known")
    assert f["entropy"] == 0.0 and f["lexical_diversity"] == 1.0
    assert f["lengthy_word_count"] == 1


def test_numbers_are_not_word_tokens():
    f = feat("9 deliciously different recipes")
    assert f["lexical_diversity"] == 1.0
    assert f["single_letter_word_count"] == 0  # '9' is not a single-letter word
    assert f["entropy"] == pytest.approx(math.log2(3))  # three equiprobable word tokens


def test_pos_counts():
    f = feat("the quick brown fox jumps over the lazy dog")
    assert f["noun_count"] >= 2
    assert f["adjective_count"] >= 2
    assert f["verb_count"] >= 1
    total = f["noun_count"] + f["adjective_count"] + f["verb_count"]
    assert total <= 9


def test_case_and_markup_invariance():
    assert np.array_equal(extract_features("THE CAT IS HERE"), extract_features("the cat is here"))
    assert np.array_equal(extract_features("<b>the cat</b> https://x.com"), extract_features("the cat"))


def test_finite_on_long_input():
    v = extract_features("word " * 2000)
    assert np.all(np.isfinite(v)) and not math.isnan(v[3])

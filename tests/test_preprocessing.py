from src.preprocessing import preprocess


def test_lowercase_and_whitespace():
    assert preprocess("  Area   MAN\tPasses \n Up  ") == "area man passes up"


def test_urls_removed():
    assert preprocess("read this https://www.theonion.com/abc-123 now") == "read this now"
    assert preprocess("see www.example.com/page today") == "see today"


def test_html_removed_and_entities_decoded():
    assert preprocess("<b>Breaking</b>: cats &amp; dogs") == "breaking : cats & dogs"


def test_emoji_removed():
    assert preprocess("great news \U0001F600\U0001F389 today") == "great news today"


def test_punctuation_and_negation_preserved():
    out = preprocess("He DIDN'T say no, did he?!")
    assert out == "he didn't say no, did he?!"
    assert preprocess("Not good") == "not good"  # negation word kept


def test_curly_apostrophe_normalised():
    assert preprocess("don’t") == "don't"


def test_stop_words_not_removed():
    assert preprocess("the cat is on the mat") == "the cat is on the mat"


def test_no_stemming():
    assert preprocess("running scientists") == "running scientists"


def test_non_string_and_empty():
    assert preprocess(None) == ""
    assert preprocess(123) == ""
    assert preprocess("") == ""


def test_deterministic_and_idempotent():
    s = "Dem Rep. <i>TOTALLY</i> nails it https://x.co \U0001F600"
    assert preprocess(s) == preprocess(s)
    assert preprocess(preprocess(s)) == preprocess(s)

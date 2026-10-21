"""Fixed linguistic resources used by the feature extractor.

Resources (all lower-cased, loaded once):
  * stop words        : NLTK ``stopwords`` corpus, English list (179 entries, incl. contractions)
  * known vocabulary  : union of NLTK ``words`` corpus (en) and the CMU Pronouncing Dictionary
                        headwords (the latter adds inflected forms and many proper nouns)
  * easy-word list    : the 2,941-entry Dale-Chall familiar-word list vendored from the
                        ``textstat`` 0.7.13 package (MIT licence) in
                        ``src/resources/dale_chall_easy_words.txt``
  * pronunciations    : NLTK ``cmudict`` (syllable counts)
  * POS tagger        : NLTK averaged perceptron tagger (``averaged_perceptron_tagger_eng``)

NLTK data are downloaded on first use into the project-local ``nltk_data/`` directory.
NLTK itself is pinned in requirements.txt (3.10.4); the NLTK *data* packages carry no version
number, so ``resource_fingerprint()`` stores content hashes that are saved with the models and
checked at inference time.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import nltk

from .config import PROJECT_ROOT, load_config, project_path

EASY_WORDS_FILE = Path(__file__).resolve().parent / "resources" / "dale_chall_easy_words.txt"

_NLTK_PACKAGES = {
    "stopwords": "corpora/stopwords",
    "words": "corpora/words",
    "cmudict": "corpora/cmudict",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


@lru_cache(maxsize=1)
def ensure_nltk_data() -> Path:
    """Make sure all required NLTK data exist locally (downloading if necessary)."""
    try:
        data_dir = project_path(load_config()["paths"]["nltk_data_dir"])
    except Exception:  # config missing: fall back to the default location
        data_dir = PROJECT_ROOT / "nltk_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_dir))
    for package, locator in _NLTK_PACKAGES.items():
        try:
            nltk.data.find(locator)
        except LookupError:
            ok = nltk.download(package, download_dir=str(data_dir), quiet=True)
            if not ok:
                raise RuntimeError(
                    f"Could not download NLTK package '{package}'. Check your internet connection "
                    f"or run: python -m nltk.downloader -d \"{data_dir}\" {package}"
                )
            nltk.data.find(locator)
    return data_dir


@lru_cache(maxsize=1)
def stop_words() -> frozenset[str]:
    ensure_nltk_data()
    from nltk.corpus import stopwords
    return frozenset(w.lower() for w in stopwords.words("english"))


@lru_cache(maxsize=1)
def cmu_dict() -> dict[str, list[list[str]]]:
    ensure_nltk_data()
    from nltk.corpus import cmudict
    return cmudict.dict()


@lru_cache(maxsize=1)
def known_vocabulary() -> frozenset[str]:
    ensure_nltk_data()
    from nltk.corpus import words
    vocab = {w.lower() for w in words.words()}
    vocab.update(cmu_dict().keys())
    return frozenset(vocab)


@lru_cache(maxsize=1)
def easy_words() -> frozenset[str]:
    if not EASY_WORDS_FILE.is_file():
        raise FileNotFoundError(f"Missing easy-word list: {EASY_WORDS_FILE}")
    with open(EASY_WORDS_FILE, "r", encoding="utf-8") as fh:
        return frozenset(line.strip().lower() for line in fh if line.strip())


@lru_cache(maxsize=1)
def pos_tagger():
    ensure_nltk_data()
    from nltk.tag.perceptron import PerceptronTagger
    return PerceptronTagger(load=True)


def _hash_words(words) -> str:
    return hashlib.sha256("\n".join(sorted(words)).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def resource_fingerprint() -> dict:
    """Content hashes + versions of every resource that influences the features."""
    ensure_nltk_data()
    return {
        "nltk_version": nltk.__version__,
        "stop_words_sha256": _hash_words(stop_words()),
        "stop_words_count": len(stop_words()),
        "known_vocabulary_sha256": _hash_words(known_vocabulary()),
        "known_vocabulary_count": len(known_vocabulary()),
        "easy_words_sha256": _hash_words(easy_words()),
        "easy_words_count": len(easy_words()),
        "cmudict_entries": len(cmu_dict()),
    }

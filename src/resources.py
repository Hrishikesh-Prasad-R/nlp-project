"""Fixed linguistic resources used by the feature extractor.

Resources (all lower-cased, loaded once):
  * stop words        : NLTK ``stopwords`` corpus, English list (198 entries, incl. contractions)
  * known vocabulary  : union of NLTK ``words`` corpus (en, en-basic) and the CMU Pronouncing
                        Dictionary headwords (the latter adds inflected forms and many proper nouns)
  * easy-word list    : the 2,941-entry Dale-Chall familiar-word list vendored from the
                        ``textstat`` 0.7.13 package (MIT licence) in
                        ``src/resources/dale_chall_easy_words.txt``
  * pronunciations    : NLTK ``cmudict`` (syllable counts)
  * POS tagger        : NLTK averaged perceptron tagger (``averaged_perceptron_tagger_eng``)

The NLTK data files are committed in the project-local ``nltk_data/`` directory and are read
directly from there. NLTK's own data lookup (``nltk.data.find``) is deliberately not used: NLTK 3.10
refuses to read from a directory that is writable by, or owned by, another account, which is the
situation of a checked-out repository on Streamlit Community Cloud. If a file is missing it is
downloaded once with ``nltk.download`` as a fallback.

NLTK itself is pinned in requirements.txt (3.10.4); the NLTK *data* carries no version number, so
``resource_fingerprint()`` stores content hashes that are saved with the models and checked at
inference time.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import nltk

from .config import PROJECT_ROOT, load_config, project_path

EASY_WORDS_FILE = Path(__file__).resolve().parent / "resources" / "dale_chall_easy_words.txt"

TAGGER_DIR = "taggers/averaged_perceptron_tagger_eng"
_TAGGER_FILES = [
    "averaged_perceptron_tagger_eng.weights.json",
    "averaged_perceptron_tagger_eng.tagdict.json",
    "averaged_perceptron_tagger_eng.classes.json",
]
# NLTK package name -> files (relative to the data directory) that must exist
_REQUIRED = {
    "stopwords": ["corpora/stopwords/english"],
    "words": ["corpora/words/en", "corpora/words/en-basic"],
    "cmudict": ["corpora/cmudict/cmudict"],
    "averaged_perceptron_tagger_eng": [f"{TAGGER_DIR}/{f}" for f in _TAGGER_FILES],
}


def _data_dir() -> Path:
    try:
        return project_path(load_config()["paths"]["nltk_data_dir"])
    except Exception:  # config missing: fall back to the default location
        return PROJECT_ROOT / "nltk_data"


@lru_cache(maxsize=1)
def ensure_nltk_data() -> Path:
    """Return the data directory after checking that every required file exists.

    Missing packages are downloaded once as a fallback; the files are then read directly.
    """
    data_dir = _data_dir()
    for package, files in _REQUIRED.items():
        if all((data_dir / f).is_file() for f in files):
            continue
        data_dir.mkdir(parents=True, exist_ok=True)
        nltk.download(package, download_dir=str(data_dir), quiet=True)
        missing = [f for f in files if not (data_dir / f).is_file()]
        if missing:
            raise RuntimeError(
                f"NLTK data file(s) {missing} not found in '{data_dir}' and could not be downloaded. "
                f"Restore them from the repository or run: "
                f"python -m nltk.downloader -d \"{data_dir}\" {package}"
            )
    return data_dir


def _lines(relative: str) -> list[str]:
    """Lines of a bundled data file (split like NLTK's line tokenizer, blank lines removed)."""
    path = ensure_nltk_data() / relative
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


@lru_cache(maxsize=1)
def stop_words() -> frozenset[str]:
    return frozenset(w.lower() for w in _lines("corpora/stopwords/english"))


@lru_cache(maxsize=1)
def cmu_dict() -> dict[str, list[list[str]]]:
    """word -> list of pronunciations (each a list of phonemes), as ``cmudict.dict()`` returns."""
    out: dict[str, list[list[str]]] = {}
    for line in _lines("corpora/cmudict/cmudict"):
        pieces = line.split()
        out.setdefault(pieces[0].lower(), []).append(pieces[2:])
    return out


@lru_cache(maxsize=1)
def known_vocabulary() -> frozenset[str]:
    vocab = {w.lower() for name in ("en", "en-basic") for w in _lines(f"corpora/words/{name}")}
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
    """The NLTK averaged perceptron tagger, built from the bundled JSON model files."""
    from nltk.tag.perceptron import PerceptronTagger

    model_dir = ensure_nltk_data() / TAGGER_DIR
    tagger = PerceptronTagger(load=False)
    params = []
    for name in tagger.param_files("eng"):
        with open(model_dir / name, "r", encoding="utf-8") as fh:
            params.append(json.load(fh))
    tagger.decode_json_params(params)
    return tagger


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

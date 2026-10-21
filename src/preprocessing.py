"""Deterministic text preprocessing, used identically in training and inference.

Operations, applied in this order:
 1. Non-string / None input becomes the empty string.
 2. HTML entities are decoded (``&amp;`` -> ``&``).
 3. HTML tags are removed (``<b>word</b>`` -> `` word ``).
 4. URLs (``http(s)://...``, ``www....``) are removed.
 5. Emoji and pictographic symbols (incl. variation selectors / ZWJ) are removed.
 6. Typographic apostrophes/quotes are normalised to ASCII (``’`` -> ``'``) so contractions match.
 7. Text is lower-cased.
 8. Runs of whitespace collapse to one space; leading/trailing whitespace is stripped.

Deliberately NOT done: punctuation removal, negation removal, stop-word removal (stop-word
count is a feature), stemming and lemmatisation. The original headline is kept untouched by
the callers for display.
"""
from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U0001F300-\U0001FAFF"  # pictographs, emoticons, transport, supplemental symbols
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U0001F000-\U0001F2FF"  # mahjong, cards, enclosed
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero width joiner
    "\U000020E3"             # keycap
    "]+"
)
_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"'})
_WS_RE = re.compile(r"\s+")


def preprocess(text) -> str:
    """Return the preprocessed form of ``text`` (see module docstring)."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMOJI_RE.sub("", text)
    text = text.translate(_QUOTE_MAP)
    text = text.lower()
    return _WS_RE.sub(" ", text).strip()

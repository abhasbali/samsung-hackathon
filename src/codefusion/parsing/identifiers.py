"""Code-aware identifier decomposition and tokenization.

Examples
--------
>>> split_identifier("getUserAuthenticationToken")
['get', 'User', 'Authentication', 'Token']
>>> split_identifier("HTTPRequest")
['HTTP', 'Request']
>>> {"OAuth", "OAuth2", "Token"} <= identifier_variants("OAuth2Token")
True

The tokenizer always preserves the original identifier (lower-cased) *and*
emits its components, so an exact-identifier query and a natural-language
query ("user authentication token") can both match.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Iterable
from functools import lru_cache

_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_SEGMENT_SPLIT_RE = re.compile(r"[_$\W]+")
# Acronym followed by a capitalised word | optional capital + lowercase run | capital run | digits
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just let me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she should
    so some such than that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you your yours
    yourself yourselves also given get gets using use used way code function functions snippet snippets""".split()
)

# Keywords that carry no retrieval signal on their own. ``return``/``raise``/``import`` are
# intentionally *kept*: they matter for structural matching ("what does X return?").
CODE_STOPWORDS: frozenset[str] = frozenset(
    {"self", "cls", "this", "def", "var", "let", "const", "pass", "none", "null", "true", "false",
     "elif", "else", "then", "end", "public", "private", "protected", "static", "void", "new", "in", "is",
     "and", "or", "not", "as", "if", "for", "while", "do", "int", "str", "print"}
)

_PY_KEYWORDS = frozenset(k.lower() for k in keyword.kwlist)


def split_identifier(identifier: str) -> list[str]:
    """Split an identifier into its semantic components, preserving case.

    Handles camelCase, PascalCase, snake_case, SCREAMING_SNAKE_CASE, digits and
    acronyms. A single leading capital directly followed by a capitalised word is
    merged (``OAuth`` stays one component instead of ``O`` + ``Auth``).
    """
    parts: list[str] = []
    for segment in _SEGMENT_SPLIT_RE.split(identifier):
        if not segment:
            continue
        pieces = _CAMEL_RE.findall(segment)
        merged: list[str] = []
        i = 0
        while i < len(pieces):
            p = pieces[i]
            nxt = pieces[i + 1] if i + 1 < len(pieces) else None
            if (
                len(p) == 1
                and p.isupper()
                and nxt is not None
                and len(nxt) > 1
                and nxt[0].isupper()
                and nxt[1:].islower()
            ):
                merged.append(p + nxt)  # OAuth, IPhone, ...
                i += 2
                continue
            merged.append(p)
            i += 1
        parts.extend(merged)
    return parts


@lru_cache(maxsize=200_000)
def identifier_variants(identifier: str) -> frozenset[str]:
    """Return the original identifier plus all useful component forms (case preserved)."""
    out: set[str] = {identifier}
    parts = split_identifier(identifier)
    out.update(parts)
    # Also keep the unmerged camel pieces (``AString`` -> ``A`` + ``String``).
    for segment in _SEGMENT_SPLIT_RE.split(identifier):
        if segment:
            out.update(_CAMEL_RE.findall(segment))
    # Re-attach digits to the preceding component: OAuth + 2 -> OAuth2, sha + 256 -> sha256.
    for a, b in zip(parts, parts[1:]):
        if b.isdigit() and not a.isdigit():
            out.add(a + b)
    return frozenset(p for p in out if p)


def _simple_stem(token: str) -> str:
    """Tiny deterministic suffix stripper used only when PyStemmer is unavailable."""
    for suffix in ("ations", "ation", "ingly", "ings", "ing", "edly", "ies", "ers", "ed", "es", "er", "ly", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            return stem + "y" if suffix == "ies" else stem
    return token


@lru_cache(maxsize=1)
def _get_stemmer():  # pragma: no cover - trivial
    try:
        import Stemmer  # PyStemmer

        return Stemmer.Stemmer("english")
    except Exception:  # noqa: BLE001 - optional dependency
        return None


@lru_cache(maxsize=500_000)
def stem(token: str) -> str:
    stemmer = _get_stemmer()
    if stemmer is not None:
        return stemmer.stemWord(token)
    return _simple_stem(token)


class CodeTokenizer:
    """Tokenizer for both code and natural-language queries.

    Parameters
    ----------
    lowercase: lower-case all tokens.
    split_identifiers: emit identifier components in addition to the original identifier.
    stem_tokens: apply English stemming (Snowball via PyStemmer if installed).
    remove_stopwords: drop English stopwords and low-signal code keywords.
    min_token_len: drop tokens shorter than this.
    """

    def __init__(
        self,
        lowercase: bool = True,
        split_identifiers: bool = True,
        stem_tokens: bool = True,
        remove_stopwords: bool = True,
        min_token_len: int = 2,
        keep_numbers: bool = False,
    ) -> None:
        self.lowercase = lowercase
        self.split_identifiers = split_identifiers
        self.stem_tokens = stem_tokens
        self.remove_stopwords = remove_stopwords
        self.min_token_len = min_token_len
        self.keep_numbers = keep_numbers
        self._stop = ENGLISH_STOPWORDS | CODE_STOPWORDS

    def _emit(self, token: str) -> str | None:
        t = token.lower() if self.lowercase else token
        if len(t) < self.min_token_len:
            return None
        if not self.keep_numbers and t.isdigit():
            return None
        if self.remove_stopwords and t.lower() in self._stop:
            return None
        if self.stem_tokens:
            t = stem(t)
        return t

    def tokenize(self, text: str, dedupe: bool = False) -> list[str]:
        tokens: list[str] = []
        for m in _IDENT_RE.finditer(text):
            ident = m.group(0)
            if self.split_identifiers:
                variants = identifier_variants(ident)
                # Original first (keeps exact identifiers matchable), then components.
                ordered = [ident] + sorted(v for v in variants if v != ident)
            else:
                ordered = [ident]
            for v in ordered:
                e = self._emit(v)
                if e is not None:
                    tokens.append(e)
        if dedupe:
            seen: set[str] = set()
            tokens = [t for t in tokens if not (t in seen or seen.add(t))]
        return tokens

    def tokenize_many(self, texts: Iterable[str], dedupe: bool = False) -> list[list[str]]:
        return [self.tokenize(t, dedupe=dedupe) for t in texts]


def normalize_symbol(name: str) -> str:
    """Case/underscore-insensitive symbol key: ``validate_user`` == ``validateUser`` == ``ValidateUser``."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def extract_identifiers(text: str) -> list[str]:
    """Return raw identifiers appearing in ``text`` (keywords removed), preserving order."""
    return [m.group(0) for m in _IDENT_RE.finditer(text) if m.group(0).lower() not in _PY_KEYWORDS]


def looks_like_code_identifier(token: str) -> bool:
    """Heuristic: does ``token`` look like a code identifier rather than an English word?"""
    if "_" in token.strip("_") or "." in token:
        return True
    if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
        return True
    # camelCase / PascalCase with an inner capital, or ALLCAPS of length >= 2
    if re.search(r"[a-z][A-Z]", token):
        return True
    if token.isupper() and len(token) >= 2:
        return True
    return token.endswith("()")

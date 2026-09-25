"""Deterministic query expansion (no LLM).

Two sources, both optional and ablated:

1. A small curated map of programming concepts to identifier vocabulary
   ("authentication" -> auth, login, token, jwt, verify ...).
2. Corpus-vocabulary expansion: indexed identifiers whose *components* contain a query word
   (query "retries" -> ``MAX_RETRIES``, ``retry_with_backoff``), capped and ranked by rarity.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable

from codefusion.parsing.identifiers import ENGLISH_STOPWORDS, split_identifier, stem

CONCEPTS: dict[str, list[str]] = {
    "auth": ["auth", "authenticate", "login", "token", "jwt", "password", "credential", "session", "verify"],
    "authent": ["auth", "authenticate", "login", "token", "jwt", "password", "credential", "verify"],
    "login": ["login", "auth", "authenticate", "password", "session"],
    "databas": ["db", "database", "sql", "query", "cursor", "commit", "insert", "save", "persist", "repository"],
    "persist": ["save", "store", "db", "write", "insert"],
    "save": ["save", "store", "persist", "insert", "write", "commit"],
    "config": ["config", "settings", "env", "environ", "option", "constant"],
    "configur": ["config", "settings", "env", "environ", "option", "constant"],
    "set": ["config", "settings"],
    "normal": ["normalize", "clean", "strip", "lower", "sanitize", "preprocess", "transform"],
    "preprocess": ["preprocess", "normalize", "clean", "tokenize", "transform", "validate"],
    "valid": ["validate", "check", "verify", "is_valid", "assert", "schema"],
    "error": ["error", "exception", "raise", "fail", "except"],
    "except": ["exception", "raise", "error", "except"],
    "retri": ["retry", "retries", "backoff", "attempt"],
    "retry": ["retry", "retries", "backoff", "attempt"],
    "cach": ["cache", "memo", "lru", "ttl"],
    "request": ["request", "http", "client", "fetch", "get", "post", "response"],
    "log": ["log", "logger", "logging"],
    "pars": ["parse", "tokenize", "split", "read", "decode"],
    "predict": ["predict", "model", "inference", "score"],
    "hash": ["hash", "digest", "sha", "md5", "bcrypt"],
    "encrypt": ["encrypt", "decrypt", "cipher", "key"],
    "user": ["user", "account", "profile"],
}


def concept_expansion(text: str, max_terms: int = 8) -> list[str]:
    words = {stem(w.lower()) for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in ENGLISH_STOPWORDS}
    out: list[str] = []
    for w in sorted(words):
        for key, terms in CONCEPTS.items():
            if w.startswith(key) or key.startswith(w) and len(w) >= 4:
                out.extend(terms)
    seen: set[str] = set()
    q = {w for w in re.findall(r"[a-z]+", text.lower())}
    res = [t for t in out if t not in q and not (t in seen or seen.add(t))]
    return res[:max_terms]


class VocabularyExpander:
    """Maps query words to indexed identifiers that contain them as a component."""

    def __init__(self) -> None:
        self.by_component: dict[str, set[str]] = defaultdict(set)
        self.df: dict[str, int] = defaultdict(int)
        self.n_docs = 0

    def add_identifiers(self, identifiers_per_doc: Iterable[Iterable[str]]) -> None:
        for idents in identifiers_per_doc:
            self.n_docs += 1
            for ident in set(idents):
                parts = split_identifier(ident)
                if len(parts) < 2:
                    continue
                self.df[ident] += 1
                for p in parts:
                    if len(p) >= 3:
                        self.by_component[stem(p.lower())].add(ident)

    def expand(self, text: str, max_terms: int = 8) -> list[str]:
        scored: dict[str, float] = {}
        for w in re.findall(r"[A-Za-z]+", text):
            lw = w.lower()
            if lw in ENGLISH_STOPWORDS or len(lw) < 4:
                continue
            for ident in self.by_component.get(stem(lw), ()):
                idf = math.log(1 + self.n_docs / (1 + self.df[ident]))
                scored[ident] = scored.get(ident, 0.0) + idf
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        return [k for k, _ in ranked[:max_terms]]

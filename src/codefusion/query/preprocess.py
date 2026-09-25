"""Query preprocessing: cleaning, identifier extraction, symbol candidates, intent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from codefusion.config.schema import QueryConfig
from codefusion.parsing.identifiers import (
    ENGLISH_STOPWORDS,
    extract_identifiers,
    looks_like_code_identifier,
    normalize_symbol,
)
from codefusion.query.classifier import RuleBasedIntentClassifier

# Section headers used by competitive-programming statements (AppsRetrieval / CodeContests style).
_ERROR_WORDS = frozenset({"error", "errors", "exception", "exceptions", "raised", "raise", "raises", "thrown", "throw",
                          "fails", "fail", "failed", "failure", "which", "what", "when"})
JOIN_CONNECTORS = frozenset({"with", "to", "by", "for", "of", "from", "and", "or", "in", "on", "at", "as", "if"})
_EXAMPLES_RE = re.compile(r"\n\s*-{3,}\s*(Examples?|Sample Input|Example Input)\s*-{3,}.*?(?=\n\s*-{3,}\s*Note\s*-{3,}|\Z)", re.S | re.I)
_ALT_EXAMPLES_RE = re.compile(r"\n\s*(Examples?|Sample Input\s*\d*)\s*:?\s*\n.*", re.S)


@dataclass
class ProcessedQuery:
    original: str
    text: str
    dense_text: str
    intent: str = "GENERAL"
    intent_confidence: float = 0.0
    intent_evidence: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    symbol_candidates: list[tuple[str, float]] = field(default_factory=list)
    bm25_tokens: list[str] = field(default_factory=list)
    expansion_terms: list[str] = field(default_factory=list)
    version: str | None = None
    dense_vector: np.ndarray | None = None

    def summary(self) -> dict:
        return {
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "intent_evidence": self.intent_evidence[:6],
            "identifiers": self.identifiers[:12],
            "target_symbols": self.target_symbols[:8],
            "expansion_terms": self.expansion_terms[:12],
            "bm25_tokens": self.bm25_tokens[:30],
        }


def strip_examples(text: str) -> str:
    """Remove sample input/output sections from programming-problem statements."""
    out = _EXAMPLES_RE.sub("\n", text)
    if out == text:
        m = _ALT_EXAMPLES_RE.search(text)
        if m and m.start() > len(text) * 0.4:
            out = text[: m.start()]
    return out.strip()


class QueryPreprocessor:
    def __init__(self, cfg: QueryConfig) -> None:
        self.cfg = cfg
        self.classifier = RuleBasedIntentClassifier() if cfg.classifier == "rules" else None

    def symbol_candidates(self, text: str, identifiers: list[str], targets: list[str],
                          intent: str | None = None) -> list[tuple[str, float]]:
        cands: dict[str, float] = {}

        def add(sym: str, w: float) -> None:
            ns = normalize_symbol(sym)
            if len(ns) >= 3:
                cands[ns] = max(cands.get(ns, 0.0), w)

        for t in targets:
            add(t, 1.5)
            if "." in t:
                add(t.rsplit(".", 1)[-1], 1.2)
        for ident in identifiers:
            add(ident, 1.0)
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", text)]
        # Adjacent word joins: "validate user" -> validateuser (matches validate_user / validateUser).
        # Long natural-language statements are capped to keep the candidate list focused.
        words_for_joins = words[:80]
        for n in (2, 3):
            for i in range(len(words_for_joins) - n + 1):
                grams = words_for_joins[i : i + n]
                # Connectors may sit *inside* a join (retry_with_backoff, save_to_db, get_by_id).
                if grams[0].lower() in ENGLISH_STOPWORDS or grams[-1].lower() in ENGLISH_STOPWORDS:
                    continue
                if any(g.lower() in ENGLISH_STOPWORDS and g.lower() not in JOIN_CONNECTORS for g in grams[1:-1]):
                    continue
                add("".join(grams), 0.8)
        for w in words[:80]:
            if w.lower() not in ENGLISH_STOPWORDS and len(w) >= 4:
                add(w, 0.35)
        if intent == "ERROR":
            # Exception classes are conventionally <Noun>Error / <Noun>Exception:
            # "database connection fails" -> DatabaseError, ConnectionError.
            for w in words[:40]:
                lw = w.lower()
                if lw not in ENGLISH_STOPWORDS and len(lw) >= 3 and lw not in _ERROR_WORDS:
                    add(lw + "error", 0.7)
                    add(lw + "exception", 0.7)
        return sorted(cands.items(), key=lambda kv: -kv[1])[:60]

    def process(self, query: str, version: str | None = None) -> ProcessedQuery:
        original = query
        text = " ".join(query.split()) if len(query) < 400 else query.strip()
        text = text[: self.cfg.max_query_chars]
        dense_text = strip_examples(text) if self.cfg.strip_examples else text
        identifiers = [t for t in extract_identifiers(text) if looks_like_code_identifier(t)]
        identifiers += [t.strip("`").rstrip("()") for t in re.findall(r"`[^`]+`", text)]
        identifiers = list(dict.fromkeys(identifiers))
        intent, conf, evidence, targets = "GENERAL", 0.0, [], []
        if self.classifier is not None:
            r = self.classifier.classify(text)
            intent, conf, evidence, targets = r.intent, r.confidence, r.evidence, r.target_symbols
        return ProcessedQuery(
            original=original,
            text=text,
            dense_text=dense_text,
            intent=intent,
            intent_confidence=conf,
            intent_evidence=evidence,
            identifiers=identifiers,
            target_symbols=targets,
            symbol_candidates=self.symbol_candidates(text, identifiers, targets, intent),
            version=version,
        )

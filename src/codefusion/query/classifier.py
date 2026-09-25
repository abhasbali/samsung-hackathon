"""Deterministic, code-specific query intent classifier.

Each intent owns weighted regex rules; the intent with the highest total wins (ties are
broken by a fixed priority). The classifier also extracts *target symbols* (e.g.
``validate_user`` in "Who calls validate_user?") used by symbol/graph retrieval.

No LLM is involved; classification costs microseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from codefusion.parsing.identifiers import ENGLISH_STOPWORDS, looks_like_code_identifier

_ID = r"`?([A-Za-z_][\w.]*(?:\(\))?)`?"

# (pattern, weight) per intent. Patterns are matched case-insensitively on the query.
RULES: dict[str, list[tuple[str, float]]] = {
    "EVOLUTION": [
        (r"\b(how|what) (did|has|have)\b.*\b(change|evolve|evolved|changed)\b", 3.0),
        (r"\b(changed?|changes|evolv\w*|history|historical|previous(ly)?|older|earlier) (version|versions|commit|commits|implementation)\b", 2.5),
        (r"\bbetween (versions|commits|releases)\b", 3.0),
        (r"\b(over time|across (versions|commits)|used to|was renamed|renamed|refactor(ed|ing)?|regression|introduced in|last commit|git (log|history))\b", 2.0),
        (r"\b(version|commit)s?\b", 0.8),
    ],
    "CALLER": [
        (r"\b(who|what|which)( \w+){0,2} (calls?|invokes?|triggers?)\b", 3.0),
        (r"\bcallers? of\b", 3.0),
        (r"\b(called|invoked) (by|from)\b", 2.5),
        (r"\bwhere (is|are) " + _ID + r" (called|invoked)\b", 3.0),
    ],
    "CALLEE": [
        (r"\bwhat (functions?|methods?|apis?)( \w+){0,2} (does|do) \S+ (call|invoke|use)\b", 3.5),
        (r"\bwhat does \S+ call\b", 3.0),
        # "functions (are) called by main" = main's callees (outscores CALLER's generic "called by")
        (r"\b(callees?|calls made by|called (inside|within|in)|(functions?|methods?) (are |is )?called (by|in|inside|from|within))\b", 3.0),
        # "... called before/after X" asks about ordering along a flow, not X's callees.
        (r"\b(what|which) (functions?|methods?) (are|is) called\b(?! (before|after|prior))", 2.5),
    ],
    "DEFINITION": [
        # "where is X implemented" is an IMPLEMENTATION question (see IMPLEMENTATION rules).
        (r"\bwhere (is|are) " + _ID + r" (defined|declared|created)\b", 3.0),
        (r"\b(definition|declaration) of\b", 3.0),
        (r"\b(defined|declared)\b", 1.5),
        (r"\b(find|show)( me)? the (class|function|method|constant|variable|type|struct|interface)\b", 1.5),
        (r"^\s*(what|where) is (the )?`?[A-Za-z_]\w*(\(\))?`?\s*\??\s*$", 1.5),
    ],
    "CONFIGURATION": [
        (r"\b(config|configuration|configured|configure|settings?|setting up|env(ironment)? var\w*|environment|feature flag|flags?|options?|constants?|defaults?)\b", 1.5),
        (r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b", 1.5),
        (r"\b(timeout|max(imum)? retr\w*|port|host|url|api key|threshold|limit)\b", 1.0),
        (r"\.(ya?ml|toml|ini|env|json)\b", 1.5),
    ],
    "ERROR": [
        (r"\b(errors?|exceptions?|raises?|raised|throws?|thrown|fail(s|ed|ure|ing)?|crash\w*|traceback|invalid|catch(es)?|handled?|handling)\b", 1.2),
        (r"\b(error|exception) handling\b", 2.0),
        (r"\bwhat happens (when|if)\b", 1.0),
    ],
    "DEPENDENCY": [
        (r"\b(import(s|ed)?|librar(y|ies)|packages?|dependenc(y|ies)|depends on|third[- ]party|module)\b", 1.5),
        (r"\b(redis|postgres\w*|mysql|sqlite|mongo\w*|kafka|rabbitmq|s3|boto3|requests|numpy|pandas|torch|jwt|flask|django|fastapi|react|express)\b", 1.3),
        (r"\bwhere (is|are) \S+ used\b", 0.8),
    ],
    "USAGE": [
        (r"\bwhere (is|are) " + _ID + r" (used|referenced|accessed|read|written|set)\b", 3.0),
        (r"\b(usages?|uses|references?) of\b", 3.0),
        (r"\bhow (do|to|can) (i|we|you)? ?use\b", 2.0),
        (r"\bexamples? of (using|calling)\b", 2.0),
    ],
    "DATA_FLOW": [
        (r"\b(before|after|prior to)\b.*\b(goes|go|passed|sent|reaches|saved|stored|returned|going|processing|prediction|predict|main|authenticat\w*|written)\b", 2.5),
        (r"\b(data ?flow|flows?|pipeline|propagat\w*|passed to|passes to|transform\w*|pre-?process\w*|post-?process\w*)\b", 1.8),
        (r"\bhow (is|are) (the )?(input|data|request|payload|value|user input)s? (processed|validated|normali[sz]ed|transformed|cleaned|parsed|sanitized|handled)\b", 3.0),
        (r"\b(input|output) (is|are)? ?(validated|normali[sz]ed|sanitized|parsed)\b", 1.5),
    ],
    "IMPLEMENTATION": [
        (r"\bhow (is|are|does|do) .* (implemented|work|works|computed|calculated|done|handled|performed)\b", 2.0),
        (r"\b(implementation|implement\w*|logic|algorithm|compute\w*|calculat\w*)\b", 1.2),
        (r"\bwhere is \S+ (implemented|handled)\b", 1.5),
    ],
}

PRIORITY = [
    "EVOLUTION", "CALLER", "CALLEE", "DEFINITION", "USAGE", "DATA_FLOW", "CONFIGURATION",
    "ERROR", "DEPENDENCY", "IMPLEMENTATION", "GENERAL",
]

_COMPILED = {k: [(re.compile(p, re.I), w) for p, w in v] for k, v in RULES.items()}
# The SCREAMING_CASE constant pattern must stay case-sensitive.
_COMPILED["CONFIGURATION"][1] = (re.compile(RULES["CONFIGURATION"][1][0]), RULES["CONFIGURATION"][1][1])

_TARGET_PATTERNS = [
    re.compile(r"\bcalls?\s+" + _ID, re.I),
    re.compile(r"\bdoes\s+" + _ID + r"\s+(?:call|invoke|use)\b", re.I),
    re.compile(r"\b(?:where|how) (?:is|are)\s+" + _ID + r"\s+(?:defined|declared|implemented|used|called|configured|set|referenced|created|handled)\b", re.I),
    re.compile(r"\b(?:definition|declaration|callers?|callees?|usages?|uses|references?) of\s+" + _ID, re.I),
    re.compile(r"\b(?:called|invoked) (?:by|from|in|inside)\s+" + _ID, re.I),
    re.compile(r"\bbefore (?:the )?" + _ID + r"\b", re.I),
]

LONG_QUERY_CHARS = 400


@dataclass
class IntentResult:
    intent: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)


class RuleBasedIntentClassifier:
    def classify(self, query: str) -> IntentResult:
        q = query.strip()
        targets = self.extract_targets(q)
        if len(q) >= LONG_QUERY_CHARS:
            # Long problem statements (e.g. AppsRetrieval) describe behaviour to implement;
            # keyword rules would fire on incidental words ("error", "input"), so skip them.
            return IntentResult("IMPLEMENTATION", 0.5, {"IMPLEMENTATION": 1.0}, ["long_problem_statement"], targets)
        scores: dict[str, float] = {}
        evidence: list[str] = []
        for intent, rules in _COMPILED.items():
            s = 0.0
            for rx, w in rules:
                m = rx.search(q)
                if m:
                    s += w
                    evidence.append(f"{intent}:{m.group(0)[:40]}")
            if s > 0:
                scores[intent] = s
        if not scores:
            return IntentResult("GENERAL", 0.0, {}, [], targets)
        best = max(scores.items(), key=lambda kv: (kv[1], -PRIORITY.index(kv[0])))
        total = sum(scores.values())
        return IntentResult(best[0], round(best[1] / total, 3), scores, evidence, targets)

    @staticmethod
    def extract_targets(q: str) -> list[str]:
        out: list[str] = []
        for rx in _TARGET_PATTERNS:
            for m in rx.finditer(q):
                t = m.group(1).rstrip("()").strip(".")
                if t and t.lower() not in ENGLISH_STOPWORDS and t.lower() not in {"is", "are", "the", "it", "function", "method", "class", "input", "user"}:
                    out.append(t)
        for tok in re.findall(r"`([^`]+)`", q):
            out.append(tok.rstrip("()"))
        for tok in re.findall(r"[A-Za-z_][\w.]*(?:\(\))?", q):
            if looks_like_code_identifier(tok):
                out.append(tok.rstrip("()").strip("."))
        seen: set[str] = set()
        return [t for t in out if not (t in seen or seen.add(t))]

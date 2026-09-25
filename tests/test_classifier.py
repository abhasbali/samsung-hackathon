import pytest

from codefusion.query.classifier import RuleBasedIntentClassifier
from codefusion.query.preprocess import strip_examples

CASES = [
    ("Where is MAX_RETRIES defined?", "DEFINITION"),
    ("Who calls preprocess?", "CALLER"),
    ("Who calls validate_user?", "CALLER"),
    ("What functions does preprocess call?", "CALLEE"),
    ("How is input transformed before authentication?", "DATA_FLOW"),
    ("How is input normalized before prediction?", "DATA_FLOW"),
    ("How did authentication change?", "EVOLUTION"),
    ("How did authentication change between versions?", "EVOLUTION"),
    ("Where is MAX_RETRIES configured?", "CONFIGURATION"),
    ("What exception is raised when the token expires?", "ERROR"),
    ("Where is the redis client used?", "DEPENDENCY"),
    ("Where is save_user used?", "USAGE"),
    ("How is password hashing implemented?", "IMPLEMENTATION"),
    ("Where is authentication implemented?", "IMPLEMENTATION"),
    ("What functions are called before the user is saved?", "DATA_FLOW"),
    ("Which functions are called by main?", "CALLEE"),
    ("sorting numbers", "GENERAL"),
]


@pytest.mark.parametrize("query,intent", CASES)
def test_intents(query, intent):
    assert RuleBasedIntentClassifier().classify(query).intent == intent


def test_redis_usage_is_dependency_or_usage():
    assert RuleBasedIntentClassifier().classify("Where is Redis used?").intent in ("DEPENDENCY", "USAGE")


def test_target_symbol_extraction():
    c = RuleBasedIntentClassifier()
    assert "validate_user" in c.classify("Who calls validate_user?").target_symbols
    assert "preprocess" in c.classify("What functions does preprocess call?").target_symbols
    assert "MAX_RETRIES" in c.classify("Where is MAX_RETRIES defined?").target_symbols


def test_long_problem_statement_is_not_misclassified():
    q = "You are given an array. " * 30 + "Print an error if the input is invalid."
    r = RuleBasedIntentClassifier().classify(q)
    assert r.intent == "IMPLEMENTATION" and "long_problem_statement" in r.evidence


def test_strip_examples():
    q = "Find the max.\n\n-----Input-----\nOne int.\n\n-----Examples-----\nInput\n3\nOutput\n3\n\n-----Note-----\nEasy."
    out = strip_examples(q)
    assert "Examples" not in out and "Input" in out and "Note" in out


def test_error_queries_propose_exception_class_names():
    from codefusion.config.schema import QueryConfig
    from codefusion.query.preprocess import QueryPreprocessor

    pq = QueryPreprocessor(QueryConfig()).process("Which errors are raised when the database connection fails?")
    assert pq.intent == "ERROR"
    syms = dict(pq.symbol_candidates)
    assert "databaseerror" in syms and "connectionerror" in syms
    assert "errorserror" not in syms

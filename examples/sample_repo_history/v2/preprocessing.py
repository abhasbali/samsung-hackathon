"""Input preprocessing applied before prediction."""

import re

ALLOWED_FIELDS = ("age", "income", "country")


def normalize_input(raw):
    """Normalize a raw request payload: trim strings, lower-case keys, drop unknown fields."""
    cleaned = {}
    for key, value in raw.items():
        key = key.strip().lower()
        if key not in ALLOWED_FIELDS:
            continue
        if isinstance(value, str):
            value = re.sub(r"\s+", " ", value.strip())
        cleaned[key] = value
    return cleaned


def validate_input(record):
    """Reject records without an age."""
    if "age" not in record:
        raise ValueError("missing field: age")
    return record

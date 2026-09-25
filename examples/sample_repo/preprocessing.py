"""Input preprocessing applied before prediction."""

import re

from errors import ValidationError

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
    """Validate a normalized record and raise ValidationError on bad values."""
    if "age" not in record:
        raise ValidationError("missing field: age")
    age = int(record["age"])
    if age < 0 or age > 130:
        raise ValidationError(f"age out of range: {age}")
    record["age"] = age
    return record


def preprocess_input(raw):
    """Full preprocessing pipeline: normalize, then validate."""
    record = normalize_input(raw)
    return validate_input(record)

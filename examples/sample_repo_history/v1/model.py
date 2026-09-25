"""A tiny scoring model."""

import math

WEIGHTS = {"age": 0.03, "income": 0.00002}
BIAS = -1.5


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def predict(record):
    """Return the probability score for a validated record."""
    z = BIAS + sum(WEIGHTS[k] * float(record.get(k, 0)) for k in WEIGHTS)
    return sigmoid(z)

"""Application configuration."""

import os

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.5
DATABASE_URL = os.environ.get("APP_DATABASE_URL", "sqlite:///app.db")
TOKEN_TTL_SECONDS = 3600
SECRET_KEY = os.environ.get("APP_SECRET_KEY", "change-me")


def load_config():
    """Return runtime settings as a dictionary."""
    return {"max_retries": MAX_RETRIES, "database_url": DATABASE_URL, "token_ttl": TOKEN_TTL_SECONDS}

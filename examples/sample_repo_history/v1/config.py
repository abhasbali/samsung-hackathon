"""Application configuration."""

MAX_RETRIES = 3
DATABASE_URL = "sqlite:///app.db"
SECRET_KEY = "change-me"


def load_config():
    """Return runtime settings as a dictionary."""
    return {"max_retries": MAX_RETRIES, "database_url": DATABASE_URL}

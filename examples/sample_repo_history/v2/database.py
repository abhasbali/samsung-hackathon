"""Persistence layer with retrying writes."""

import sqlite3
import time

from config import MAX_RETRIES, RETRY_BACKOFF_SECONDS
from errors import DatabaseError


def connect(url):
    """Open a SQLite connection from a sqlite:/// URL."""
    return sqlite3.connect(url.replace("sqlite:///", ""))


def with_retries(operation):
    """Run a database operation, retrying up to MAX_RETRIES times."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except sqlite3.OperationalError:
            time.sleep(RETRY_BACKOFF_SECONDS)
    raise DatabaseError(f"operation failed after {MAX_RETRIES} retries")


class UserRepository:
    """Stores and loads users."""

    def __init__(self, conn):
        self.conn = conn

    def find_user(self, username):
        """Load a user row by username, or None."""
        row = self.conn.execute(
            "SELECT username, salt, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            return None
        return {"username": row[0], "salt": row[1], "password_hash": row[2]}

    def save_user(self, user):
        """Insert or update a user, retrying on transient database errors."""
        def _write():
            self.conn.execute(
                "INSERT OR REPLACE INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
                (user["username"], user["salt"], user["password_hash"]),
            )
            self.conn.commit()

        return with_retries(_write)

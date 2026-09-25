"""Persistence layer."""

import sqlite3


def connect(url):
    """Open a SQLite connection from a sqlite:/// URL."""
    return sqlite3.connect(url.replace("sqlite:///", ""))


def save_user(conn, user):
    """Insert or update a user."""
    conn.execute(
        "INSERT OR REPLACE INTO users (username, password) VALUES (?, ?)",
        (user["username"], user["password"]),
    )
    conn.commit()

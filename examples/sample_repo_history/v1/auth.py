"""User authentication."""


def authenticate(users, username, password):
    """Authenticate a user by comparing the stored plain-text password."""
    user = users.get(username)
    if user is None or user["password"] != password:
        return None
    return username

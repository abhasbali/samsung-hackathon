"""User authentication: password hashing and signed session tokens."""

import hashlib
import hmac
import time

from config import SECRET_KEY, TOKEN_TTL_SECONDS
from errors import AuthenticationError


def hash_password(password, salt):
    """Derive a password hash with PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def issue_token(username):
    """Create a signed token of the form 'username:expiry:signature'."""
    expiry = int(time.time()) + TOKEN_TTL_SECONDS
    payload = f"{username}:{expiry}"
    signature = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


class AuthService:
    """Authenticates users against the user repository."""

    def __init__(self, repository):
        self.repository = repository

    def authenticate(self, username, password):
        """Authenticate a user by username and password and return a session token."""
        user = self.repository.find_user(username)
        if user is None:
            raise AuthenticationError("unknown user")
        if hash_password(password, user["salt"]) != user["password_hash"]:
            raise AuthenticationError("wrong password")
        return issue_token(username)

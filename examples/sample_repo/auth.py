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


def verify_token(token):
    """Check a token's signature and expiry; return the username if valid."""
    username, expiry, signature = token.rsplit(":", 2)
    payload = f"{username}:{expiry}"
    expected = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise AuthenticationError("bad token signature")
    if int(expiry) < time.time():
        raise AuthenticationError("token expired")
    return username


class AuthService:
    """Authenticates users against the user repository."""

    def __init__(self, repository):
        self.repository = repository

    def validate_user(self, username, password):
        """Authenticate a user by username and password and return a session token."""
        user = self.repository.find_user(username)
        if user is None:
            raise AuthenticationError("unknown user")
        if not hmac.compare_digest(hash_password(password, user["salt"]), user["password_hash"]):
            raise AuthenticationError("wrong password")
        return issue_token(username)

    def current_user(self, token):
        """Resolve the user behind a session token."""
        username = verify_token(token)
        return self.repository.find_user(username)

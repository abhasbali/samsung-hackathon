"""Application exceptions."""


class AppError(Exception):
    """Base class for application errors."""


class AuthenticationError(AppError):
    """Raised when credentials or tokens are invalid."""


class DatabaseError(AppError):
    """Raised when the database cannot be reached after retries."""

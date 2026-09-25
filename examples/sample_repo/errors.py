"""Application exceptions."""


class AppError(Exception):
    """Base class for application errors."""


class ValidationError(AppError):
    """Raised when user input fails validation."""


class AuthenticationError(AppError):
    """Raised when credentials or tokens are invalid."""


class DatabaseError(AppError):
    """Raised when the database cannot be reached after retries."""

"""Service entry point: authenticate, preprocess, predict and store."""

from auth import AuthService
from config import load_config
from database import UserRepository, connect
from model import predict
from preprocessing import preprocess_input


def register_user(repository, username, salt, password_hash):
    """Create a new user record and save it."""
    user = {"username": username, "salt": salt, "password_hash": password_hash}
    repository.save_user(user)
    return user


def handle_request(service, token, payload):
    """Authenticate the caller, preprocess the payload and return a prediction."""
    service.current_user(token)
    record = preprocess_input(payload)
    return {"score": predict(record)}


def main():
    settings = load_config()
    repository = UserRepository(connect(settings["database_url"]))
    service = AuthService(repository)
    token = service.validate_user("alice", "wonderland")
    print(handle_request(service, token, {"Age": " 42 ", "income": 50000}))


if __name__ == "__main__":
    main()

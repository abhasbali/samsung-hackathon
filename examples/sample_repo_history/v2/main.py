"""Service entry point: authenticate, preprocess and predict."""

from auth import AuthService
from config import load_config
from database import UserRepository, connect
from model import predict
from preprocessing import normalize_input, validate_input


def main():
    settings = load_config()
    repository = UserRepository(connect(settings["database_url"]))
    service = AuthService(repository)
    service.authenticate("alice", "wonderland")
    record = validate_input(normalize_input({"Age": " 42 ", "income": 50000}))
    print(predict(record))


if __name__ == "__main__":
    main()

"""Service entry point."""

from auth import authenticate
from config import load_config
from database import connect, save_user
from model import predict
from text_utils import normalize_input


def main():
    settings = load_config()
    conn = connect(settings["database_url"])
    users = {"alice": {"username": "alice", "password": "wonderland"}}
    save_user(conn, users["alice"])
    if authenticate(users, "alice", "wonderland"):
        print(predict(normalize_input({"Age": " 42 ", "income": 50000})))


if __name__ == "__main__":
    main()

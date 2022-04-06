import argparse
import requests

from src.match import match
from typing import Tuple


def get_backend_and_model() -> Tuple[str, str]:
    url = "http://localhost:8000/"
    headers = {"Accept": "application/json"}
    res = requests.get(url, headers=headers).json()
    return res['backend'], res['model']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    print(get_backend_and_model())
    # match(args.headless)

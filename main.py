import os
import argparse
import requests

from src.match import match
from src.data import get_validation_zip
from typing import Tuple


def get_backend_and_model() -> Tuple[str, str]:
    url = os.environ.get("CHECKLIST_API_URL")
    if not url:
        url = "http://localhost:8000/"
    headers = {"Accept": "application/json"}
    res = requests.get(url, headers=headers).json()
    return res['backend'], res['model']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    detector_backend, model_name = get_backend_and_model()
    print(detector_backend, model_name)
    for face, face_name in get_validation_zip():
        print('running:', face_name)
        match(
            args.headless,
            detector_backend=detector_backend,
            face=face,
            face_name=face_name,
            model_name=model_name,
        )

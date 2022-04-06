import json
from typing import Tuple

from flask import Flask
from flask.json import jsonify

app = Flask(__name__)


def get_first_unused(checklist: dict):
    for backend in checklist:
        for model, used in checklist[backend].items():
            if not used:
                return backend, model
    return None, None


def get_and_update() -> Tuple[str, str]:
    with open("./checklist.json", "r+") as f:
        checklist = json.loads(f.read())
    backend, model = get_first_unused(checklist)
    if not backend and not model:
        return None, None
    checklist[backend][model] = True
    with open("./checklist.json", "w+") as f:
        f.write(json.dumps(checklist, indent=2))
    return backend, model


@app.route("/")
def get_backend_and_model_to_test():
    backend, model = get_and_update()
    return jsonify({"backend": backend, "model": model})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)

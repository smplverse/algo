import numpy as np
import cv2
import os
import json
import hashlib

from typing import List


def write_file(obj: dict, fname=None) -> str:
    if fname is None:
        fname = hashlib.sha256().hexdigest()[:8]
    path = f"log/{fname}.json"
    with open(path, "w+") as f:
        f.write(json.dumps(obj, indent=2))
    print(f"wrote {path}")
    return fname


def is_valid(path: str):
    if "png" not in path:
        return False
    if "copy" not in path and "seg" not in path:
        return True
    return False


def load_smpls(base_path: str) -> List[np.ndarray]:
    paths = os.listdir(base_path)
    if len(paths) == 0:
        raise Exception("no smpls found")
    paths = [path for path in paths if is_valid(path)][:100]
    paths = [base_path + "/" + path for path in paths]
    smpls = [cv2.imread(path) for path in paths]
    return paths, smpls

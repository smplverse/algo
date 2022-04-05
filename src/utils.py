import numpy as np
import cv2
import os
import json
import hashlib

from typing import Any, List


def write_file(obj: Any, fname: str = None, sort: bool = False) -> str:
    if fname is None:
        fname = hashlib.sha256().hexdigest()[:8]
    path = f"results/{fname}.json"
    if sort == True and type(obj) is list:
        # TODO
        # obj = sorted(obj, key=lambda x: x.values(['distance'])
        pass
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

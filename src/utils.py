import numpy as np
import pickle
import cv2
import json
import hashlib

from typing import Any


def write_file(
    obj: Any,
    path: str = None,
    sort: bool = False,
) -> str:
    if path is None:
        fname = hashlib.sha256().hexdigest()[:8]
        path = f"results/{fname}.json"
    if sort == True and type(obj) is list:
        # TODO
        # obj = sorted(obj, key=lambda x: x.values(['distance'])
        pass
    with open(path, "w+") as f:
        f.write(json.dumps(obj, indent=2))
    print("wrote json: %s" % path)
    return path


def is_valid_png(path: str):
    if "png" not in path:
        return False
    if "copy" not in path and "seg" not in path:
        return True
    return False


def how_many_total_computations():
    total_computations = 0
    total_smpls = 7667
    while (total_smpls != 0):
        total_computations += total_smpls
        total_smpls -= 1
    print(f"total computations: {total_computations:,}")


def merge(face: np.ndarray, smpl: np.ndarray) -> np.ndarray:
    if face.shape != smpl.shape:
        h, w, _ = smpl.shape
        face = cv2.resize(face, dsize=(w, h))
    merged = np.concatenate((face, smpl), axis=1)
    return merged


def serialize(obj: Any, fpath: str):
    with open(fpath, "wb") as f:
        pickle.dump(obj, f)


def deserialize(fpath: str) -> Any:
    with open(fpath, "rb") as f:
        obj = pickle.load(f)
    return obj

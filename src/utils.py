import pickle
import numpy as np
import cv2

from typing import Any


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

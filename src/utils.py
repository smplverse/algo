import numpy as np
import cv2
import os
import json
import hashlib

from typing import Any, List, Tuple


def write_file(obj: Any, path: str = None, sort: bool = False) -> str:
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
    paths = [path for path in paths if is_valid(path)]
    paths = [base_path + "/" + path for path in paths]
    smpls = [cv2.imread(path) for path in paths]
    return paths, smpls


def get_face() -> Tuple[np.ndarray, str]:
    all_faces = os.listdir("data/input")
    rand = np.random.randint(len(all_faces))
    face_name = all_faces[rand]
    path = "data/input/" + face_name
    face = cv2.imread(path)
    face_name = face_name.replace(".jpg", "")
    return face, face_name


def how_many_total_computations():
    total_computations = 0
    total_smpls = 7667
    while (total_smpls != 0):
        total_computations += total_smpls
        total_smpls -= 1
    print(f"{total_computations:,}")


def merge(face: np.ndarray, smpl: np.ndarray) -> np.ndarray:
    if face.shape != smpl.shape:
        h, w, _ = smpl.shape
        face = cv2.resize(face, dsize=(w, h))
    merged = np.concatenate((face, smpl), axis=1)
    return merged

import os
import cv2
import numpy as np
import os

from random import shuffle
from typing import List, Tuple
from src.utils import is_valid_png


def get_validation_zip():
    with open("validation_sample.txt", "r+") as f:
        validation_faces = f.read().split("\n")
    shuffle(validation_faces)
    validation_sample = ["data/input/" + face for face in validation_faces]
    validation_sample = [cv2.imread(path) for path in validation_sample]
    assert len(validation_sample) == len(validation_faces) == 100
    return zip(validation_sample, validation_faces)


def get_smpls(base_path: str) -> List[np.ndarray]:
    paths = os.listdir(base_path)
    if len(paths) == 0:
        raise Exception("no smpls found")
    paths = [path for path in paths if is_valid_png(path)]
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

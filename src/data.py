import os
import os
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import is_valid_png


def get_famous_people():
    path = "data/famous_people/"
    fnames = os.listdir(path)
    famous_people = [cv2.imread(path + f) for f in fnames]
    assert len(famous_people)
    assert isinstance(famous_people[0], np.ndarray)
    return fnames, famous_people


def get_famous_people_zip():
    names, faces = get_famous_people()
    famous_people_zip = zip(names, faces)
    return famous_people_zip


def get_ibug_faces():
    path = "data/ibug_faces/"
    fnames = os.listdir(path)
    ibug_faces = [cv2.imread(path + f) for f in fnames]
    assert len(ibug_faces)
    assert isinstance(ibug_faces[0], np.ndarray)
    return fnames, ibug_faces


def get_smpls(base_path: str) -> List[np.ndarray]:
    paths = os.listdir(base_path)
    if len(paths) == 0:
        raise Exception("no smpls found")
    paths = [path for path in paths if is_valid_png(path)]
    paths = [base_path + "/" + path for path in paths]
    print("loading smpls..")
    smpls = []
    for path in tqdm(paths):
        smpl = cv2.imread(path)
        smpls.append(smpl)
    return paths, smpls


def get_face() -> Tuple[np.ndarray, str]:
    all_faces = os.listdir("data/input")
    rand = np.random.randint(len(all_faces))
    face_name = all_faces[rand]
    path = "data/input/" + face_name
    face = cv2.imread(path)
    face_name = face_name.replace(".jpg", "")
    return face, face_name


def get_ibug_zip():
    names, faces = get_ibug_faces()
    ibug_zip = zip(names, faces)
    return ibug_zip

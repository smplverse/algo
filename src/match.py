import os
import numpy as np
import cv2

from src.detect import crop_face
from typing import List, Union

# workflow
# detect face in webcam capture
# detect face in data frame
# match and collect scores

# I have a few methods in mind, it might be better to use cropped faces but might not
# it also depends on dimensions, I think its best to run some examples

# later on this shall be all moved to a class interface to be used along with a server


def load_smpls() -> List[np.ndarray]:
    # in the future will use a dataloader here, 7k pieces is a lot
    paths = os.listdir("data/smpls/")
    if len(paths) == 0:
        raise Exception("no smpls found")
    paths = [p for p in paths if "png" in p][:10]
    smpls = [cv2.imread("data/smpls/" + p) for p in paths]
    return smpls


def match(img1: np.ndarray, img2: np.ndarray):
    # TODO
    return 0


def match_smpl_to_face(frame: np.ndarray) -> Union[np.ndarray, None]:
    smpls = load_smpls()
    face = crop_face(frame)
    scores = []
    for smpl in smpls:
        score = match(face, smpl)
        scores.push(score)
    return smpls[np.argmax(scores)]

import os
import numpy as np
import cv2

# workflow
# detect face in webcam capture
# detect face in data frame
# match and collect scores


def match_face(face: np.ndarray):
    paths = os.listdir("data/smpls/")
    paths = [p for p in paths if "png" in p][:10]
    smpls = [cv2.imread("data/smpls/" + p) for p in paths]
    print(face.shape)
    return len(smpls)

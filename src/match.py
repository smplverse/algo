import cv2
import os
import numpy as np

from deepface import DeepFace
from typing import List, Tuple

# workflow
# detect face in webcam capture
# detect face in data frame
# match and collect scores

# I have a few methods in mind, it might be better to use cropped faces but might not
# it also depends on dimensions, I think its best to run some examples

# later on this shall be all moved to a class interface to be used along with a server

# 100 mock input images
# 1000 smpls
# tensorboard
# store hash of the image uploaded in the address mapping for given indices


def is_valid(path: str):
    if "png" not in path:
        return False
    if "copy" not in path and "seg" not in path:
        return True
    return False


def load_smpls(base_path: str) -> List[np.ndarray]:
    # in the future will use a dataloader here, 7k pieces is a lot
    # there is also an option to use a database
    paths = os.listdir(base_path)
    if len(paths) == 0:
        raise Exception("no smpls found")
    paths = [path for path in paths if is_valid(path)][:100]
    paths = [base_path + "/" + path for path in paths]
    smpls = [cv2.imread(path) for path in paths]
    return smpls


def match_smpl_to_face(face: np.ndarray) -> Tuple[np.ndarray, dict]:
    # also loop could break if there is a good match, good enough to verify
    smpls = load_smpls("data/smpls")
    scores = []
    results = []
    for smpl in smpls:
        try:
            result = DeepFace.verify(img1_path=face, img2_path=smpl)
            print(result)
            results.append(result)
            scores.append(result['distance'])
        except:
            scores.append(1)
            results.append(None)
    print(f'successful matches: {np.count_nonzero(results)}/{len(results)}')
    best_score_idx = np.argmin(scores)  # best score is lowest distance
    return smpls[best_score_idx], results[best_score_idx]

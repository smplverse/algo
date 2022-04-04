import cv2
import os
import numpy as np

from deepface import DeepFace
from typing import List, Tuple


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
    return smpls


def match_smpl_to_face(face: np.ndarray) -> Tuple[np.ndarray, dict]:
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
    best_score_idx = np.argmin(scores)
    return smpls[best_score_idx], results[best_score_idx]

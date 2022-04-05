import numpy as np
from deepface import DeepFace
from src.utils import write_file, load_smpls
from typing import Tuple


def match_smpl_to_face(face: np.ndarray) -> Tuple[np.ndarray]:
    paths, smpls = load_smpls("data/smpls")
    scores = []
    # build ensemble before feeding the images
    res = {}
    for path, smpl in zip(paths, smpls):
        try:
            result = DeepFace.verify(img1_path=face,
                                     img2_path=smpl,
                                     model_name="Ensemble")
            res[path] = result
            scores.append(result['distance'])
        except:
            scores.append(1)
    best_score_idx = np.argmin(scores)
    write_file(res)
    return smpls[best_score_idx]

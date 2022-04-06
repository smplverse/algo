import cv2
import numpy as np
import time

from typing import Any, List
from deepface import DeepFace
from src.utils import write_file, merge
from src.data import get_smpls
from src.visualization import show_comparison_cv


def log_stats(scores: List[float]) -> int:
    best_score_idx = np.argmin(scores)
    best_match = scores[best_score_idx]
    landed = np.count_nonzero(np.array(scores) != 1)
    print("detection rate: %d/%d" % (landed, len(scores)))
    print("best match: %.2f" % best_match)
    return best_score_idx


def match(
    headless: bool,
    face: np.ndarray,
    face_name: str,
    model: Any,
    model_name: Any,
    detector_backend: str,
):
    paths, smpls = get_smpls("data/smpls")
    scores = []
    res = []
    inference_times = []
    tic = time.time()
    for idx, [path, smpl] in enumerate(zip(paths, smpls)):
        try:
            inference_tic = time.time()
            result = DeepFace.verify(
                img1_path=face,
                img2_path=smpl,
                model=model,
                detector_backend=detector_backend,
            )
            inference_toc = time.time()
            inference_times.append(float(inference_toc - inference_tic))
            res.append({path: result})
            show_comparison_cv(face, smpl)
            distance = result['distance']
            scores.append(distance)
            if idx % 10 == 1:
                log_stats(scores)
        except:
            scores.append(1)
    toc = time.time()

    best_score_idx = log_stats(scores)
    print("total time elapsed: %.2fs" % float(toc - tic))
    print("average time per image: %.2fs" % np.mean(inference_times))
    smpl = smpls[best_score_idx]
    merged = merge(face, smpl)
    base_path = f"results/{detector_backend}/{model_name}"
    fpath = f"{base_path}/json/{face_name.replace('.jpg', '')}.json"
    write_file(res, path=fpath)
    cv2.imwrite(f"{base_path}/image/{face_name}.png", merged)
    print("saved img %s" % face_name)
    if not headless:
        show_comparison_cv(face, smpl, final=True)

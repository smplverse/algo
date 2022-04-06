import cv2
import numpy as np
import time

from deepface import DeepFace
from src.utils import write_file, load_smpls, get_face
from src.visualization import show_comparison_cv


def match(
    headless: bool,
    write=True,
    face: np.ndarray = None,
    face_name: str = None,
    model_name="VGG-Face",
    detector_backend="opencv",
):
    tic = time.time()
    if face is None or face_name is None:
        face, face_name = get_face()
    paths, smpls = load_smpls("data/smpls")
    scores = []
    res = []
    inference_times = []
    model = DeepFace.build_model(model_name)
    print("built", model_name)
    for path, smpl in zip(paths, smpls):
        try:
            inference_tic = time.time()
            result = DeepFace.verify(
                img1_path=face,
                img2_path=smpl,
                model=model,
                detector_backend=detector_backend,
            )
            inference_toc = time.time()
            inference_times.append(float(inference_tic - inference_toc))
            res.append({path: result})
            show_comparison_cv(face, smpl, headless)
            distance = result['distance']
            scores.append(distance)
            if (result['verified']):
                print("verified smpl %s with %.2f distance" % (path, distance))
        except:
            scores.append(1)
    best_score_idx = np.argmin(scores)
    best_match = scores[best_score_idx]
    toc = time.time()
    print("detection rate: %d/%d" % (np.bincount(scores), len(scores)))
    print("best match: %.2f" % best_match)
    print("total time elapsed: %.2fs" % float(toc - tic))
    print("average time per image: %.2fs" % np.mean(inference_times))
    smpl = smpls[best_score_idx]
    if write:
        merged = np.concatenate((face, smpl), axis=1)
        base_path = f"results/{detector_backend}/{model_name}"
        write_file(res, path=f"{base_path}/json/{face_name}.png")
        cv2.imwrite(f"{base_path}/image/{face_name}.png", merged)
        print("saved img %s" % face_name)
    if best_match < 0.4:
        show_comparison_cv(face, smpl, headless, final=True)

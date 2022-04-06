import cv2
import numpy as np
import argparse
import time

from deepface import DeepFace
from src.utils import write_file, load_smpls, get_face
from src.visualization import show_comparison_cv


def main(headless: bool, write=True):
    tic = time.time()
    face, face_name = get_face()
    face_name = face_name.replace(".jpg", "")
    paths, smpls = load_smpls("data/smpls")
    scores = []
    res = []
    model_name = "VGG_Face"
    detector_backend = "opencv"
    for path, smpl in zip(paths, smpls):
        try:
            result = DeepFace.verify(
                img1_path=face,
                img2_path=smpl,
                model_name=model_name,
                detector_backend=detector_backend,
            )
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
    print("best match: %.2f" % best_match)
    print("time elapsed: %.2fs" % float(toc - tic))
    smpl = smpls[best_score_idx]
    if write:
        merged = np.concatenate((face, smpl), axis=1)
        base_path = f"results/{detector_backend}/{model_name}"
        write_file(res, path=f"{base_path}/json/{face_name}.png")
        cv2.imwrite(f"{base_path}/image/{face_name}.png", merged)
        print("saved img %s" % face_name)
    if best_match < 0.4:
        show_comparison_cv(face, smpl, headless, final=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(args.headless)

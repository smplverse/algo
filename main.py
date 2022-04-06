import os
import cv2
import numpy as np

from deepface import DeepFace
from src.utils import write_file, load_smpls, get_face
from src.visualization import show_comparison_cv


def main():
    face, face_name = get_face()
    face_name = face_name.replace(".jpg", "")
    paths, smpls = load_smpls("data/smpls")
    scores = []
    res = []
    # build ensemble before feeding the images
    for path, smpl in zip(paths, smpls):
        show_comparison_cv(face, smpl)
        try:
            result = DeepFace.verify(img1_path=face, img2_path=smpl)
            res.append({path: result})
            scores.append(result['distance'])
        except:
            scores.append(1)
    best_score_idx = np.argmin(scores)
    print(best_score_idx, len(smpls))
    best_match = scores[best_score_idx]
    print("best match: %.2f" % best_match)
    smpl = smpls[best_score_idx]
    write_file(res, fname=face_name, sort=True)
    if best_match < 0.4:
        show_comparison_cv(face, smpl, final=True, write=True, fname=face_name)


if __name__ == "__main__":
    main()

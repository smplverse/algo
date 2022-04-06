import argparse
import numpy as np
import time

from deepface import DeepFace
from src.utils import write_file, load_smpls, get_face
from src.visualization import show_comparison_cv


def main(headless: bool):
    tic = time.time()
    face, face_name = get_face()
    face_name = face_name.replace(".jpg", "")
    paths, smpls = load_smpls("data/smpls")
    scores = []
    res = []
    # build ensemble before feeding the images
    for path, smpl in zip(paths, smpls):
        try:
            result = DeepFace.verify(img1_path=face, img2_path=smpl)
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
    write_file(res, fname=face_name, sort=True)
    if best_match < 0.4:
        show_comparison_cv(face,
                           smpl,
                           headless,
                           final=True,
                           write=True,
                           fname=face_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(args.headless)

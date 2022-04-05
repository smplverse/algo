import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from deepface import DeepFace
from src.utils import write_file, load_smpls


def get_face() -> np.ndarray:
    all_faces = os.listdir("data/input")
    rand = np.random.randint(len(all_faces))
    face_name = all_faces[rand]
    path = "data/input/" + face_name
    face = cv2.imread(path)
    return face, face_name


def show_comparison_cv(face: np.ndarray, smpl: np.ndarray, final=False):
    if face.shape != smpl.shape:
        h, w, _ = smpl.shape
        face = cv2.resize(face, dsize=(w, h))
    cv2.imshow("img", np.concatenate((face, smpl)))
    waitKeyTime = 1
    if final:
        waitKeyTime = 0
    if cv2.waitKey(waitKeyTime) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def show_comparison_mpl(face: np.ndarray, smpl: np.ndarray):
    _, axes = plt.subplots(2, 1)
    for img, ax in zip([face, smpl], axes):
        ax.imshow(img)
    plt.show()


def main():
    face, face_name = get_face()
    paths, smpls = load_smpls("data/smpls")
    scores = []
    res = {}
    # build ensemble before feeding the images
    for path, smpl in zip(paths, smpls):
        show_comparison_cv(face, smpl)
        try:
            result = DeepFace.verify(img1_path=face, img2_path=smpl)
            res[path] = result
            scores.append(result['distance'])
        except:
            scores.append(1)
    best_score_idx = np.argmin(scores)
    print(best_score_idx, len(smpls))
    print("best match: %.2f" % scores[best_score_idx])
    smpl = smpls[best_score_idx]
    write_file(res, fname=face_name.replace(".jpg", ""))
    show_comparison_cv(face, smpl, final=True)


if __name__ == "__main__":
    main()

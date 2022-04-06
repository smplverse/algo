import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_comparison_cv(face: np.ndarray,
                       smpl: np.ndarray,
                       headless: bool,
                       final: bool = False,
                       write: bool = False,
                       fname: str = None):
    if headless:
        return
    if face.shape != smpl.shape:
        h, w, _ = smpl.shape
        face = cv2.resize(face, dsize=(w, h))
    merged = np.concatenate((face, smpl), axis=1)
    cv2.imshow("img", merged)
    waitKeyTime = 1
    if final:
        waitKeyTime = 0
    if cv2.waitKey(waitKeyTime) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    if write and fname:
        cv2.imwrite(f"results/{fname}.png", merged)
        print("saved img %s" % fname)


def show_comparison_mpl(face: np.ndarray, smpl: np.ndarray):
    _, axes = plt.subplots(2, 1)
    for img, ax in zip([face, smpl], axes):
        ax.imshow(img)
    plt.show()

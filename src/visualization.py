import cv2
import numpy as np

from matplotlib import pyplot as plt
from src.utils import merge


def show_comparison_cv(
    face: np.ndarray,
    smpl: np.ndarray,
    headless: bool,
    final: bool = False,
):
    if headless:
        return
    merged = merge(face, smpl)
    cv2.imshow("img", merged)
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

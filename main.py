import cv2
import numpy as np

from matplotlib import pyplot as plt
from src.match import match_smpl_to_face


def show_result():
    # TODO implement an evaluation func
    pass


def main():
    face_path = "data/input/AJ_Cook/AJ_Cook_0001.jpg"
    face = cv2.imread(face_path)
    smpl, top_result = match_smpl_to_face(face)
    print("\n\nbest match:", top_result)
    _, axes = plt.subplots(2, 1)
    for img, ax in zip([face, smpl], axes):
        ax.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()

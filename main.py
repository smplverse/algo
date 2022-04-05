import cv2

from matplotlib import pyplot as plt
from src.match import match_smpl_to_face


def show_result():
    pass


def main():
    face_path = "data/input/AJ_Cook_0001.jpg"
    face = cv2.imread(face_path)
    smpl = match_smpl_to_face(face)
    _, axes = plt.subplots(2, 1)
    for img, ax in zip([face, smpl], axes):
        ax.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()

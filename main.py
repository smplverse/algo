import cv2

from src.detect import crop_face
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img_path = 'data/input/AJ_Cook/AJ_Cook_0001.jpg'
    img = cv2.imread(img_path)
    cropped_face = crop_face(img)
    plt.imshow(cropped_face)
    plt.show()

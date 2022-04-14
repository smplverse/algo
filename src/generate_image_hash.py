import argparse
import cv2
import hashlib
import numpy as np
import glob


def generate_image_hash(collection_size: int):
    ncols = int(np.sqrt(collection_size))
    images = [cv2.imread(i, -1) for i in glob.glob('data/smpls/*.png')]
    img = np.concatenate(images[:ncols], axis=1)
    images = images[ncols:]
    while len(images) > ncols:
        new_row = np.concatenate(images[:ncols], axis=1)
        row_height, *_ = new_row.shape
        img = np.concatenate((img, new_row), axis=0)
        images = images[ncols:]
    filler = np.zeros((row_height, (ncols - len(images)) * 100, 4))
    last_row = np.concatenate(images, axis=1)
    last_row = np.concatenate((last_row, filler), axis=1)
    img = np.concatenate((img, last_row), axis=0)
    print(img.shape)
    cv2.imwrite('collection_image.png', img)
    img = cv2.imread('collection_image.png', -1)
    img_hash = hashlib.sha256(img).hexdigest()
    print(f'img hash: {img_hash}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_size', type=int, required=True)
    args = parser.parse_args()

    generate_image_hash(args.collection_size)

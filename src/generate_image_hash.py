import argparse
import glob
from hashlib import sha256

import os
import cv2
import numpy as np
from tqdm import tqdm
from src.utils import deserialize, serialize


def generate_image_hash(collection_size: int, a: int):
    ncols = np.ceil(np.sqrt(collection_size)).astype(int)
    images = []
    print("loading images...")
    got_pickled = "smpls.p" in os.listdir("data")
    if got_pickled:
        print("loading pickled images...")
        images = deserialize("data/smpls.p")
    else:
        for filename in (pbar := tqdm(glob.glob("data/smpls/*.png"))):
            pbar.set_description(f"loading {filename}")
            image = cv2.imread(filename)
            image = cv2.resize(
                image,
                dsize=(a, a),
                interpolation=cv2.INTER_AREA,
            )
            images.append(image)
        serialize(images, "data/smpls.p")
    _, _, c = images[2].shape

    print("adding rows...")
    rows = []
    for _ in tqdm(range(len(images) // ncols)):
        # concatenate rows in the top till have enough images to fill up row
        new_row = np.concatenate(images[:ncols], axis=1)
        rows.append(new_row)

    # apply filler once there is no more images yet row has to be filled up
    filler = np.zeros((a, (ncols - len(images)) * a, c))
    last_row = np.concatenate(images, axis=1)
    last_row = np.concatenate((last_row, filler), axis=1)
    rows.append(last_row)

    # merge
    img = np.concatenate(rows, axis=1)

    # save results
    print(img.shape)
    cv2.imwrite('collection_image.png', img)
    img = cv2.imread('collection_image.png', -1)
    img_hash = sha256(img).hexdigest()
    print(f'img hash: {img_hash}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection-size', type=int, required=True)
    parser.add_argument('--a', type=int, default=256)
    args = parser.parse_args()

    generate_image_hash(args.collection_size, args.a)

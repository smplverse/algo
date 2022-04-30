import glob

import os
import cv2
import numpy as np
from tqdm import tqdm
from src.utils import deserialize, serialize


def generate_collection_image(collection_size: int, a: int, force=False):
    ncols = np.ceil(np.sqrt(collection_size)).astype(int)
    images = []
    print("loading images...")
    got_pickled = "smpls.p" in os.listdir("data")
    if got_pickled and not force:
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
        images = images[ncols:]
        rows.append(new_row)

    # apply filler once there is no more images yet row has to be filled up
    remaining_images_to_fill = ncols - len(images)
    fillers = [
        np.zeros((a, a, c), dtype=np.uint8)
        for _ in range(remaining_images_to_fill)
    ]
    images += fillers
    assert len(images) == ncols
    last_row = np.concatenate(
        images,
        axis=1,
    )
    rows.append(last_row)

    # merge
    img = np.vstack(rows)

    # save results
    print("written image of shape", img.shape)
    cv2.imwrite('collection_image.png', img)
    img = cv2.imread('collection_image.png', -1)

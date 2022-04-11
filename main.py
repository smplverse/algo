import argparse
import time

from src.data import get_validation_zip
from src.matcher import Matcher
from src.vgg_face2 import VGGFace2


def main(headless: bool):
    model = VGGFace2()
    for face, face_name in get_validation_zip():
        tic = time.time()
        print('\nrunning for:', face_name)
        matcher = Matcher(model=model, headless=headless)
        matcher.match(face)
        toc = time.time()
        print("total time elapsed: %.2fs" % float(toc - tic))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    main(args.headless)

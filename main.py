import argparse

from tqdm import tqdm

from src.data import get_validation_zip
from src.matcher import Matcher
from src.vgg_face2 import VGGFace2


def main(headless: bool):
    model = VGGFace2()
    validation_zip = get_validation_zip()
    for face, face_name in (pbar := tqdm(list(validation_zip))):
        pbar.set_description_str('running for: %s' % face_name)
        matcher = Matcher(model=model, headless=headless)
        matcher.match(face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    main(args.headless)

import argparse

from tqdm import tqdm

from src.data import get_validation_zip, get_ibug_faces
from src.matcher import Matcher
from src.onnx_model import OnnxModel


def main_famous_people(headless: bool):
    model = OnnxModel()
    validation_zip = get_validation_zip()
    for face, face_name in (pbar := tqdm(list(validation_zip))):
        pbar.set_description_str('running for: %s' % face_name)
        matcher = Matcher(model=model, headless=headless)
        matcher.match(face)


def main_ibug_faces(headless: bool):
    model = OnnxModel()
    names, faces = get_ibug_faces()
    ibug_zip = zip(names, faces)
    for name, face in (pbar := tqdm(list(ibug_zip))):
        pbar.set_description_str('running for: %s' % name)
        matcher = Matcher(model=model, headless=headless)
        matcher.match(face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    main_ibug_faces(args.headless)

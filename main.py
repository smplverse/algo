import argparse

from src.matcher import Matcher
from src.data import get_validation_zip
from deepface.DeepFace import build_model


def main(headless: bool):
    model = build_model("VGG-Face")
    for face, face_name in get_validation_zip():
        print('\nrunning for:', face_name)
        matcher = Matcher(
            headless,
            face=face,
            face_name=face_name,
            model=model,
        )
        matcher.loop_through_all_smpls()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    main(args.headless)

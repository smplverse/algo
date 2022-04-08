import argparse

from src.match import Matcher
from src.data import get_validation_zip
from deepface.DeepFace import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # detector_backend, model_name = get_backend_and_model()
    detector_backend = "opencv"
    # to improve this pipeline Im going to need to deep-dive into
    # the pipeline of verify method
    model_name = "DeepID"
    model = build_model(model_name)
    print("built %s to be used with %s" % (model_name, detector_backend))
    for face, face_name in get_validation_zip():
        print('\nrunning for:', face_name)
        matcher = Matcher(
            args.headless,
            detector_backend=detector_backend,
            face=face,
            face_name=face_name,
            model=model,
            model_name=model_name,
        )
        matcher.match()


if __name__ == "__main__":
    main()

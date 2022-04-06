import cv2


def get_validation_zip():
    with open("validation_sample.txt", "r+") as f:
        validation_faces = f.read().split("\n")
    validation_sample = ["data/input/" + face for face in validation_faces]
    validation_sample = [cv2.imread(path) for path in validation_sample]
    assert len(validation_sample) == len(validation_faces) == 100
    return zip(validation_sample, validation_faces)

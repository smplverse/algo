from deepface.commons.functions import detect_face
from src.data import get_smpls
from src.visualization import show_img_cv


def generate_output():
    pass


def benchmark():
    # before implementing our own, lets try the deepface builtins
    smpls = get_smpls()
    detector_backends = [
        'opencv',
        'ssd',
        'dlib',
        'mtcnn',
        'retinaface',
        'mediapipe',
    ]
    for smpl in smpls:
        region, _ = detect_face(smpl, detector_backends[0])
        show_img_cv(region, waitKey=1)


if __name__ == "__main__":
    benchmark()

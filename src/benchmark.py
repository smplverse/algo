from deepface.commons.functions import detect_face


def benchmark():
    # before implementing our own, lets try the deepface builtins
    detector_backends = ["", ""]

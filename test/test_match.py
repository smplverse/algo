import pytest

from deepface import DeepFace


@pytest.mark.skip()
def test_match():
    from src.match import match
    from src.utils import get_face
    face, face_name = get_face()
    detector_backend = "opencv"
    model_name = "VGG-Face"
    model = DeepFace.build_model(model_name)
    headless = True
    match(
        headless,
        face,
        face_name,
        model,
        model_name,
        detector_backend,
    )


def test_deepface_performs_single_match():
    model = DeepFace.build_model('DeepID')
    res = DeepFace.verify(
        img1_path="data/input/AJ_Cook_0001.jpg",
        img2_path="data/input/AJ_Cook_0001.jpg",
        model=model,
    )
    assert round(res['distance'], 5) == 0

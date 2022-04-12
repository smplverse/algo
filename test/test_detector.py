import pytest
import cv2

from src.detector import Detector


@pytest.fixture()
def detector():
    detector = Detector()
    yield detector


def test_detector_detects(detector):
    img = cv2.imread("data/famous_people/AJ_Cook_0001.jpg")
    face = detector.detect_face(img)
    assert face is not None

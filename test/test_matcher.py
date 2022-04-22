import cv2
import pytest
import numpy as np

from src.matcher import Matcher


@pytest.fixture()
def matcher():
    matcher = Matcher(headless=True, model="resnet100")
    yield matcher
    del matcher.model.session


@pytest.fixture()
def smpl():
    return cv2.imread("data/smpls/000056.png")


@pytest.fixture()
def face():
    return cv2.imread("data/famous_people/AJ_Cook_0001.jpg")


def test_matches_images(
    matcher: Matcher,
    face: np.ndarray,
    smpl: np.ndarray,
):
    result = matcher.match_images(face, smpl)
    assert isinstance(result, float)
    assert 0 < result < 1

import cv2
import pytest
import numpy as np

from src.matcher import Matcher


@pytest.fixture()
def matcher():
    matcher = Matcher()
    yield matcher
    del matcher.model.session


@pytest.fixture()
def smpl():
    return cv2.imread("data/smpls/000056.png")


@pytest.fixture()
def face():
    return cv2.imread("data/input/AJ_Cook_0001.jpg")


def test_matching(
    matcher: Matcher,
    face: np.ndarray,
    smpl: np.ndarray,
):
    result = matcher.match(face, smpl)
    print(result)
    assert 0

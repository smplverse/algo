import cv2
import pytest
import numpy as np

from src.vgg_face2 import VGGFace2
from deepface import DeepFace


@pytest.fixture()
def img():
    yield cv2.imread("data/input/AJ_Cook_0001.jpg")


@pytest.fixture()
def vgg():
    vgg = VGGFace2()
    yield vgg
    del vgg.session


def test_preprocess_works(vgg: VGGFace2, img: np.ndarray):
    inp = vgg.preprocess(img)
    assert inp.shape == (1, 3, 224, 224)
    assert inp.dtype == np.float32
    assert inp.max() < 1.


def test_call_works(vgg: VGGFace2, img: np.ndarray):
    out = vgg(img)
    assert out.shape == (1, 512)

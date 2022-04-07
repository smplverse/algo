import cv2
import onnx
import pytest

from src.vgg_face2 import VGGFace2


@pytest.fixture()
def small_face():
    pass


@pytest.fixture()
def sample_input():
    pass


@pytest.fixture()
def vgg():
    vgg = VGGFace2()
    yield vgg
    del vgg.session


def test_preprocessing(vgg):
    pass


def test_padding(vgg):
    pass


def test_if_works_as_deepface_vgg(vgg):
    pass

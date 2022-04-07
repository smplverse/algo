import cv2
import onnx
import pytest

from src.vgg_face2 import VGGFace2


@pytest.fixture()
def small_face():
    pass


@pytest.fixture()
def vgg():
    vgg = VGGFace2()
    yield vgg
    del vgg.model


def test_model_ok(vgg):
    onnx.checker.check_model(vgg.model)


def test_padding(vgg):
    pass

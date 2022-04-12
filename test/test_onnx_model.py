import cv2
import pytest
import numpy as np

from src.onnx_model import OnnxModel


@pytest.fixture()
def img():
    yield cv2.imread("data/ibug_faces/indoor_001.png")


@pytest.fixture()
def model():
    model = OnnxModel()
    yield model
    del model.session


def test_preprocess_works(model: OnnxModel, img: np.ndarray):
    inp = model.preprocess(img)
    assert inp.shape == (1, 3, 224, 224)
    assert inp.dtype == np.float32
    assert inp.max() <= 1.


def test_call_works(model: OnnxModel, img: np.ndarray):
    out = model(img)
    assert out.shape == (512, )

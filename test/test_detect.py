import pytest
import cv2
import numpy as np

from src.detect import crop_face, detect_face


@pytest.fixture()
def img():
    img_path = 'data/input/AJ_Cook/AJ_Cook_0001.jpg'
    img = cv2.imread(img_path)
    yield img


@pytest.fixture()
def no_img():
    no_img = np.zeros(shape=(250, 250, 3))
    yield no_img


def test_detect(img):
    face = detect_face(img)
    assert 'bbox' in face


def test_detect_none(no_img):
    no_face = detect_face(no_img)
    assert no_face == None


def test_crop(img):
    cropped_face = crop_face(img)
    assert len(cropped_face.shape) == 3
    assert isinstance(cropped_face, np.ndarray)


def test_crop_none(no_img):
    no_cropped_face = crop_face(no_img)
    assert no_cropped_face == None

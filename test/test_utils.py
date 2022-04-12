import numpy as np


def test_is_valid_png():
    from src.utils import is_valid_png
    assert is_valid_png("asdfasdf") == False
    assert is_valid_png("img.png") == True
    assert is_valid_png("img_copy.png") == False

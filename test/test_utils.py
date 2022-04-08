import os
import pytest
import numpy as np


@pytest.fixture()
def mock_dir():
    mock_dir = "data/no_smpls"
    os.mkdir(mock_dir)
    yield mock_dir
    os.removedirs(mock_dir)


def test_get_smpls():
    from src.data import get_smpls
    _, smpls = get_smpls("data/smpls")
    assert len(smpls) >= 10
    assert isinstance(smpls[3], np.ndarray)
    assert np.any(smpls[3])
    assert len(smpls[3].shape) == 3


def test_load_smpls_throws_if_no_smpls(mock_dir):
    from src.data import get_smpls
    with pytest.raises(Exception) as e:
        get_smpls(mock_dir)
    assert str(e.value) == "no smpls found"


def test_is_valid_png():
    from src.utils import is_valid_png
    assert is_valid_png("asdfasdf") == False
    assert is_valid_png("img.png") == True
    assert is_valid_png("img_copy.png") == False


def test_write_file():
    from src.utils import write_file
    obj = {"key": "value"}
    path = write_file(obj)
    assert path.replace("results/", "") in os.listdir('results')
    os.remove(path)

import pytest
import numpy as np
import os

from src.match import load_smpls


@pytest.fixture()
def mock_dir():
    mock_dir = "data/no_smpls"
    os.mkdir(mock_dir)
    yield mock_dir
    os.removedirs(mock_dir)


@pytest.fixture()
def frame():
    pass


def test_load_smpls():
    smpls = load_smpls("data/smpls")
    assert len(smpls) >= 10
    assert isinstance(smpls[3], np.ndarray)


def test_load_smpls_throws_if_no_smpls(mock_dir):
    with pytest.raises(Exception) as e:
        load_smpls(mock_dir)
    assert str(e.value) == "no smpls found"

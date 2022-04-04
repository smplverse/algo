import pytest

from src.match import load_smpls


@pytest.fixture()
def frame():
    pass


def test_load_smpls():
    smpls = load_smpls()
    assert len(smpls) >= 10
    assert isinstance(smpls[3], str)

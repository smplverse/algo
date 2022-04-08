import pytest

from src.matcher import Matcher


@pytest.fixture()
def matcher():
    matcher = Matcher()
    yield matcher
    del matcher.model.session

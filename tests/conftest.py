import os
from pathlib import Path
import pytest


@pytest.fixture
def example_repo_path():
    if 'HEXRD_EXAMPLE_REPO_PATH' not in os.environ:
        pytest.fail('Environment varable HEXRD_EXAMPLE_REPO_PATH not set!')

    repo_path = os.environ['HEXRD_EXAMPLE_REPO_PATH']
    return Path(repo_path)


@pytest.fixture
def test_dir():
    return Path(__file__).resolve().parent


@pytest.fixture
def test_data_dir(test_dir):
    return test_dir / 'data'

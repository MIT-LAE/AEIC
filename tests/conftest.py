import os
from pathlib import Path

import pytest

os.environ['AEIC_PATH'] = str(
    (Path(__file__).parent.parent / 'data').resolve()
)


@pytest.fixture(scope='session')
def test_data_dir():
    return Path(__file__).parent / 'data'

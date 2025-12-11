import os
from pathlib import Path

import pytest

os.environ['AEIC_DATA_DIR'] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data')
)

print(os.environ['AEIC_DATA_DIR'])


@pytest.fixture(scope='session')
def test_data_dir():
    return Path(__file__).parent / 'data'

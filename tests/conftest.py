import os
import tomllib
from pathlib import Path

import pytest

from AEIC.config import Config

# Path to main data directory from here.
DATA_DIR = (Path(__file__).parent.parent / 'data').resolve()

# Set the path to include just the data directory.
os.environ['AEIC_PATH'] = str(DATA_DIR)
DEFAULT_CONFIG = DATA_DIR / 'IO' / 'default_config.toml'

# Absolute path to test data directory.
TEST_DATA_DIR = (Path(__file__).parent / 'data').resolve()

# Read default configuration and modify weather data directory for tests.
with open(DEFAULT_CONFIG, 'rb') as fp:
    CONFIG_DATA = tomllib.load(fp)
    CONFIG_DATA['weather']['weather_data_dir'] = str(TEST_DATA_DIR / 'weather')


@pytest.fixture(scope='session')
def test_data_dir():
    return TEST_DATA_DIR


# Set up and tear down global configuration around each test.
@pytest.fixture(autouse=True)
def reset_config():
    cfg = Config.model_validate(CONFIG_DATA)
    yield cfg
    Config.reset()

import os
import tomllib
from copy import deepcopy
from pathlib import Path

import pytest

from AEIC.config import Config

# Path to main data directory.
DATA_DIR = (Path(__file__).parent.parent / 'data').resolve()

# Absolute path to test data directory.
TEST_DATA_DIR = (Path(__file__).parent / 'data').resolve()

# Set the path to include the test and main data directories.
os.environ['AEIC_PATH'] = f'{TEST_DATA_DIR}:{DATA_DIR}'

# Path to default configuration file: this is in the `src` directory to ensure
# that it ends up in the built wheel.
DEFAULT_CONFIG = (Path(__file__).parent.parent) / 'src' / 'AEIC' / 'default_config.toml'


# Read default configuration and modify weather data directory for tests.
with open(DEFAULT_CONFIG, 'rb') as fp:
    DEFAULT_CONFIG_DATA = tomllib.load(fp)
    DEFAULT_CONFIG_DATA['weather']['weather_data_dir'] = str(TEST_DATA_DIR / 'weather')


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        'markers',
        'config_updates(**kwargs): '
        'Mark test to update default config with given key-value pairs.',
    )


@pytest.fixture
def test_data_dir():
    return TEST_DATA_DIR


@pytest.fixture
def default_config_data():
    return deepcopy(DEFAULT_CONFIG_DATA)


# Set up and tear down global configuration around each test.
@pytest.fixture(autouse=True)
def default_config(request, default_config_data):
    config_data = default_config_data
    data_marker = request.node.get_closest_marker('config_updates')
    if data_marker is not None:
        # Update config data with marker values.
        for key, value in data_marker.kwargs.items():
            if '__' not in key:
                config_data[key] = value
            else:
                section, param = key.split('__', 1)
                if section in config_data:
                    config_data[section][param] = value
                else:
                    config_data[section] = {param: value}
    cfg = Config.model_validate(config_data)
    yield cfg
    Config.reset()

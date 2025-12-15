import os
from pathlib import Path

import pytest

from AEIC.config import Config

# Absolute path to test data directory.
TEST_DATA_DIR = (Path(__file__).parent / 'data').resolve()

# Set the path to include the test data directory.
os.environ['AEIC_PATH'] = str(TEST_DATA_DIR)


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


# Set up and tear down global configuration around each test.
@pytest.fixture(autouse=True)
def default_config(request):
    # Updates to the default configuration are pulled from the config_updates
    # marker.
    data_marker = request.node.get_closest_marker('config_updates')

    # By default, load the default configuration with no updates.
    config_data = {}

    if data_marker is not None:
        # Build configuration data updates from values in marker.
        for key, value in data_marker.kwargs.items():
            if '__' not in key:
                config_data[key] = value
            else:
                section, param = key.split('__', 1)
                if section in config_data:
                    config_data[section][param] = value
                else:
                    config_data[section] = {param: value}

    # Load the default configuration with updates applied.
    Config.load(**config_data)

    # Test goes here...
    yield

    # Clear the configuration after the test.
    Config.reset()

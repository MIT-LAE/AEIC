import os
import tomllib
from pathlib import Path

import pytest

from AEIC.config import Config, config
from AEIC.missions import Mission
from AEIC.performance.models import PerformanceModel

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

    # Load the default configuration with updates applied, including an extra
    # override to force loading of data files from the tests/data directory if
    # such a file exists.
    Config.load(**config_data, data_path_overrides=[TEST_DATA_DIR])

    # Test goes here...
    yield

    # Clear the configuration after the test.
    Config.reset()


@pytest.fixture
def sample_missions():
    missions_file = config.file_location('missions/sample_missions_10.toml')
    with open(missions_file, 'rb') as f:
        mission_dict = tomllib.load(f)
    return Mission.from_toml(mission_dict)


@pytest.fixture
def performance_model():
    return PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )

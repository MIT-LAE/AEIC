import copy
import os
import tomllib
from pathlib import Path

import pytest

import AEIC
from AEIC.config import Config, config
from AEIC.missions import Mission
from AEIC.parsers.piano_reader import PianoData, PianoOverrides
from AEIC.parsers.ptf_reader import PTFData
from AEIC.performance.edb import EDBEntry
from AEIC.performance.model_selector import SimplePerformanceModelSelector
from AEIC.performance.models import PerformanceModel
from AEIC.performance.models.base import LTOPerformanceInput
from AEIC.types import Fuel

# Absolute path to test data directory.
TEST_DATA_DIR = (Path(__file__).parent / 'data').resolve()

# Source data the performance model builders are tested against. These paths
# are resolved directly rather than through `config.file_location`, so that the
# fixtures reading them can be session-scoped: the autouse `default_config`
# fixture below is function-scoped, and a session-scoped fixture cannot depend
# on it.
EDB_FILE = Path(AEIC.__file__).parent / 'data' / 'engines' / 'sample_edb.xlsx'
EDB_UID = '01P11CM121'
EDB_THRUST_FRACTIONS = (0.07, 0.30, 0.85, 1.0)
PIANO_DIR = TEST_DATA_DIR / 'performance' / 'piano'
PIANO_CLIMB_MASSES_KG = [54000.0, 50000.0, 46000.0, 42000.0]
PTF_FILE = TEST_DATA_DIR / 'verification' / 'legacy' / 'legacy_performance.PTF'

# Set the path to include the test data directory. This is done at module
# import time deliberately so the value is inherited by subprocesses spawned
# via `tests/subproc.py::run_in_subprocess`, which need the same AEIC_PATH
# when they re-import this conftest. Mutating it here on import (vs. inside
# a fixture) keeps the parent and child processes in agreement.
os.environ['AEIC_PATH'] = str(TEST_DATA_DIR)


def pytest_addoption(parser):
    """Register CLI flags."""
    parser.addoption(
        '--run-slow',
        action='store_true',
        default=False,
        help='also run tests marked @pytest.mark.slow (skipped by default)',
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        'markers',
        'config_updates(**kwargs): '
        'Mark test to update default config with given key-value pairs.',
    )
    config.addinivalue_line(
        'markers',
        'slow: long-running test, skipped unless --run-slow is passed.',
    )


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.slow tests unless --run-slow was passed."""
    if config.getoption('--run-slow'):
        return
    skip_slow = pytest.mark.skip(reason='slow test; pass --run-slow to enable')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def test_data_dir():
    return TEST_DATA_DIR


# Set up and tear down global configuration around each test.
@pytest.fixture(autouse=True)
def default_config(request):
    """Autouse fixture that loads the default AEIC `Config` for every test.

    Tests that need to drive `Config.load()` themselves (e.g. to assert on
    the load mechanics, or to load multiple configs in sequence) opt out by
    redefining a fixture of the same name in their own module — pytest's
    fixture resolution will prefer the closer override. See
    `tests/test_config.py` for an example. An overriding fixture should
    accept `request` and assert that no `@pytest.mark.config_updates`
    marker is attached, so users who write the marker on a test that
    happens to fall under the override aren't silently ignored.
    """
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


@pytest.fixture
def performance_model_selector():
    return SimplePerformanceModelSelector(
        config.file_location('performance/simple_selector')
    )


@pytest.fixture
def fuel():
    with open(config.emissions.fuel_file, 'rb') as fp:
        return Fuel.model_validate(tomllib.load(fp))


@pytest.fixture(scope='session')
def lto() -> LTOPerformanceInput:
    """LTO data for the sample EDB engine, as performance model input."""
    entry = EDBEntry.get_engine(EDB_FILE, EDB_UID)
    return LTOPerformanceInput.from_internal(
        entry.make_lto_performance(EDB_THRUST_FRACTIONS)
    )


@pytest.fixture(scope='session')
def _piano_data_cached() -> PianoData:
    """Parse the sample PIANO outputs once. Use `piano_data` instead."""
    return PianoData.load(
        str(PIANO_DIR / 'cruise.txt'),
        str(PIANO_DIR / 'climb.txt'),
        str(PIANO_DIR / 'descent.txt'),
        overrides=PianoOverrides(climb_masses_kg=PIANO_CLIMB_MASSES_KG),
    )


@pytest.fixture
def piano_data(_piano_data_cached) -> PianoData:
    """Parsed sample PIANO exports, one copy per test.

    Each test is given its copy because PianoData is a mutable
    dataclass."""
    return copy.deepcopy(_piano_data_cached)


@pytest.fixture
def ptf_data() -> PTFData:
    """Parsed sample BADA PTF file."""
    return PTFData.load(str(PTF_FILE))

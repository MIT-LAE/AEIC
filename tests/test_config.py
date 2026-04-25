import os
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from AEIC.config import Config, config
from AEIC.config.emissions import ClimbDescentMode, EINOxMethod


# Disable default configuration loading fixture just for this file.
@pytest.fixture(autouse=True)
def default_config():
    yield None
    Config.reset()


def test_missing_config():
    with pytest.raises(ValueError, match='AEIC configuration is not set'):
        Config.get()


def test_load_default_config():
    Config.load()
    # Pin a handful of concrete defaults from default_config.toml so an
    # overlay-logic regression that drops the default branch surfaces here.
    assert config.performance_model.name == 'sample_performance_model.toml'
    assert config.weather.weather_data_dir.name == 'weather'
    assert config.weather.use_weather is True
    assert config.emissions.nox_method == EINOxMethod.BFFM2
    assert config.emissions.climb_descent_mode == ClimbDescentMode.TRAJECTORY


def test_reinitialize_config():
    Config.load()
    with pytest.raises(RuntimeError, match='Config has already been initialized.'):
        Config.load()


def test_load_config_data():
    # Test overrides from dictionary.
    with open(Path(__file__).parent / 'data/config/config-1.toml', 'rb') as fp:
        config_data = tomllib.load(fp)
    Config.load(**config_data)
    config1_checks()


def test_load_config_file():
    # Test overrides from file.
    Config.load(config_file=Path(__file__).parent / 'data/config/config-1.toml')
    config1_checks()


def test_proxy_and_get_match():
    # The `config` module-level proxy and `Config.get()` are two access paths
    # to the same singleton — assert they resolve to equal values rather than
    # duplicating the smoke checks in `test_load_default_config` against each.
    Config.load()
    cfg = Config.get()
    assert cfg.performance_model == config.performance_model
    assert cfg.weather.weather_data_dir == config.weather.weather_data_dir
    assert cfg.emissions.nox_method == config.emissions.nox_method


def config1_checks():
    assert config.weather.use_weather is False
    assert config.emissions.sox_enabled is False
    assert config.emissions.nox_method == EINOxMethod.P3T3
    assert config.emissions.apu_enabled is False


def test_frozen_outer():
    Config.load()
    with pytest.raises(ValidationError):
        config.performance_model = None


def test_frozen_inner():
    Config.load()
    with pytest.raises(ValidationError):
        config.weather.use_weather = False


def test_escape_allows_re_validation_without_overwriting_singleton():
    """`Config.escape()` lets reproducibility-replay code build a second
    ``Config`` without tripping the singleton check or replacing the
    canonical instance — the path used by
    ``trajectories.store._read_reproducibility_info``.
    """
    Config.load()
    primary = Config.get()

    with pytest.raises(RuntimeError, match='Config has already been initialized.'):
        Config.load()

    with Config.escape():
        replay = Config.load()

    assert replay is not primary
    assert Config.get() is primary


def test_data_file_locations(tmp_path: Path, monkeypatch):
    # Test that data file locations are resolved correctly.
    d = tmp_path / 'data'
    d.mkdir()
    test1 = d / 'test1.dat'
    test1.write_text('test')
    sd = d / 'subdir'
    sd.mkdir()
    test2 = sd / 'test2.dat'
    test2.write_text('test')

    monkeypatch.setenv('AEIC_PATH', os.environ['AEIC_PATH'] + ':' + str(d))
    Config.load()

    assert config.file_location(test1.resolve()) == test1.resolve()
    assert config.file_location('test1.dat') == test1.resolve()
    assert config.file_location('subdir/test2.dat') == test2.resolve()
    with pytest.raises(FileNotFoundError):
        config.file_location('nonexistent.dat')
    with pytest.raises(FileNotFoundError):
        config.file_location((d / 'nonexistent.dat').resolve())


def test_default_data_file_location_missing_raises():
    # Pin the raise at config/core.py for a path that exists in neither
    # data_path_overrides nor src/AEIC/data/ — the only branch in
    # `default_data_file_location` not exercised by `test_data_file_locations`.
    Config.load()
    cfg = Config.get()
    with pytest.raises(FileNotFoundError, match='not found in AEIC default data'):
        cfg.default_data_file_location('definitely_does_not_exist.dat')

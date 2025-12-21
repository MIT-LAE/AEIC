import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from AEIC.config import Config, config
from AEIC.emissions.config import EINOxMethod


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
    assert config.performance_model is not None
    assert config.weather.weather_data_dir is not None


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

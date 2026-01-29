# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

import os
import tomllib
from importlib.resources import as_file, files
from pathlib import Path

from pydantic import ConfigDict, Field, model_validator

from AEIC.utils.helpers import deep_update
from AEIC.utils.models import CIBaseModel

from .emissions import EmissionsConfig
from .weather import WeatherConfig


class Config(CIBaseModel):
    """Global AEIC configuration settings.

    This is a singleton class; only one instance can be created. This instance
    can then be accessed as `AEIC.config.config` via the module-level proxy. To
    use this, create an instance of `Config` at the start of your program
    (probably using the `load` method), then anywhere else in the codebase you
    can access the configuration simply by doing `from AEIC.config import
    config`.

    The intention here is to provide a single source of truth for configuration
    settings that can be accessed throughout AEIC without needing to pass
    configuration objects around explicitly. Restricting to a single instance
    of the configuration class helps to avoid inconsistencies in settings during
    execution of code using AEIC."""

    model_config = ConfigDict(frozen=True)
    """Configuration is frozen after creation."""

    path: list[Path] = Field(default_factory=list)
    """List of paths to search for data files. If not initialized explicitly,
    this is taken from the AEIC_PATH environment variable if set, or defaults
    to the current working directory only."""

    data_path_overrides: list[Path] = Field(default_factory=list)
    """List of paths used for overriding the normal data directory search. Used
    for testing."""

    performance_model: Path
    """Path to performance model data file."""

    engine_file: Path
    """Path to engine database file."""

    weather: WeatherConfig
    """Global weather configuration settings."""

    emissions: EmissionsConfig
    """Global emissions configuration settings."""

    # There is some slightly tricky stuff going on here. We want configuration
    # instances to be frozen (immutable) after initialization, but we also want
    # to perform some validation and path normalization after the instance is
    # created. Because of the way that model freezing works in Pydantic, we
    # can't modify the instance in-place after creation in the model
    # validators. Instead, we use `object.__setattr__` to bypass the
    # immutability and set the attributes we need to modify. From the user's
    # perspective, the instance is still immutable after creation.

    @model_validator(mode='after')
    def normalize_search_paths(self):
        """Resolve search paths and initialize the global configuration
        singleton."""

        global _config
        if _config is not None:
            raise RuntimeError('Config has already been initialized.')
        try:
            self._normalize_path()
        finally:
            _config = self

        return self

    @model_validator(mode='after')
    def resolve_paths(self):
        if getattr(self, 'performance_model') is not None:
            object.__setattr__(
                self,
                'performance_model',
                Path(self.file_location(getattr(self, 'performance_model'))).resolve(),
            )
        if getattr(self, 'engine_file') is not None:
            object.__setattr__(
                self,
                'engine_file',
                Path(self.file_location(getattr(self, 'engine_file'))).resolve(),
            )
        if self.weather.weather_data_dir is not None:
            object.__setattr__(
                self.weather,
                'weather_data_dir',
                Path(self.file_location(self.weather.weather_data_dir)).resolve(),
            )
        return self

    def file_location(self, f: Path | str) -> Path:
        """Get path to a file, checking local and configured paths."""

        f = f if isinstance(f, Path) else Path(f)

        if f.exists():
            return f.resolve()
        return self.data_file_location(f)

    def default_data_file_location(
        self, f: Path | str, missing_ok: bool = False
    ) -> Path:
        """Get the full path to a file within the default data directory."""

        for override_path in self.data_path_overrides:
            override_file = override_path / f
            if override_file.exists():
                return override_file.resolve()
        with as_file(files('AEIC') / 'data') as data_dir:
            default_path = data_dir / f
            if default_path.exists() or missing_ok:
                return default_path.resolve()

        raise FileNotFoundError(f'File {f} not found in AEIC default data directory.')

    def data_file_location(self, f: Path | str) -> Path:
        """Get the full path to a file within the configured paths."""

        f = f if isinstance(f, Path) else Path(f)

        if f.is_absolute():
            if f.exists():
                return f
            else:
                raise FileNotFoundError(f'File {f} not found.')

        for p in self.path:
            file_path = p / f
            if file_path.exists():
                return file_path

        raise FileNotFoundError(f'File {f} not found in AEIC search path.')

    @classmethod
    def get(cls) -> Config:
        """Get the global configuration singleton.

        Raises an error if the configuration has not yet been initialized."""
        global _config
        if _config is None:
            raise ValueError('AEIC configuration is not set')
        return _config

    @classmethod
    def load(cls, config_file: str | Path | None = None, **kwargs) -> Config:
        """Load configuration from TOML files.

        The `default_config.toml` file included with AEIC is loaded first, and
        then TOML data from any `config_file` provided is loaded and overlaid
        on top. Additional keyword arguments are finally applied on top of the
        resulting configuration data. This allows users to only specify
        configuration options that differ from the defaults."""

        # Read default configuration data: this is in the top-level package
        # source directory to ensure that it ends up in the wheel.
        with open(
            Path(__file__).parent.parent / 'data/default_config.toml', 'rb'
        ) as fp:
            default_data = tomllib.load(fp)

        # Overlay data can come either from a file or from keyword arguments.
        overlay_data = {}

        # Read overlay configuration data from TOML file.
        if config_file is not None:
            with open(config_file, 'rb') as fp:
                overlay_data = tomllib.load(fp)

        # Handle overlay data from keyword arguments.
        overlay_data = deep_update(overlay_data, kwargs)

        # Combine default and overlay data and validate.
        return cls.model_validate(deep_update(default_data, overlay_data))

    @staticmethod
    def reset():
        """Reset the global configuration singleton.

        This is mostly intended for testing purposes, where it can be useful to
        modify the configuration between or within tests. In most non-test use
        cases, the intention is to create a single configuration instance at the
        start of the program and use that instance throughout."""
        global _config
        _config = None

    def _normalize_path(self) -> None:
        # Path was explicitly set when constructing the instance.
        if len(self.path) > 0:
            object.__setattr__(self, 'path', [Path(p).resolve() for p in self.path])
            return

        # Otherwise initialize from the AEIC_PATH environment variable.
        path_env = os.environ.get('AEIC_PATH', '')
        if path_env != '':
            # Path from AEIC_PATH environment variable.
            object.__setattr__(
                self, 'path', [Path(p).resolve() for p in path_env.split(os.pathsep)]
            )

        # Add package data directory as a fallback.
        with as_file(files('AEIC') / 'data') as data_dir:
            object.__setattr__(self, 'path', self.path + [data_dir.resolve()])


# Module property-like access to configuration via a proxy to allow late
# initialization.

_config: Config | None = None


class ConfigProxy:
    def __getattr__(self, name):
        global _config
        if _config is None:
            raise ValueError('AEIC configuration is not set')
        return getattr(_config, name)

    def __setattr__(self, name, value):
        global _config
        if _config is None:
            raise ValueError('AEIC configuration is not set')
        return setattr(_config, name, value)


config = ConfigProxy()

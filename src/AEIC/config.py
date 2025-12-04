import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Global AEIC configuration settings.

    This is a singleton class; only one instance should be created. This
    instance can then be accessed as `AEIC.config.config` via the module-level
    proxy. To use this, create an instance of `Config` at the start of your
    program, then anywhere else in the codebase you can access the configuration
    simply by doing `from AEIC.config import config`."""

    path: list[Path] = field(default_factory=list)
    """List of paths to search for data files. If not initialized explicitly,
    this is taken from the AEIC_PATH environment variable if set, or defaults
    to the current working directory only."""

    def __post_init__(self) -> None:
        """Resolve search paths and initialize the global configuration
        singleton."""

        global _config
        if _config is not None:
            raise RuntimeError('Config has already been initialized.')
        try:
            self._normalize_path()
        finally:
            _config = self

    def _normalize_path(self) -> None:
        # Explicitly set path.
        if len(self.path) > 0:
            self.path = [Path(p).resolve() for p in self.path]
            return
        path_env = os.environ.get('AEIC_PATH', '')
        if path_env != '':
            # Path from AEIC_PATH environment variable.
            self.path = [Path(p).resolve() for p in path_env.split(os.pathsep)]
        else:
            # Default to current working directory.
            self.path = [Path.cwd().resolve()]

    def file_location(self, f: Path | str) -> Path:
        """Get path to a file, checking local and configured paths."""

        f = f if isinstance(f, Path) else Path(f)

        if f.exists():
            return f.resolve()
        return self.data_file_location(f)

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


# Module property-like access to configuration via a proxy to allow late
# initialization.

_config: Config | None = None


class ConfigProxy:
    def __getattr__(self, name):
        global _config
        if _config is None:
            print('Configuration is not set, using default values')
            _config = Config()
        return getattr(_config, name)


config = ConfigProxy()

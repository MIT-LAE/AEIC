import logging
import tomllib
from pathlib import Path
from typing import Protocol, runtime_checkable

from cachetools import LRUCache
from pydantic import ValidationError

from AEIC.missions import Mission
from AEIC.performance.models import BasePerformanceModel, PerformanceModel

logger = logging.getLogger(__name__)


@runtime_checkable
class PerformanceModelSelector(Protocol):
    """Protocol defining the interface for performance model selectors. A
    performance model selector is a callable that takes a Mission and returns a
    performance model to use for that mission."""

    def __call__(self, mission: Mission) -> BasePerformanceModel: ...


class SimplePerformanceModelSelector:
    """A simple performance model selector that selects a performance model
    based on the aircraft type of the mission. The performance model files are
    stored in a directory provided to the class constructor. A TOML
    configuration file in the same specifies a default performance model and
    performance models to use for aircraft types that do not have their own
    performance models.

    Performance model lookup proceeds by first looking for a performance model
    file with the same name as the aircraft type (with a `.toml` extension). If
    no such file exists, then the configuration file is checked for a synonym
    for that aircraft type, and if a synonym exists, the performance model file
    for that aircraft type is used. If no synonym exists, then the default
    performance model is used."""

    def __init__(self, directory: Path, cache_size: int = 16):
        """Initialize the performance model selector.

        The directory must contain a `config.toml` file with lines of the form
        `<aircraft-type> = <performance-model-file>`, which specify synonyms
        for aircraft types that do not have their own performance model files.
        The `config.toml` file must also contain a `default` entry specifying
        the default performance model file to use when no performance model
        file or synonym exists for a given aircraft type.

        The performance model selector maintains a LRU cache of performance
        models that have been loaded, keyed by the name of the performance
        model file (i.e., the aircraft type). The `cache_size` parameter
        specifies the maximum size of this cache."""

        if not directory.exists():
            raise ValueError(f'performance model directory does not exist: {directory}')
        if not (directory / 'config.toml').exists():
            raise ValueError(
                f'performance model directory {directory} does not contain config.toml'
            )

        self.directory = directory

        # Cache for performance models, keyed by name of performance model file
        # (i.e., aircraft type).
        self._cache: LRUCache[str, BasePerformanceModel] = LRUCache(maxsize=cache_size)

        # Read configuration TOML file and make mapping from aircraft type to
        # performance model file.
        with open(directory / 'config.toml', 'rb') as fp:
            self.synonyms = {k: str(v) for k, v in tomllib.load(fp).items()}

        # Check existence of referenced performance model files.
        for f in self.synonyms.values():
            if not (directory / f'{f}.toml').exists():
                logger.warning(
                    f'performance model file {f} referenced '
                    'in config.toml does not exist'
                )

        # Handle default performance model.
        self.default_pm: BasePerformanceModel | None = None
        if 'default' in self.synonyms:
            self.default_pm = self._get(self.synonyms['default'])
            del self.synonyms['default']
        else:
            logger.warning(
                'performance model selector does not contain a "default" entry'
            )

    def _exists(self, ac_type: str) -> bool:
        """Check if a performance model file exists for the given aircraft type."""
        return (self.directory / f'{ac_type}.toml').exists()

    def _get(self, ac_type: str) -> BasePerformanceModel | None:
        """Get a performance model from the cache, or load it from file if not
        in the cache."""
        if ac_type in self._cache:
            return self._cache[ac_type]
        pm_path = self.directory / f'{ac_type}.toml'
        try:
            self._cache[ac_type] = PerformanceModel.load(pm_path)
            return self._cache[ac_type]
        except FileNotFoundError:
            return None
        except ValidationError:
            logger.exception(
                'failed to validate performance model for aircraft type %s at %s',
                ac_type,
                pm_path,
            )
            raise

    def __call__(self, mission: Mission) -> BasePerformanceModel | None:
        """Main API for looking up a performance model for a mission. This is
        all that's required to satisfy the `PerformanceModelSelector`
        protocol."""

        # Get aircraft type from mission.
        ac_type = mission.aircraft_type

        # Find a performance model file with that name.
        if self._exists(ac_type):
            return self._get(ac_type)

        # None exists, so look for synonyms of that aircraft type from the
        # configuration file.
        if ac_type in self.synonyms:
            return self._get(self.synonyms[ac_type])

        # No aircraft type file exists, and no synonym exists. Use default
        # performance model.
        return self.default_pm

import re
from pathlib import Path

from pydantic import ConfigDict, model_validator

from AEIC.utils.models import CIBaseModel, CIStrEnum


class WeatherResolution(CIStrEnum):
    """ERA5 data temporal layout on disk."""

    ANNUAL_MEAN = 'annual_mean'
    """One file per year containing an annual mean (no within-file time
    dimension, or a single-entry valid_time that gets squeezed)."""

    MONTHLY_MEAN = 'monthly_mean'
    """One file per month containing a monthly mean."""

    DAILY_MEAN = 'daily_mean'
    """One file per day containing a daily mean."""

    HOURLY_DAILY_FILES = 'hourly_daily_files'
    """One file per day, each with a 24-entry hourly ``valid_time`` dim."""

    HOURLY_MONTHLY_FILES = 'hourly_monthly_files'
    """One file per month, each with an hourly ``valid_time`` dim spanning
    the month. This is the most common CDS download layout."""


# Tokens permitted anywhere in ``file_format``.
_ALLOWED_TOKENS: frozenset[str] = frozenset({'Y', 'm', 'd', 'j', 'H'})

# Per-resolution validation rules. ``required_any`` is a set of tokens where
# at least one must be present; ``allowed`` is the full set of permitted
# tokens for that resolution.
_RESOLUTION_RULES: dict[WeatherResolution, tuple[frozenset[str], frozenset[str]]] = {
    WeatherResolution.ANNUAL_MEAN: (frozenset(), frozenset({'Y'})),
    WeatherResolution.MONTHLY_MEAN: (frozenset({'m'}), frozenset({'Y', 'm'})),
    WeatherResolution.DAILY_MEAN: (
        frozenset({'d', 'j'}),
        frozenset({'Y', 'm', 'd', 'j'}),
    ),
    WeatherResolution.HOURLY_DAILY_FILES: (
        frozenset({'d', 'j'}),
        frozenset({'Y', 'm', 'd', 'j'}),
    ),
    WeatherResolution.HOURLY_MONTHLY_FILES: (frozenset({'m'}), frozenset({'Y', 'm'})),
}

# Matches strftime tokens in a format string, ignoring literal ``%%``.
_TOKEN_RE = re.compile(r'%(.)')


def _extract_tokens(file_format: str) -> set[str]:
    """Return the set of non-literal strftime tokens present in ``file_format``."""
    return {m.group(1) for m in _TOKEN_RE.finditer(file_format) if m.group(1) != '%'}


class WeatherConfig(CIBaseModel):
    """Configuration settings for weather module."""

    model_config = ConfigDict(frozen=True)
    """Configuration is frozen after creation."""

    use_weather: bool = True
    """Whether to use weather data for emissions calculations."""

    weather_data_dir: Path | None = None
    """Directory path for weather data files. Filenames within this directory
    are resolved from ``file_format`` via ``strftime``. If None, defaults to
    the current working directory."""

    resolution: WeatherResolution = WeatherResolution.HOURLY_DAILY_FILES
    """Temporal layout of the ERA5 data on disk. See ``WeatherResolution``."""

    file_format: str = '%Y%m%d.nc'
    """``strftime``-style pattern (relative to ``weather_data_dir``) used to
    resolve a filename from a timestamp. Permitted tokens: ``%Y``, ``%m``,
    ``%d``, ``%j``, ``%H``. Allowed/required tokens depend on
    ``resolution``."""

    @model_validator(mode='after')
    def _validate_file_format(self) -> 'WeatherConfig':
        tokens = _extract_tokens(self.file_format)
        unknown = tokens - _ALLOWED_TOKENS
        if unknown:
            raise ValueError(
                f'file_format contains unsupported strftime token(s): '
                f'{sorted("%" + t for t in unknown)}. Allowed tokens: '
                f'{sorted("%" + t for t in _ALLOWED_TOKENS)}.'
            )

        required_any, allowed = _RESOLUTION_RULES[self.resolution]

        forbidden = tokens - allowed
        if forbidden:
            raise ValueError(
                f'file_format contains token(s) '
                f'{sorted("%" + t for t in forbidden)} which are not allowed '
                f'for resolution {self.resolution}. Allowed tokens for this '
                f'resolution: {sorted("%" + t for t in allowed)}.'
            )

        if required_any and not (tokens & required_any):
            raise ValueError(
                f'file_format must contain at least one of '
                f'{sorted("%" + t for t in required_any)} for resolution '
                f'{self.resolution}.'
            )

        if not tokens and self.resolution is not WeatherResolution.ANNUAL_MEAN:
            raise ValueError(
                f'file_format {self.file_format!r} has no strftime tokens; '
                f'every timestamp would resolve to the same file. Only '
                f'{WeatherResolution.ANNUAL_MEAN} permits a literal filename.'
            )

        return self

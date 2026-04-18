import re
from pathlib import Path

from pydantic import ConfigDict, model_validator

from AEIC.utils.models import CIBaseModel, CIStrEnum


class TemporalResolution(CIStrEnum):
    """Temporal resolution for weather files and the data within them."""

    HOURLY = 'hourly'
    DAILY = 'daily'
    MONTHLY = 'monthly'
    ANNUAL = 'annual'


# Rank for ordering. Used to enforce ``data_resolution <= file_resolution``.
_RANK: dict[TemporalResolution, int] = {
    TemporalResolution.HOURLY: 1,
    TemporalResolution.DAILY: 2,
    TemporalResolution.MONTHLY: 3,
    TemporalResolution.ANNUAL: 4,
}

# Resolutions valid as ``file_resolution``. Per-hour files are not supported.
_FILE_RESOLUTIONS: frozenset[TemporalResolution] = frozenset(
    {
        TemporalResolution.DAILY,
        TemporalResolution.MONTHLY,
        TemporalResolution.ANNUAL,
    }
)

# Default ``file_format`` derived from ``file_resolution``.
_DEFAULT_FILE_FORMATS: dict[TemporalResolution, str] = {
    TemporalResolution.ANNUAL: '%Y.nc',
    TemporalResolution.MONTHLY: '%Y-%m.nc',
    TemporalResolution.DAILY: '%Y-%m-%d.nc',
}

# Per-file_resolution validation rules for ``file_format`` overrides.
# ``required_any`` is a set of tokens where at least one must be present;
# ``allowed`` is the full set of permitted strftime tokens.
_FILE_FORMAT_RULES: dict[TemporalResolution, tuple[frozenset[str], frozenset[str]]] = {
    TemporalResolution.ANNUAL: (frozenset(), frozenset({'Y'})),
    TemporalResolution.MONTHLY: (frozenset({'m'}), frozenset({'Y', 'm'})),
    TemporalResolution.DAILY: (
        frozenset({'d', 'j'}),
        frozenset({'Y', 'm', 'd', 'j'}),
    ),
}

# Tokens permitted anywhere in ``file_format``. No time-of-day tokens since
# ``file_resolution`` cannot be hourly.
_ALLOWED_TOKENS: frozenset[str] = frozenset({'Y', 'm', 'd', 'j'})

# Matches strftime tokens, ignoring literal ``%%``.
_TOKEN_RE = re.compile(r'%(.)')


def _extract_tokens(file_format: str) -> set[str]:
    """Return the set of non-literal strftime tokens present in ``file_format``."""
    return {m.group(1) for m in _TOKEN_RE.finditer(file_format) if m.group(1) != '%'}


def default_file_format(file_resolution: TemporalResolution) -> str:
    """Default ``file_format`` for a given ``file_resolution``."""
    return _DEFAULT_FILE_FORMATS[file_resolution]


def resolution_le(a: TemporalResolution, b: TemporalResolution) -> bool:
    """Return True iff ``a`` is finer-or-equal to ``b`` (``a <= b``)."""
    return _RANK[a] <= _RANK[b]


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

    file_resolution: TemporalResolution = TemporalResolution.DAILY
    """Temporal layout of files on disk: one file per ``file_resolution``
    period. One of ``annual``, ``monthly``, ``daily`` (``hourly`` is rejected
    -- per-hour files are not supported)."""

    data_resolution: TemporalResolution | None = None
    """Temporal resolution of the data within each file. Optional; defaults to
    ``file_resolution`` (i.e., one period-mean per file). Must satisfy
    ``data_resolution <= file_resolution``."""

    file_format: str | None = None
    """``strftime``-style filename pattern (relative to ``weather_data_dir``).
    Optional; if omitted, defaults are derived from ``file_resolution``:
    ``%Y.nc`` for annual, ``%Y-%m.nc`` for monthly, ``%Y-%m-%d.nc`` for daily.
    Permitted tokens depend on ``file_resolution``; ``data_resolution`` does
    not influence filename rules."""

    @property
    def effective_data_resolution(self) -> TemporalResolution:
        """``data_resolution`` with the file_resolution default applied."""
        if self.data_resolution is None:
            return self.file_resolution
        return self.data_resolution

    @property
    def effective_file_format(self) -> str:
        """``file_format`` with the file_resolution-derived default applied."""
        if self.file_format is None:
            return _DEFAULT_FILE_FORMATS[self.file_resolution]
        return self.file_format

    @model_validator(mode='after')
    def _validate(self) -> 'WeatherConfig':
        if self.file_resolution not in _FILE_RESOLUTIONS:
            raise ValueError(
                f'file_resolution must be one of '
                f'{sorted(r.value for r in _FILE_RESOLUTIONS)}; got '
                f'{self.file_resolution.value!r}. Per-hour files are not '
                f'supported.'
            )

        data_res = self.effective_data_resolution
        if not resolution_le(data_res, self.file_resolution):
            raise ValueError(
                f'data_resolution ({data_res.value}) must be finer-or-equal '
                f'to file_resolution ({self.file_resolution.value}).'
            )

        fmt = self.effective_file_format
        tokens = _extract_tokens(fmt)
        unknown = tokens - _ALLOWED_TOKENS
        if unknown:
            raise ValueError(
                f'file_format contains unsupported strftime token(s): '
                f'{sorted("%" + t for t in unknown)}. Allowed tokens: '
                f'{sorted("%" + t for t in _ALLOWED_TOKENS)}.'
            )

        required_any, allowed = _FILE_FORMAT_RULES[self.file_resolution]

        forbidden = tokens - allowed
        if forbidden:
            raise ValueError(
                f'file_format contains token(s) '
                f'{sorted("%" + t for t in forbidden)} which are not allowed '
                f'for file_resolution={self.file_resolution.value}. Allowed '
                f'tokens: {sorted("%" + t for t in allowed)}.'
            )

        if required_any and not (tokens & required_any):
            raise ValueError(
                f'file_format must contain at least one of '
                f'{sorted("%" + t for t in required_any)} for '
                f'file_resolution={self.file_resolution.value}.'
            )

        if not tokens and self.file_resolution is not TemporalResolution.ANNUAL:
            raise ValueError(
                f'file_format {fmt!r} has no strftime tokens; every '
                f'timestamp would resolve to the same file. Only '
                f'file_resolution=annual permits a literal filename.'
            )

        return self

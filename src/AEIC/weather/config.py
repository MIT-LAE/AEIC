from pathlib import Path

from pydantic import ConfigDict

from AEIC.utils.types import CIBaseModel


class WeatherConfig(CIBaseModel):
    """Configuration settings for weather module."""

    # Freeze model after initialization.
    model_config = ConfigDict(frozen=True)

    use_weather: bool = True
    """Whether to use weather data for emissions calculations."""

    weather_data_dir: Path | None = None
    """Directory path for weather data files. (Files should be NetCDF files
    following ERA5 conventions with names of the form YYYYMMDD.nc.) If None,
    defaults to the current working directory."""

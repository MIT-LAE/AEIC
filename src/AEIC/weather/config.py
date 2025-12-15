from pathlib import Path

from pydantic import ConfigDict

from AEIC.utils.models import CIBaseModel


class WeatherConfig(CIBaseModel):
    """Configuration settings for weather module."""

    model_config = ConfigDict(frozen=True)
    """Configuration is frozen after creation."""

    use_weather: bool = True
    """Whether to use weather data for emissions calculations."""

    weather_data_dir: Path | None = None
    """Directory path for weather data files. (Files should be NetCDF files
    following ERA5 conventions with names of the form YYYYMMDD.nc.) If None,
    defaults to the current working directory."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from AEIC.config import config
from AEIC.config.weather import WeatherResolution
from AEIC.trajectories.ground_track import GroundTrack
from AEIC.utils.standard_atmosphere import pressure_at_altitude_isa_bada4

_HOURLY_RESOLUTIONS = frozenset(
    {
        WeatherResolution.HOURLY_DAILY_FILES,
        WeatherResolution.HOURLY_MONTHLY_FILES,
    }
)


class Weather:
    """
    A class to query weather data variables and ground speed along
    ground track points.

    Parameters
    ----------
    data_dir : str | Path
        Path to directory containing ERA5 weather data NetCDF files. The
        filename for a given timestamp is resolved via ``file_format``.
        Files should contain variables ``t``, ``u``, ``v`` with coordinates
        ``pressure_level``, ``latitude``, ``longitude``. A ``valid_time``
        coord is required for hourly resolutions and ignored (or squeezed
        if length 1) for mean resolutions.
    resolution : WeatherResolution
        Layout of the ERA5 data on disk.
    file_format : str
        ``strftime``-style pattern (relative to ``data_dir``) for mapping a
        timestamp to a filename. Tokens permitted depend on ``resolution``.
        Validated by :class:`AEIC.config.weather.WeatherConfig`.
    """

    def __init__(
        self,
        data_dir: str | Path,
        resolution: WeatherResolution = WeatherResolution.HOURLY_DAILY_FILES,
        file_format: str = '%Y%m%d.nc',
    ):
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f'Weather data directory not found: {self.data_dir}'
            )
        self._resolution = resolution
        self._file_format = file_format

        self._main_ds: xr.Dataset | None = None
        self._ds_key: str | None = None
        self._ds: xr.Dataset | None = None
        self._last_sel_time: pd.Timestamp | None = None

    @staticmethod
    def _to_utc_naive(time: pd.Timestamp) -> pd.Timestamp:
        """Coerce a timestamp to tz-naive UTC. Tz-naive inputs are assumed UTC."""
        if time.tzinfo is not None:
            time = time.tz_convert('UTC').tz_localize(None)
        return time

    def _resolved_name(self, time: pd.Timestamp) -> str:
        return time.strftime(self._file_format)

    def _nc_path(self, time: pd.Timestamp) -> Path:
        return Path(
            config.file_location(str(self.data_dir / self._resolved_name(time)))
        )

    def _require_main_ds(self, time: pd.Timestamp):
        key = self._resolved_name(time)
        if self._main_ds is not None and self._ds_key == key:
            return

        self._ds = None
        self._last_sel_time = None

        if self._main_ds is not None:
            self._main_ds.close()
            self._main_ds = None
            gc.collect()

        self._main_ds = xr.open_dataset(self._nc_path(time))
        self._ds_key = key

        if (
            self._resolution in _HOURLY_RESOLUTIONS
            and 'valid_time' in self._main_ds.dims
        ):
            valid_time_dtype = self._main_ds['valid_time'].dtype
            if not np.issubdtype(valid_time_dtype, np.datetime64):
                raise TypeError(
                    f'{self._nc_path(time)}: valid_time has non-datetime '
                    f'dtype {valid_time_dtype}; hourly resolutions require a '
                    f'datetime64 valid_time coord.'
                )

    def _require_data(self, time: pd.Timestamp):
        time = self._to_utc_naive(time)
        self._require_main_ds(time)

        if self._ds is not None and self._last_sel_time == time:
            return

        assert self._main_ds is not None

        if self._resolution in _HOURLY_RESOLUTIONS:
            if 'valid_time' in self._main_ds.dims:
                self._ds = self._main_ds.sel(
                    valid_time=time,
                    method='nearest',
                    tolerance=pd.Timedelta('1h'),
                )
            else:
                self._ds = self._main_ds
        else:
            # Mean resolutions: squeeze a length-1 valid_time if present.
            if 'valid_time' in self._main_ds.dims:
                self._ds = self._main_ds.squeeze('valid_time', drop=True)
            else:
                self._ds = self._main_ds

        self._last_sel_time = time

    def get_ground_speed(
        self,
        time: pd.Timestamp,
        gt_point: GroundTrack.Point,
        altitude: float,
        true_airspeed: float,
        azimuth: float | None = None,
    ) -> float:
        """
        Compute ground speed at a point along the mission.

        Parameters
        ----------
        time: pd.Timestamp
            Time at the ground track point. Interpreted as UTC; tz-aware
            timestamps are converted to UTC, tz-naive timestamps are assumed
            UTC.
        gt_point : GroundTrack.Point
            Spatial point along the ground track from the origin.
        altitude : float
            Altitude above sea level [meters].
        true_airspeed : float
            True airspeed [m/s].
        azmiuth : float, optional
            Azimuth [degrees].
            If omitted, use the precomputed ground-track azmith.

        Returns
        -------
        ground_speed: float
            Ground speed [m/s]
        """

        self._require_data(time)
        assert self._ds is not None

        # NOTE: pressure levels in weather files are in hPa, not Pa.
        wind_u = self._ds['u'].interp(
            pressure_level=pressure_at_altitude_isa_bada4(altitude) / 100.0,
            latitude=gt_point.location.latitude,
            longitude=gt_point.location.longitude,
        )
        wind_v = self._ds['v'].interp(
            pressure_level=pressure_at_altitude_isa_bada4(altitude) / 100.0,
            latitude=gt_point.location.latitude,
            longitude=gt_point.location.longitude,
        )
        if wind_u.isnull().values.any() or wind_v.isnull().values.any():
            raise ValueError('ground track point is outside weather data domain')

        if azimuth is None:
            heading_rad = np.deg2rad(gt_point.azimuth)
        else:
            heading_rad = np.deg2rad(azimuth)

        u_air = true_airspeed * np.cos(heading_rad)
        v_air = true_airspeed * np.sin(heading_rad)

        return float(np.hypot(u_air + wind_u, v_air + wind_v))

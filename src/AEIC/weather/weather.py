import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from AEIC.config import config
from AEIC.trajectories.ground_track import GroundTrack


class Weather:
    """
    A class to query weather data variables and ground speed along
    ground track points.

    Parameters
    ----------
    data_dir : str | Path
        Path to directory containing ERA5 weather data NetCDF files.
        The files should have names of the form 'YYYYMMDD.nc', one per day.
        File should contain variables: 't', 'u', 'v'
        with coordinates 'pressure_level', 'latitude', 'longitude',
        'valid_time' (optional)
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f'Weather data directory not found: {self.data_dir}'
            )
        self._main_ds: xr.Dataset | None = None
        self._ds_date: pd.Timestamp | None = None
        self._ds: xr.Dataset | None = None
        self._ds_time_idx: int | None = None

    def _nc_path(self, time: pd.Timestamp) -> Path:
        fname = time.strftime('%Y%m%d.nc')
        return Path(config.file_location(str(self.data_dir / fname)))

    def _require_main_ds(self, time: pd.Timestamp):
        # If we already have the main Dataset for this date, return.
        if self._main_ds is not None and self._ds_date == time:
            return

        # If we're reading a new dataset, we will need to slice it.
        self._ds = None
        self._ds_time_idx = None

        # If we already have a weather dataset open, close it and force the
        # garbage collector to close the underlying NetCDF file.
        if self._main_ds is not None:
            self._main_ds.close()
            self._main_ds = None
            gc.collect()

        # Read weather file into main Dataset.
        self._main_ds = xr.open_dataset(self._nc_path(time))
        self._ds_date = time

    def _require_data(self, time: pd.Timestamp):
        # Make sure we have the main Dataset for this date.
        self._require_main_ds(time)

        # If we already have the sliced Dataset for this time, return.
        if self._ds is not None and self._ds_time_idx == time.hour:
            return

        # If valid_hour exists, slice weather to get hour of departure.
        assert self._main_ds is not None
        self._ds = self._main_ds
        self._ds_time_idx = None
        if 'valid_time' in self._main_ds.dims:
            self._ds = self._main_ds.isel(valid_time=time.hour)
            self._ds_time_idx = time.hour

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
            Time at the ground track point [UTC].
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

        wind_u = self._ds['u'].interp(
            pressure_level=_altitude_to_pressure_level_hPa(altitude),
            latitude=gt_point.location.latitude,
            longitude=gt_point.location.longitude,
        )
        wind_v = self._ds['v'].interp(
            pressure_level=_altitude_to_pressure_level_hPa(altitude),
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


def _altitude_to_pressure_level_hPa(altitude: float) -> float:
    """Convert altitude to pressure level."""
    return 1013.25 * (1.0 - altitude / 44330.0) ** 5.255

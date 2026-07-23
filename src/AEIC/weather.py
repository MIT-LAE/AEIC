import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from AEIC.config import config
from AEIC.config.weather import (
    TemporalResolution,
    default_file_format,
    resolution_le,
)
from AEIC.trajectories.ground_track import GroundTrack
from AEIC.utils.standard_atmosphere import pressure_at_altitude_isa_bada4


@dataclass(frozen=True)
class GroundTrackVector:
    """Aircraft ground speed and heading required
    to follow a prescribed ground track."""

    ground_speed: float
    heading: float


def solve_track_vector(
    track_azimuth: float,
    horizontal_airspeed: float,
    wind_east: float,
    wind_north: float,
) -> GroundTrackVector:
    """Solve the wind triangle for a prescribed ground-track azimuth."""
    if horizontal_airspeed <= 0:
        raise ValueError('horizontal airspeed must be positive')

    track_rad = np.deg2rad(track_azimuth)
    wind_parallel = wind_east * np.sin(track_rad) + wind_north * np.cos(track_rad)
    wind_cross = wind_east * np.cos(track_rad) - wind_north * np.sin(track_rad)
    if abs(wind_cross) >= horizontal_airspeed:
        raise ValueError('crosswind is too strong to maintain the prescribed track')

    air_parallel = np.sqrt(horizontal_airspeed**2 - wind_cross**2)
    ground_speed = air_parallel + wind_parallel
    if ground_speed <= 0:
        raise ValueError('wind produces non-positive along-track ground speed')

    # Difference between an aircraft's heading (where the nose is pointed)
    # and its actual track over the ground
    crab_angle = np.arctan2(-wind_cross, air_parallel)
    heading = (track_azimuth + np.rad2deg(crab_angle)) % 360
    return GroundTrackVector(float(ground_speed), float(heading))


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
        coord is required when ``data_resolution`` is finer than
        ``file_resolution`` and is otherwise either absent or length-1.
    file_resolution : TemporalResolution
        Temporal layout of files on disk: one file per ``file_resolution``
        period. Must be ``annual``, ``monthly``, or ``daily``.
    data_resolution : TemporalResolution, optional
        Temporal resolution of the data within each file. Defaults to
        ``file_resolution`` (one period-mean per file). Must satisfy
        ``data_resolution <= file_resolution``.
    file_format : str, optional
        ``strftime``-style pattern (relative to ``data_dir``) for mapping a
        timestamp to a filename. Defaults are derived from
        ``file_resolution``: ``%Y.nc`` (annual), ``%Y-%m.nc`` (monthly),
        ``%Y-%m-%d.nc`` (daily).
    """

    def __init__(
        self,
        data_dir: str | Path,
        file_resolution: TemporalResolution,
        data_resolution: TemporalResolution | None = None,
        file_format: str | None = None,
    ):
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f'Weather data directory not found: {self.data_dir}'
            )

        if file_resolution is TemporalResolution.HOURLY:
            raise ValueError(
                'file_resolution=hourly is not supported (per-hour files).'
            )

        self._file_resolution = file_resolution
        self._data_resolution = (
            data_resolution if data_resolution is not None else file_resolution
        )
        if not resolution_le(self._data_resolution, self._file_resolution):
            raise ValueError(
                f'data_resolution ({self._data_resolution.value}) must be '
                f'finer-or-equal to file_resolution '
                f'({self._file_resolution.value}).'
            )
        self._file_format = (
            file_format
            if file_format is not None
            else default_file_format(file_resolution)
        )

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

    def _validate_file_content(self, ds: xr.Dataset, path: Path) -> None:
        """L1 check: file's valid_time length is consistent with data_resolution."""
        has_vt = 'valid_time' in ds.dims
        n = ds.sizes['valid_time'] if has_vt else 0

        if self._data_resolution == self._file_resolution:
            if has_vt and n > 1:
                raise ValueError(
                    f'{path}: data_resolution=={self._data_resolution.value} '
                    f'(equal to file_resolution); expected 0 or 1 valid_time '
                    f'entries but file has {n}. Either the config is wrong or '
                    f'the file contains finer-resolution data.'
                )
        else:
            if not has_vt:
                raise ValueError(
                    f'{path}: data_resolution={self._data_resolution.value} '
                    f'in {self._file_resolution.value} files requires multiple '
                    f'valid_time entries, but file has no valid_time dim.'
                )
            if n <= 1:
                raise ValueError(
                    f'{path}: data_resolution={self._data_resolution.value} '
                    f'in {self._file_resolution.value} files requires multiple '
                    f'valid_time entries, but file has {n}.'
                )

        if has_vt:
            valid_time_dtype = ds['valid_time'].dtype
            if not np.issubdtype(valid_time_dtype, np.datetime64):
                raise TypeError(
                    f'{path}: valid_time has non-datetime dtype '
                    f'{valid_time_dtype}; a datetime64 valid_time coord is '
                    f'required.'
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

        path = self._nc_path(time)
        self._main_ds = xr.open_dataset(path)
        self._ds_key = key

        self._validate_file_content(self._main_ds, path)

    def _select_by_components(self, ds: xr.Dataset, time: pd.Timestamp) -> xr.Dataset:
        """Pick the entry whose date components match ``time`` for the
        configured ``data_resolution``. Round-then-exact-match semantics."""
        vt = ds['valid_time']

        if self._data_resolution is TemporalResolution.HOURLY:
            target = pd.Timestamp(time).floor('h')
            positions = np.where(vt.dt.floor('h').values == np.datetime64(target))[0]
        elif self._data_resolution is TemporalResolution.DAILY:
            positions = np.where(
                (vt.dt.year.values == time.year)
                & (vt.dt.month.values == time.month)
                & (vt.dt.day.values == time.day)
            )[0]
        elif self._data_resolution is TemporalResolution.MONTHLY:
            positions = np.where(
                (vt.dt.year.values == time.year) & (vt.dt.month.values == time.month)
            )[0]
        else:
            # Annual data only ever lives in annual files (data <= file), and
            # that case takes the squeeze path, never this one.
            raise AssertionError(
                f'unexpected data_resolution {self._data_resolution} in '
                f'component-match path'
            )

        if len(positions) == 0:
            raise KeyError(
                f'no {self._data_resolution.value} entry matching {time} in '
                f'{self._ds_key}'
            )
        if len(positions) > 1:
            raise ValueError(
                f'multiple {self._data_resolution.value} entries matching '
                f'{time} in {self._ds_key}'
            )
        return ds.isel(valid_time=int(positions[0]))

    def _require_data(self, time: pd.Timestamp):
        time = self._to_utc_naive(time)
        self._require_main_ds(time)

        if self._ds is not None and self._last_sel_time == time:
            return

        assert self._main_ds is not None

        if self._data_resolution == self._file_resolution:
            # Squeeze a length-1 valid_time if present; otherwise use as-is.
            if 'valid_time' in self._main_ds.dims:
                self._ds = self._main_ds.squeeze('valid_time', drop=True)
            else:
                self._ds = self._main_ds
        else:
            self._ds = self._select_by_components(self._main_ds, time)

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
        azimuth : float, optional
            Azimuth [degrees].
            If omitted, use the precomputed ground-track azimuth.

        Returns
        -------
        ground_speed: float
            Ground speed [m/s]
        """

        return self.get_track_vector(
            time, gt_point, altitude, true_airspeed, azimuth
        ).ground_speed

    def get_track_vector(
        self,
        time: pd.Timestamp,
        gt_point: GroundTrack.Point,
        altitude: float,
        horizontal_airspeed: float,
        track_azimuth: float | None = None,
    ) -> GroundTrackVector:
        """Compute ground speed and crabbed heading along a prescribed track."""
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

        if track_azimuth is None:
            track_azimuth = gt_point.azimuth
        return solve_track_vector(
            track_azimuth,
            horizontal_airspeed,
            wind_east=wind_u.item(),
            wind_north=wind_v.item(),
        )

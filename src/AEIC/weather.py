import gc
import math
from collections import OrderedDict
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
class _WindGrid:
    """Materialized wind grid for fast point interpolation.

    Coordinate axes are strictly ascending and the ``u``/``v`` arrays are
    oriented to match, so a plain ``searchsorted`` locates bracketing samples.
    """

    plev_asc: np.ndarray
    lat_asc: np.ndarray
    lon_asc: np.ndarray
    lon_max: float
    u: np.ndarray
    v: np.ndarray


# Process-wide cache of materialized wind grids, keyed by ``(file, selection)``.
# A fresh ``Weather`` instance is created for every simulated flight, but they
# all read the same on-disk weather; without a shared cache each instance would
# re-read and re-decode hundreds of MB per flight. The grids are read-only, so
# sharing them across instances is safe. ``maxsize`` is intentionally small
# because each entry can be hundreds of MB (e.g. a global ERA5 annual mean is
# ~0.6 GB for u+v); an LRU of 2 bounds memory while still serving the common
# single-file case with a permanent hit, and keeps at most one previous slice
# resident while iterating finer-resolution data.
_GRID_CACHE: OrderedDict[str, _WindGrid] = OrderedDict()
_GRID_CACHE_MAXSIZE = 2


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
        ``pressure_level``, ``latitude``, ``longitude``. Longitude follows
        ERA5's ``[0, 360)`` degrees-east convention. A ``valid_time`` coord is
        required when ``data_resolution`` is finer than
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

        # NumPy fast-path cache. The wind interpolation is called once per
        # trajectory point (tens of millions of times for a full run), so the
        # per-call overhead of ``xarray.DataArray.interp`` (which builds new
        # datasets, dispatches through ``apply_ufunc``, and calls SciPy) is
        # prohibitive. Instead we materialize the ``u``/``v`` grids and their
        # coordinate axes into plain NumPy arrays once per data selection and
        # interpolate points directly. The materialized grid is held in a
        # process-wide cache (``_GRID_CACHE``) so that the per-flight ``Weather``
        # instances share it; ``_grid`` is this instance's reference to the
        # currently-selected grid and ``_grid_key`` records which selection it
        # is for.
        self._grid: _WindGrid | None = None
        self._grid_key: str | None = None
        self._data_sel_key: str | None = None

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

    def _ensure_arrays(self) -> None:
        """Materialize the currently-selected ``u``/``v`` grids and coordinate
        axes into NumPy arrays for fast point interpolation.

        Populates ``self._grid`` from the process-wide ``_GRID_CACHE`` (or
        builds and inserts it on a miss). Axes are reoriented to be strictly
        ascending so a plain ``searchsorted`` locates bracketing samples; the
        data arrays are flipped to match. This mirrors the linear interpolation
        semantics of ``xarray.DataArray.interp`` used previously (verified to
        agree to ~1e-15)."""
        assert self._ds is not None
        assert self._data_sel_key is not None

        if self._grid is not None and self._grid_key == self._data_sel_key:
            return

        cached = _GRID_CACHE.get(self._data_sel_key)
        if cached is not None:
            # Refresh LRU recency.
            _GRID_CACHE.move_to_end(self._data_sel_key)
            self._grid = cached
            self._grid_key = self._data_sel_key
            return

        ds = self._ds
        plev = np.asarray(ds['pressure_level'].values, dtype=float)
        lat = np.asarray(ds['latitude'].values, dtype=float)
        lon = np.asarray(ds['longitude'].values, dtype=float)
        u = np.asarray(ds['u'].values, dtype=float)
        v = np.asarray(ds['v'].values, dtype=float)

        # Reorient pressure (axis 0) and latitude (axis 1) to ascending. ERA5
        # stores both descending; longitude is already ascending [0, 360).
        if plev.size > 1 and plev[0] > plev[-1]:
            plev = plev[::-1]
            u = u[::-1]
            v = v[::-1]
        if lat.size > 1 and lat[0] > lat[-1]:
            lat = lat[::-1]
            u = u[:, ::-1]
            v = v[:, ::-1]

        grid = _WindGrid(
            plev_asc=plev,
            lat_asc=lat,
            lon_asc=lon,
            # The maximum longitude in ERA5 coordinates is 360.0° minus the
            # grid spacing; points beyond it interpolate across the 360° = 0°
            # seam.
            lon_max=float(lon[-1]),
            u=np.ascontiguousarray(u),
            v=np.ascontiguousarray(v),
        )

        _GRID_CACHE[self._data_sel_key] = grid
        _GRID_CACHE.move_to_end(self._data_sel_key)
        while len(_GRID_CACHE) > _GRID_CACHE_MAXSIZE:
            _GRID_CACHE.popitem(last=False)

        self._grid = grid
        self._grid_key = self._data_sel_key

    @staticmethod
    def _axis_weights(vals_asc: np.ndarray, x: float) -> tuple[int, int, float] | None:
        """Locate the bracketing indices and linear weight for ``x`` on a
        strictly-ascending axis. Returns ``None`` if ``x`` is outside the axis
        range (non-extrapolating, matching ``xarray.interp``'s NaN fill)."""
        if x < vals_asc[0] or x > vals_asc[-1]:
            return None
        i1 = int(np.searchsorted(vals_asc, x, side='left'))
        if i1 == 0:
            # x == vals_asc[0] exactly.
            return (0, 0, 0.0)
        i0 = i1 - 1
        denom = vals_asc[i1] - vals_asc[i0]
        weight = 0.0 if denom == 0 else (x - vals_asc[i0]) / denom
        return (i0, i1, weight)

    def _lon_weights(self, era5_longitude: float) -> tuple[int, int, float] | None:
        """Longitude bracketing indices/weight, including wrap-around across
        the 360° = 0° seam."""
        assert self._grid is not None
        if era5_longitude <= self._grid.lon_max:
            return self._axis_weights(self._grid.lon_asc, era5_longitude)
        # "Wrap-around" interpolation between the last and first longitudes.
        weight = (era5_longitude - self._grid.lon_max) / (360.0 - self._grid.lon_max)
        return (self._grid.lon_asc.size - 1, 0, weight)

    @staticmethod
    def _trilinear(
        arr: np.ndarray,
        pw: tuple[int, int, float],
        yw: tuple[int, int, float],
        xw: tuple[int, int, float],
    ) -> float:
        """Trilinear interpolation as nested bilinear-in-(pressure, latitude)
        then linear-in-longitude, matching the original interpolation order."""
        pi0, pi1, pt = pw
        yi0, yi1, yt = yw
        xi0, xi1, xt = xw

        def bilinear(xi: int) -> float:
            c00 = arr[pi0, yi0, xi]
            c10 = arr[pi1, yi0, xi]
            c01 = arr[pi0, yi1, xi]
            c11 = arr[pi1, yi1, xi]
            return (c00 * (1.0 - pt) + c10 * pt) * (1.0 - yt) + (
                c01 * (1.0 - pt) + c11 * pt
            ) * yt

        return bilinear(xi0) * (1.0 - xt) + bilinear(xi1) * xt

    def _interp_wind(
        self,
        variable: str,
        pressure_level: float,
        latitude: float,
        era5_longitude: float,
    ) -> float:
        """Interpolate a wind component at a single point, including across
        360° = 0°.

        Longitude here is ERA5-style [0, 360] degrees east, not [-180, 180].
        Returns NaN if the point lies outside the grid domain."""
        self._ensure_arrays()
        assert self._grid is not None

        pw = self._axis_weights(self._grid.plev_asc, pressure_level)
        yw = self._axis_weights(self._grid.lat_asc, latitude)
        xw = self._lon_weights(era5_longitude)
        if pw is None or yw is None or xw is None:
            return math.nan

        arr = self._grid.u if variable == 'u' else self._grid.v
        return self._trilinear(arr, pw, yw, xw)

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

        # Build a process-wide cache key from the fully-resolved file path plus
        # a stat signature (size + mtime), so that distinct files that merely
        # share a basename (common in tests using tmp dirs) never collide and
        # an updated file invalidates the cache.
        path = self._nc_path(time)
        try:
            st = path.stat()
            file_sig = f'{path}|{st.st_size}|{st.st_mtime_ns}'
        except OSError:
            file_sig = str(path)

        if self._data_resolution == self._file_resolution:
            # Squeeze a length-1 valid_time if present; otherwise use as-is.
            if 'valid_time' in self._main_ds.dims:
                self._ds = self._main_ds.squeeze('valid_time', drop=True)
            else:
                self._ds = self._main_ds
            # One selection per file; independent of the query time.
            self._data_sel_key = f'{file_sig}|all'
        else:
            self._ds = self._select_by_components(self._main_ds, time)
            # Key on the actually-selected valid_time so that distinct query
            # times that resolve to the same data slice share one cached grid.
            self._data_sel_key = f'{file_sig}|{self._ds["valid_time"].values}'

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
        self._ensure_arrays()
        assert self._grid is not None

        # Ground track longitude ([-180, 180]) to ERA5 longitude ([0, 360]).
        longitude = gt_point.location.longitude % 360.0

        # NOTE: pressure levels in weather files are in hPa, not Pa.
        pressure_level = pressure_at_altitude_isa_bada4(altitude) / 100.0

        # Locate bracketing samples once and share them between the u and v
        # components (they live on the same grid).
        pw = self._axis_weights(self._grid.plev_asc, pressure_level)
        yw = self._axis_weights(self._grid.lat_asc, gt_point.location.latitude)
        xw = self._lon_weights(longitude)
        if pw is None or yw is None or xw is None:
            raise ValueError('ground track point is outside weather data domain')

        wind_u = self._trilinear(self._grid.u, pw, yw, xw)
        wind_v = self._trilinear(self._grid.v, pw, yw, xw)
        if math.isnan(wind_u) or math.isnan(wind_v):
            raise ValueError('ground track point is outside weather data domain')

        if track_azimuth is None:
            track_azimuth = gt_point.azimuth
        return solve_track_vector(
            track_azimuth,
            horizontal_airspeed,
            wind_east=wind_u,
            wind_north=wind_v,
        )

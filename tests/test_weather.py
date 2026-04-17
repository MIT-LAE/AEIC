from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from AEIC.config.weather import WeatherConfig, WeatherResolution
from AEIC.missions import Mission
from AEIC.missions.mission import iso_to_timestamp
from AEIC.trajectories.ground_track import GroundTrack
from AEIC.types import Location
from AEIC.weather import Weather

# Sample mission
sample_mission = Mission(
    origin='BOS',
    destination='ATL',
    aircraft_type='738',
    departure=iso_to_timestamp('2024-09-01 12:00:00'),
    arrival=iso_to_timestamp('2024-09-01 18:00:00'),
    load_factor=1.0,
)


@pytest.fixture
def ground_track():
    return GroundTrack.great_circle(
        sample_mission.origin_position.location,
        sample_mission.destination_position.location,
    )


@pytest.fixture
def sample_weather(test_data_dir):
    return Weather(data_dir=test_data_dir / 'weather')


def test_weather_init_with_bad_str():
    with pytest.raises(FileNotFoundError):
        Weather(data_dir='missing-directory')


def test_compute_ground_speed(sample_weather, ground_track):
    ground_distance_m = 370400.0
    altitude_m = 9144.0
    tas_ms = 200.0

    gs = sample_weather.get_ground_speed(
        time=sample_mission.departure,
        gt_point=ground_track.location(ground_distance_m),
        altitude=altitude_m,
        true_airspeed=tas_ms,
    )

    # NOTE: Relaxed tolerance because we changed the pressure level calculation
    # slightly.
    assert gs == pytest.approx(191.02855126751604, rel=1e-4)


# ---------------------------------------------------------------------------
# Synthetic NetCDF fixture helpers
# ---------------------------------------------------------------------------

_PRESSURE_LEVELS = np.array([200.0, 300.0, 500.0])
_LATITUDES = np.arange(35.0, 46.0, 2.0)  # 35, 37, ..., 45
_LONGITUDES = np.arange(-80.0, -69.0, 2.0)  # -80, -78, ..., -70

# Test probe — inside the grid, uses pressure level 300 hPa exactly.
_PROBE_LOCATION = Location(longitude=-75.0, latitude=40.0)
_PROBE_POINT = GroundTrack.Point(location=_PROBE_LOCATION, azimuth=0.0)
_PROBE_ALT = 9144.0  # ~300 hPa by ISA
_PROBE_TAS = 200.0
# Constants in synthetic fields; with azimuth=0, cos=1 sin=0 →
# u_air=200, v_air=0, so ground speed = hypot(200 + 5, 0 + 0) = 205.
_WIND_U = 5.0
_WIND_V = 0.0
_EXPECTED_GS = 205.0


def _make_field(shape: tuple[int, ...], value: float) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def _base_dataset(valid_time: pd.DatetimeIndex | None = None) -> xr.Dataset:
    """Build a small synthetic ERA5-like Dataset.

    If ``valid_time`` is None, the dataset has no time dimension. If it is a
    length-1 or longer DatetimeIndex, a ``valid_time`` dim is included.
    """
    coords: dict[str, object] = {
        'pressure_level': _PRESSURE_LEVELS,
        'latitude': _LATITUDES,
        'longitude': _LONGITUDES,
    }
    dims = ('pressure_level', 'latitude', 'longitude')
    shape = (len(_PRESSURE_LEVELS), len(_LATITUDES), len(_LONGITUDES))

    if valid_time is not None:
        coords['valid_time'] = valid_time
        dims = ('valid_time', *dims)
        shape = (len(valid_time), *shape)

    return xr.Dataset(
        {
            'u': (dims, _make_field(shape, _WIND_U)),
            'v': (dims, _make_field(shape, _WIND_V)),
            't': (dims, _make_field(shape, 220.0)),
        },
        coords=coords,
    )


def _write_mean_file(path: Path, *, with_valid_time: bool) -> None:
    if with_valid_time:
        ds = _base_dataset(pd.DatetimeIndex([pd.Timestamp('2024-01-01')]))
    else:
        ds = _base_dataset()
    ds.to_netcdf(path)


def _write_hourly_file(path: Path, start: pd.Timestamp, hours: int) -> None:
    vt = pd.date_range(start=start, periods=hours, freq='h')
    ds = _base_dataset(vt)
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# WeatherConfig validation tests
# ---------------------------------------------------------------------------


def test_weather_resolution_enum_case_insensitive():
    assert (
        WeatherResolution('Hourly_Daily_Files') is WeatherResolution.HOURLY_DAILY_FILES
    )
    assert WeatherResolution('ANNUAL_MEAN') is WeatherResolution.ANNUAL_MEAN


def test_default_weather_config_validates():
    cfg = WeatherConfig()
    assert cfg.resolution is WeatherResolution.HOURLY_DAILY_FILES
    assert cfg.file_format == '%Y%m%d.nc'


def test_file_format_rejects_unknown_token():
    with pytest.raises(ValueError, match='unsupported strftime token'):
        WeatherConfig(
            resolution=WeatherResolution.HOURLY_DAILY_FILES,
            file_format='%Y-%M-%d.nc',  # %M = minute
        )


def test_file_format_rejects_tz_token():
    with pytest.raises(ValueError, match='unsupported strftime token'):
        WeatherConfig(
            resolution=WeatherResolution.DAILY_MEAN,
            file_format='%Y%m%d%z.nc',
        )


@pytest.mark.parametrize(
    'resolution,file_format,should_pass',
    [
        # ANNUAL_MEAN
        (WeatherResolution.ANNUAL_MEAN, 'annual.nc', True),
        (WeatherResolution.ANNUAL_MEAN, '%Y.nc', True),
        (WeatherResolution.ANNUAL_MEAN, '%Y-%m.nc', False),
        # MONTHLY_MEAN
        (WeatherResolution.MONTHLY_MEAN, '%Y-%m.nc', True),
        (WeatherResolution.MONTHLY_MEAN, '%Y.nc', False),
        (WeatherResolution.MONTHLY_MEAN, '%Y-%m-%d.nc', False),
        (WeatherResolution.MONTHLY_MEAN, 'constant.nc', False),
        # DAILY_MEAN
        (WeatherResolution.DAILY_MEAN, '%Y%m%d.nc', True),
        (WeatherResolution.DAILY_MEAN, '%Y-%j.nc', True),
        (WeatherResolution.DAILY_MEAN, '%Y-%m.nc', False),
        (WeatherResolution.DAILY_MEAN, '%Y%m%d%H.nc', False),
        # HOURLY_DAILY_FILES
        (WeatherResolution.HOURLY_DAILY_FILES, '%Y%m%d.nc', True),
        (WeatherResolution.HOURLY_DAILY_FILES, '%Y-%j.nc', True),
        (WeatherResolution.HOURLY_DAILY_FILES, '%Y%m%d%H.nc', False),
        # HOURLY_MONTHLY_FILES
        (WeatherResolution.HOURLY_MONTHLY_FILES, '%Y-%m.nc', True),
        (WeatherResolution.HOURLY_MONTHLY_FILES, '%Y-%m-%d.nc', False),
        (WeatherResolution.HOURLY_MONTHLY_FILES, 'constant.nc', False),
    ],
)
def test_format_resolution_coupling(resolution, file_format, should_pass):
    if should_pass:
        cfg = WeatherConfig(resolution=resolution, file_format=file_format)
        assert cfg.resolution is resolution
        assert cfg.file_format == file_format
    else:
        with pytest.raises(ValueError):
            WeatherConfig(resolution=resolution, file_format=file_format)


# ---------------------------------------------------------------------------
# Per-resolution integration tests
# ---------------------------------------------------------------------------


def _run_probe(w: Weather, time: pd.Timestamp) -> float:
    return w.get_ground_speed(
        time=time,
        gt_point=_PROBE_POINT,
        altitude=_PROBE_ALT,
        true_airspeed=_PROBE_TAS,
    )


def test_annual_mean_reads_single_file(tmp_path):
    _write_mean_file(tmp_path / 'annual.nc', with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.ANNUAL_MEAN,
        file_format='annual.nc',
    )
    for t in [
        pd.Timestamp('2024-01-01T00:00'),
        pd.Timestamp('2024-06-15T14:30'),
        pd.Timestamp('2024-12-31T23:00'),
    ]:
        assert _run_probe(w, t) == pytest.approx(_EXPECTED_GS, rel=1e-4)


def test_annual_mean_with_length_one_valid_time_is_squeezed(tmp_path):
    _write_mean_file(tmp_path / 'annual.nc', with_valid_time=True)
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.ANNUAL_MEAN,
        file_format='annual.nc',
    )
    assert _run_probe(w, pd.Timestamp('2024-06-15')) == pytest.approx(
        _EXPECTED_GS, rel=1e-4
    )


def test_monthly_mean_switches_files_on_month_boundary(tmp_path):
    for month in (1, 2, 3):
        _write_mean_file(tmp_path / f'2024-{month:02d}.nc', with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.MONTHLY_MEAN,
        file_format='%Y-%m.nc',
    )
    assert _run_probe(w, pd.Timestamp('2024-01-15')) == pytest.approx(
        _EXPECTED_GS, rel=1e-4
    )
    # Different month → different file.
    assert _run_probe(w, pd.Timestamp('2024-02-15')) == pytest.approx(
        _EXPECTED_GS, rel=1e-4
    )


def test_daily_mean_with_doy_format(tmp_path):
    for doy in (1, 2):
        date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=doy - 1)
        _write_mean_file(tmp_path / date.strftime('%Y-%j.nc'), with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.DAILY_MEAN,
        file_format='%Y-%j.nc',
    )
    assert _run_probe(w, pd.Timestamp('2024-01-02T06:00')) == pytest.approx(
        _EXPECTED_GS, rel=1e-4
    )


def test_hourly_daily_files_sel_by_value(tmp_path):
    _write_hourly_file(
        tmp_path / '20240601.nc',
        start=pd.Timestamp('2024-06-01T00:00'),
        hours=24,
    )
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.HOURLY_DAILY_FILES,
        file_format='%Y%m%d.nc',
    )
    for hour in (0, 7, 23):
        t = pd.Timestamp(f'2024-06-01T{hour:02d}:00')
        assert _run_probe(w, t) == pytest.approx(_EXPECTED_GS, rel=1e-4)


def test_hourly_monthly_files(tmp_path):
    # Write a file for 2024-06: 720 hourly entries.
    _write_hourly_file(
        tmp_path / '2024-06.nc',
        start=pd.Timestamp('2024-06-01T00:00'),
        hours=24 * 30,
    )
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.HOURLY_MONTHLY_FILES,
        file_format='%Y-%m.nc',
    )
    # Probe at multiple days within the month — all land in the same file.
    for day in (1, 15, 30):
        t = pd.Timestamp(f'2024-06-{day:02d}T12:00')
        assert _run_probe(w, t) == pytest.approx(_EXPECTED_GS, rel=1e-4)


# ---------------------------------------------------------------------------
# Caching / reopen behavior
# ---------------------------------------------------------------------------


def test_annual_file_opens_once(tmp_path):
    _write_mean_file(tmp_path / 'annual.nc', with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.ANNUAL_MEAN,
        file_format='annual.nc',
    )
    with patch('AEIC.weather.xr.open_dataset', wraps=xr.open_dataset) as spy:
        _run_probe(w, pd.Timestamp('2024-01-01'))
        _run_probe(w, pd.Timestamp('2024-12-31T23:00'))
        assert spy.call_count == 1


def test_daily_mean_reopens_on_midnight(tmp_path):
    for day in (1, 2):
        date = pd.Timestamp(f'2024-01-0{day}')
        _write_mean_file(tmp_path / date.strftime('%Y%m%d.nc'), with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.DAILY_MEAN,
        file_format='%Y%m%d.nc',
    )
    with patch('AEIC.weather.xr.open_dataset', wraps=xr.open_dataset) as spy:
        _run_probe(w, pd.Timestamp('2024-01-01T23:30'))
        _run_probe(w, pd.Timestamp('2024-01-02T00:30'))
        assert spy.call_count == 2


def test_hourly_monthly_files_no_reopen_within_month(tmp_path):
    _write_hourly_file(
        tmp_path / '2024-06.nc',
        start=pd.Timestamp('2024-06-01T00:00'),
        hours=24 * 30,
    )
    _write_hourly_file(
        tmp_path / '2024-07.nc',
        start=pd.Timestamp('2024-07-01T00:00'),
        hours=24 * 31,
    )
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.HOURLY_MONTHLY_FILES,
        file_format='%Y-%m.nc',
    )
    with patch('AEIC.weather.xr.open_dataset', wraps=xr.open_dataset) as spy:
        _run_probe(w, pd.Timestamp('2024-06-01T00:00'))
        _run_probe(w, pd.Timestamp('2024-06-15T12:00'))
        _run_probe(w, pd.Timestamp('2024-06-30T23:00'))
        assert spy.call_count == 1
        # Month change → reopen.
        _run_probe(w, pd.Timestamp('2024-07-01T01:00'))
        assert spy.call_count == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_hourly_sel_miss_raises(tmp_path):
    # Sparse file — only hour 0 present. Query at hour 12 → beyond 1h tolerance.
    _write_hourly_file(
        tmp_path / '20240101.nc',
        start=pd.Timestamp('2024-01-01T00:00'),
        hours=1,
    )
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.HOURLY_DAILY_FILES,
        file_format='%Y%m%d.nc',
    )
    with pytest.raises(KeyError):
        _run_probe(w, pd.Timestamp('2024-01-01T12:00'))


def test_tz_aware_timestamp_coerced_to_utc(tmp_path):
    # File for 2024-09-02 UTC.
    _write_mean_file(tmp_path / '20240902.nc', with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.DAILY_MEAN,
        file_format='%Y%m%d.nc',
    )
    # 22:00 in New York on 2024-09-01 = 02:00 UTC on 2024-09-02.
    t_local = pd.Timestamp('2024-09-01T22:00', tz='America/New_York')
    assert _run_probe(w, t_local) == pytest.approx(_EXPECTED_GS, rel=1e-4)


def test_non_datetime_valid_time_rejected(tmp_path):
    # Build a file with an integer valid_time coord.
    shape = (3, len(_PRESSURE_LEVELS), len(_LATITUDES), len(_LONGITUDES))
    ds = xr.Dataset(
        {
            'u': (
                ('valid_time', 'pressure_level', 'latitude', 'longitude'),
                _make_field(shape, _WIND_U),
            ),
            'v': (
                ('valid_time', 'pressure_level', 'latitude', 'longitude'),
                _make_field(shape, _WIND_V),
            ),
            't': (
                ('valid_time', 'pressure_level', 'latitude', 'longitude'),
                _make_field(shape, 220.0),
            ),
        },
        coords={
            'valid_time': np.array([0, 1, 2], dtype=np.int64),
            'pressure_level': _PRESSURE_LEVELS,
            'latitude': _LATITUDES,
            'longitude': _LONGITUDES,
        },
    )
    ds.to_netcdf(tmp_path / '20240101.nc')
    w = Weather(
        data_dir=tmp_path,
        resolution=WeatherResolution.HOURLY_DAILY_FILES,
        file_format='%Y%m%d.nc',
    )
    with pytest.raises(TypeError, match='datetime64'):
        _run_probe(w, pd.Timestamp('2024-01-01T00:00'))

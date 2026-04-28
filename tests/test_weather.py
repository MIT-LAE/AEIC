from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from AEIC.config.weather import TemporalResolution, WeatherConfig
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
    # On-disk fixture is a single time slice (no valid_time dim) -- a daily
    # mean -- so file_resolution=daily and data_resolution defaults to daily.
    return Weather(
        data_dir=test_data_dir / 'weather',
        file_resolution=TemporalResolution.DAILY,
    )


def test_weather_init_with_bad_str():
    with pytest.raises(FileNotFoundError):
        Weather(
            data_dir='missing-directory',
            file_resolution=TemporalResolution.DAILY,
        )


@pytest.mark.forked
def test_compute_ground_speed(sample_weather, ground_track):
    # Integration smoke test against the real ERA5 fixture (2024-09-01.nc). The
    # algorithm is exhaustively covered by the synthetic-fixture tests below
    # — which assert exact expected values derivable on paper — so this test
    # exists only to confirm the on-disk fixture still parses via Weather
    # end-to-end and yields a physically plausible answer. No specific
    # expected value is asserted: any single number we could write here would
    # be derivable only by running the SUT against the same fixture.
    gs = sample_weather.get_ground_speed(
        time=sample_mission.departure,
        gt_point=ground_track.location(370400.0),
        altitude=9144.0,
        true_airspeed=200.0,
    )
    # TAS=200 m/s; ERA5 upper-level winds are bounded ~|100| m/s typical, so
    # any ground speed outside this envelope indicates a pipeline regression
    # or fixture corruption, not a subtle science bug.
    assert np.isfinite(gs)
    assert 100.0 < gs < 300.0


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


def _write_multi_file(path: Path, valid_time: pd.DatetimeIndex) -> None:
    ds = _base_dataset(valid_time)
    ds.to_netcdf(path)


def _write_hourly_file(path: Path, start: pd.Timestamp, hours: int) -> None:
    vt = pd.date_range(start=start, periods=hours, freq='h')
    _write_multi_file(path, vt)


# ---------------------------------------------------------------------------
# WeatherConfig validation tests
# ---------------------------------------------------------------------------


def test_temporal_resolution_enum_case_insensitive():
    assert TemporalResolution('Hourly') is TemporalResolution.HOURLY
    assert TemporalResolution('ANNUAL') is TemporalResolution.ANNUAL
    assert TemporalResolution('Daily') is TemporalResolution.DAILY


def test_default_weather_config_validates():
    cfg = WeatherConfig()
    assert cfg.file_resolution is TemporalResolution.DAILY
    assert cfg.data_resolution is None
    assert cfg.effective_data_resolution is TemporalResolution.DAILY
    assert cfg.file_format is None
    assert cfg.effective_file_format == '%Y-%m-%d.nc'


def test_file_resolution_hourly_rejected():
    with pytest.raises(ValueError, match='[Pp]er-hour files'):
        WeatherConfig(file_resolution=TemporalResolution.HOURLY)


@pytest.mark.parametrize(
    'file_res,data_res,should_pass',
    [
        (TemporalResolution.ANNUAL, TemporalResolution.ANNUAL, True),
        (TemporalResolution.ANNUAL, TemporalResolution.MONTHLY, True),
        (TemporalResolution.ANNUAL, TemporalResolution.DAILY, True),
        (TemporalResolution.ANNUAL, TemporalResolution.HOURLY, True),
        (TemporalResolution.MONTHLY, TemporalResolution.MONTHLY, True),
        (TemporalResolution.MONTHLY, TemporalResolution.DAILY, True),
        (TemporalResolution.MONTHLY, TemporalResolution.HOURLY, True),
        (TemporalResolution.DAILY, TemporalResolution.DAILY, True),
        (TemporalResolution.DAILY, TemporalResolution.HOURLY, True),
        # Invalid: data coarser than file.
        (TemporalResolution.DAILY, TemporalResolution.MONTHLY, False),
        (TemporalResolution.DAILY, TemporalResolution.ANNUAL, False),
        (TemporalResolution.MONTHLY, TemporalResolution.ANNUAL, False),
    ],
)
def test_data_le_file_validation(file_res, data_res, should_pass):
    if should_pass:
        cfg = WeatherConfig(file_resolution=file_res, data_resolution=data_res)
        assert cfg.effective_data_resolution is data_res
    else:
        with pytest.raises(ValueError, match='finer-or-equal'):
            WeatherConfig(file_resolution=file_res, data_resolution=data_res)


def test_data_resolution_defaults_to_file_resolution():
    cfg = WeatherConfig(file_resolution=TemporalResolution.MONTHLY)
    assert cfg.data_resolution is None
    assert cfg.effective_data_resolution is TemporalResolution.MONTHLY


def test_file_format_defaults_per_resolution():
    assert (
        WeatherConfig(file_resolution=TemporalResolution.ANNUAL).effective_file_format
        == '%Y.nc'
    )
    assert (
        WeatherConfig(file_resolution=TemporalResolution.MONTHLY).effective_file_format
        == '%Y-%m.nc'
    )
    assert (
        WeatherConfig(file_resolution=TemporalResolution.DAILY).effective_file_format
        == '%Y-%m-%d.nc'
    )


def test_file_format_rejects_unknown_token():
    with pytest.raises(ValueError, match='unsupported strftime token'):
        WeatherConfig(
            file_resolution=TemporalResolution.DAILY,
            file_format='%Y-%M-%d.nc',  # %M = minute
        )


def test_file_format_rejects_hourly_token():
    # %H is never allowed since file_resolution can't be hourly.
    with pytest.raises(ValueError, match='unsupported strftime token'):
        WeatherConfig(
            file_resolution=TemporalResolution.DAILY,
            file_format='%Y%m%d%H.nc',
        )


def test_file_format_rejects_tz_token():
    with pytest.raises(ValueError, match='unsupported strftime token'):
        WeatherConfig(
            file_resolution=TemporalResolution.DAILY,
            file_format='%Y%m%d%z.nc',
        )


@pytest.mark.parametrize(
    'file_resolution,file_format,should_pass',
    [
        # ANNUAL
        (TemporalResolution.ANNUAL, 'annual.nc', True),
        (TemporalResolution.ANNUAL, '%Y.nc', True),
        (TemporalResolution.ANNUAL, '%Y-%m.nc', False),
        # MONTHLY
        (TemporalResolution.MONTHLY, '%Y-%m.nc', True),
        (TemporalResolution.MONTHLY, '%Y.nc', False),
        (TemporalResolution.MONTHLY, '%Y-%m-%d.nc', False),
        (TemporalResolution.MONTHLY, 'constant.nc', False),
        # DAILY
        (TemporalResolution.DAILY, '%Y%m%d.nc', True),
        (TemporalResolution.DAILY, '%Y-%j.nc', True),
        (TemporalResolution.DAILY, '%Y-%m.nc', False),
        (TemporalResolution.DAILY, 'constant.nc', False),
    ],
)
def test_format_resolution_coupling(file_resolution, file_format, should_pass):
    if should_pass:
        cfg = WeatherConfig(file_resolution=file_resolution, file_format=file_format)
        assert cfg.file_resolution is file_resolution
        assert cfg.file_format == file_format
    else:
        with pytest.raises(ValueError):
            WeatherConfig(file_resolution=file_resolution, file_format=file_format)


# ---------------------------------------------------------------------------
# Per-resolution integration tests (data == file: squeeze)
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
        file_resolution=TemporalResolution.ANNUAL,
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
        file_resolution=TemporalResolution.ANNUAL,
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
        file_resolution=TemporalResolution.MONTHLY,
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
        file_resolution=TemporalResolution.DAILY,
        file_format='%Y-%j.nc',
    )
    assert _run_probe(w, pd.Timestamp('2024-01-02T06:00')) == pytest.approx(
        _EXPECTED_GS, rel=1e-4
    )


# ---------------------------------------------------------------------------
# Per-resolution integration tests (data < file: select by date components)
# ---------------------------------------------------------------------------


def test_hourly_data_in_daily_files(tmp_path):
    _write_hourly_file(
        tmp_path / '20240601.nc',
        start=pd.Timestamp('2024-06-01T00:00'),
        hours=24,
    )
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.DAILY,
        data_resolution=TemporalResolution.HOURLY,
        file_format='%Y%m%d.nc',
    )
    for hour in (0, 7, 23):
        t = pd.Timestamp(f'2024-06-01T{hour:02d}:00')
        assert _run_probe(w, t) == pytest.approx(_EXPECTED_GS, rel=1e-4)


def test_hourly_data_in_monthly_files(tmp_path):
    _write_hourly_file(
        tmp_path / '2024-06.nc',
        start=pd.Timestamp('2024-06-01T00:00'),
        hours=24 * 30,
    )
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.MONTHLY,
        data_resolution=TemporalResolution.HOURLY,
    )
    for day in (1, 15, 30):
        t = pd.Timestamp(f'2024-06-{day:02d}T12:00')
        assert _run_probe(w, t) == pytest.approx(_EXPECTED_GS, rel=1e-4)


def test_daily_data_in_monthly_files(tmp_path):
    vt = pd.date_range(start='2024-06-01', periods=30, freq='D')
    _write_multi_file(tmp_path / '2024-06.nc', vt)
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.MONTHLY,
        data_resolution=TemporalResolution.DAILY,
    )
    for day in (1, 15, 30):
        # Query at any time within the day; should pick that day's entry.
        t = pd.Timestamp(f'2024-06-{day:02d}T18:30')
        assert _run_probe(w, t) == pytest.approx(_EXPECTED_GS, rel=1e-4)


def test_monthly_data_in_annual_file(tmp_path):
    # Mid-month timestamps (the failure mode that motivated round-then-exact).
    vt = pd.DatetimeIndex([pd.Timestamp(f'2024-{m:02d}-15') for m in range(1, 13)])
    _write_multi_file(tmp_path / '2024.nc', vt)
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.ANNUAL,
        data_resolution=TemporalResolution.MONTHLY,
    )
    # Each query inside its month (including end-of-month, which would be
    # closer to the next month's entry under nearest+tolerance).
    for month in (1, 6, 12):
        t = pd.Timestamp(f'2024-{month:02d}-28T23:00')
        assert _run_probe(w, t) == pytest.approx(_EXPECTED_GS, rel=1e-4)


def test_monthly_in_annual_picks_correct_month_at_period_boundary(tmp_path):
    # File entries are at mid-month (15th). Query at 2024-03-31:
    #   distance to 2024-03-15 = 16 days
    #   distance to 2024-04-15 = 15 days
    # nearest+tolerance would silently pick April. Round-then-exact picks March.
    # Verify by writing distinguishable values per month and checking we get the
    # March slice.
    pl_count = len(_PRESSURE_LEVELS)
    lat_count = len(_LATITUDES)
    lon_count = len(_LONGITUDES)

    # Per-month wind_u: month 1 → 1.0, month 2 → 2.0, ..., month 12 → 12.0.
    months = list(range(1, 13))
    vt = pd.DatetimeIndex([pd.Timestamp(f'2024-{m:02d}-15') for m in months])
    u_vals = np.zeros((12, pl_count, lat_count, lon_count), dtype=np.float32)
    for i, m in enumerate(months):
        u_vals[i, ...] = float(m)
    v_vals = np.zeros_like(u_vals)
    t_vals = np.full_like(u_vals, 220.0)

    ds = xr.Dataset(
        {
            'u': (
                ('valid_time', 'pressure_level', 'latitude', 'longitude'),
                u_vals,
            ),
            'v': (
                ('valid_time', 'pressure_level', 'latitude', 'longitude'),
                v_vals,
            ),
            't': (
                ('valid_time', 'pressure_level', 'latitude', 'longitude'),
                t_vals,
            ),
        },
        coords={
            'valid_time': vt,
            'pressure_level': _PRESSURE_LEVELS,
            'latitude': _LATITUDES,
            'longitude': _LONGITUDES,
        },
    )
    ds.to_netcdf(tmp_path / '2024.nc')

    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.ANNUAL,
        data_resolution=TemporalResolution.MONTHLY,
    )
    # Query on 2024-03-31. Expected u_air = 200 (TAS, azimuth=0); wind_u for
    # March is 3.0; ground speed = hypot(203, 0) = 203.
    gs = _run_probe(w, pd.Timestamp('2024-03-31T23:00'))
    assert gs == pytest.approx(203.0, rel=1e-4)
    # Sanity check: querying in April returns wind_u=4 → gs=204.
    gs = _run_probe(w, pd.Timestamp('2024-04-15T00:00'))
    assert gs == pytest.approx(204.0, rel=1e-4)


def test_daily_data_in_annual_file(tmp_path):
    vt = pd.date_range(start='2024-01-01', periods=10, freq='D')
    _write_multi_file(tmp_path / '2024.nc', vt)
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.ANNUAL,
        data_resolution=TemporalResolution.DAILY,
    )
    for day in (1, 5, 10):
        t = pd.Timestamp(f'2024-01-{day:02d}T12:00')
        assert _run_probe(w, t) == pytest.approx(_EXPECTED_GS, rel=1e-4)


def test_hourly_data_in_annual_file(tmp_path):
    # Just write a small slice of an annual hourly file: 48 hours.
    _write_hourly_file(
        tmp_path / '2024.nc',
        start=pd.Timestamp('2024-01-01T00:00'),
        hours=48,
    )
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.ANNUAL,
        data_resolution=TemporalResolution.HOURLY,
    )
    for t in [
        pd.Timestamp('2024-01-01T00:00'),
        pd.Timestamp('2024-01-01T17:00'),
        pd.Timestamp('2024-01-02T05:00'),
    ]:
        assert _run_probe(w, t) == pytest.approx(_EXPECTED_GS, rel=1e-4)


# ---------------------------------------------------------------------------
# L1 file-content validation
# ---------------------------------------------------------------------------


def test_l1_squeeze_case_rejects_multi_entry_file(tmp_path):
    # Config says daily mean (data == file == DAILY) but file has 24 entries.
    _write_hourly_file(
        tmp_path / '2024-06-01.nc',
        start=pd.Timestamp('2024-06-01T00:00'),
        hours=24,
    )
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.DAILY,
    )
    with pytest.raises(ValueError, match='expected 0 or 1 valid_time'):
        _run_probe(w, pd.Timestamp('2024-06-01T12:00'))


def test_l1_multi_case_rejects_missing_valid_time(tmp_path):
    # Config says hourly_in_daily but file has no valid_time dim.
    _write_mean_file(tmp_path / '2024-06-01.nc', with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.DAILY,
        data_resolution=TemporalResolution.HOURLY,
    )
    with pytest.raises(ValueError, match='no valid_time dim'):
        _run_probe(w, pd.Timestamp('2024-06-01T12:00'))


def test_l1_multi_case_rejects_single_entry(tmp_path):
    # Config says hourly_in_daily but file has only 1 valid_time entry.
    _write_hourly_file(
        tmp_path / '2024-06-01.nc',
        start=pd.Timestamp('2024-06-01T00:00'),
        hours=1,
    )
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.DAILY,
        data_resolution=TemporalResolution.HOURLY,
    )
    with pytest.raises(ValueError, match='but file has 1'):
        _run_probe(w, pd.Timestamp('2024-06-01T00:00'))


# ---------------------------------------------------------------------------
# Caching / reopen behavior
# ---------------------------------------------------------------------------


def test_annual_file_opens_once(tmp_path):
    _write_mean_file(tmp_path / 'annual.nc', with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.ANNUAL,
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
        file_resolution=TemporalResolution.DAILY,
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
        file_resolution=TemporalResolution.MONTHLY,
        data_resolution=TemporalResolution.HOURLY,
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
    # File has hours 0 and 1; query at hour 5 → no matching hour → KeyError.
    _write_hourly_file(
        tmp_path / '20240101.nc',
        start=pd.Timestamp('2024-01-01T00:00'),
        hours=2,
    )
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.DAILY,
        data_resolution=TemporalResolution.HOURLY,
        file_format='%Y%m%d.nc',
    )
    with pytest.raises(KeyError):
        _run_probe(w, pd.Timestamp('2024-01-01T05:00'))


def test_tz_aware_timestamp_coerced_to_utc(tmp_path):
    # File for 2024-09-02 UTC.
    _write_mean_file(tmp_path / '20240902.nc', with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.DAILY,
        file_format='%Y%m%d.nc',
    )
    # 22:00 in New York on 2024-09-01 = 02:00 UTC on 2024-09-02.
    t_local = pd.Timestamp('2024-09-01T22:00', tz='America/New_York')
    assert _run_probe(w, t_local) == pytest.approx(_EXPECTED_GS, rel=1e-4)


def test_explicit_azimuth_overrides_ground_track_azimuth(tmp_path):
    """Explicit `azimuth` argument must override the precomputed
    `gt_point.azimuth` (auto vs. explicit diverge in weather.py:263–266).
    """
    _write_mean_file(tmp_path / 'annual.nc', with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.ANNUAL,
        file_format='annual.nc',
    )
    # azimuth=0 (gt_point default) → u_air=TAS, gs = hypot(TAS+wind_u, 0).
    # azimuth=90 (east) → v_air=TAS, gs = hypot(wind_u, TAS).
    gs_auto = w.get_ground_speed(
        time=pd.Timestamp('2024-06-15'),
        gt_point=_PROBE_POINT,
        altitude=_PROBE_ALT,
        true_airspeed=_PROBE_TAS,
    )
    gs_east = w.get_ground_speed(
        time=pd.Timestamp('2024-06-15'),
        gt_point=_PROBE_POINT,
        altitude=_PROBE_ALT,
        true_airspeed=_PROBE_TAS,
        azimuth=90.0,
    )
    assert gs_auto == pytest.approx(_EXPECTED_GS, rel=1e-4)
    assert gs_east == pytest.approx(float(np.hypot(_WIND_U, _PROBE_TAS)), rel=1e-4)
    assert gs_auto != gs_east


def test_out_of_domain_raises(tmp_path):
    """A ground track point outside the weather grid must raise
    ValueError (`weather.py:261`). Covered indirectly via
    `test_trajectory_simulation_outside_weather_domain`, but the
    natural place for unit coverage is here.
    """
    _write_mean_file(tmp_path / 'annual.nc', with_valid_time=False)
    w = Weather(
        data_dir=tmp_path,
        file_resolution=TemporalResolution.ANNUAL,
        file_format='annual.nc',
    )
    far_point = GroundTrack.Point(
        location=Location(longitude=0.0, latitude=0.0),
        azimuth=0.0,
    )
    with pytest.raises(ValueError, match='outside weather data domain'):
        w.get_ground_speed(
            time=pd.Timestamp('2024-06-15'),
            gt_point=far_point,
            altitude=_PROBE_ALT,
            true_airspeed=_PROBE_TAS,
        )


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
        file_resolution=TemporalResolution.DAILY,
        data_resolution=TemporalResolution.HOURLY,
        file_format='%Y%m%d.nc',
    )
    with pytest.raises(TypeError, match='datetime64'):
        _run_probe(w, pd.Timestamp('2024-01-01T00:00'))

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from AEIC.missions import Mission
from AEIC.trajectories.ground_track import GroundTrack
from AEIC.utils.files import file_location
from AEIC.utils.helpers import iso_to_timestamp
from AEIC.utils.units import FEET_TO_METERS, NAUTICAL_MILES_TO_METERS
from AEIC.weather.weather import Weather

# Sample mission
sample_mission = Mission(
    origin='BOS',
    destination='ATL',
    aircraft_type='738',
    departure=iso_to_timestamp('2019-01-01 12:00:00'),
    arrival=iso_to_timestamp('2019-01-01 18:00:00'),
    load_factor=1.0,
)


@pytest.fixture(scope='session')
def ground_track():
    return GroundTrack.great_circle(
        sample_mission.origin_position.location,
        sample_mission.destination_position.location,
    )


@pytest.fixture(scope='session')
def weather_dataset_path():
    return file_location("weather/sample_weather_subset.nc")


@pytest.fixture(scope='session')
def sample_weather(ground_track, weather_dataset_path):
    return Weather(
        ds=str(weather_dataset_path),
        mission=sample_mission,
        ground_track=ground_track,
        fl_min=10,
        fl_max=300,
        fl_spacing=10,
    )


def test_weather_init_without_dataset_requires_weather_data(ground_track):
    with pytest.raises(ValueError, match='Weather dataset must be provided'):
        Weather(ds=None, mission=sample_mission, ground_track=ground_track)


def test_weather_initializes_regridded_maps(sample_weather):
    assert sample_weather.valid_time_index == sample_mission.departure.hour
    assert sample_weather.total_nm == pytest.approx(821.9662304583491, rel=1e-9)

    assert sample_weather.u_regridded is not None
    assert sample_weather.v_regridded is not None

    u_map = sample_weather.u_regridded
    assert {'Flight_Level', 'NM_0.0', 'NM_822.0'}.issubset(u_map.columns)

    # Native grid spacing should give us intermediate columns, not just endpoints.
    dist_cols = [c for c in u_map.columns if c.startswith('NM_')]
    assert len(dist_cols) > 2

    start_u = u_map.loc[u_map['Flight_Level'] == 10, 'NM_0.0'].iloc[0]
    assert start_u == pytest.approx(5.73948156392396, rel=1e-6)


def test_horizontal_spacing_follows_native_grid(sample_weather):
    spacing_m = sample_weather._native_grid_spacing_m()
    assert spacing_m is not None and spacing_m > 0

    spacing_nm = spacing_m / NAUTICAL_MILES_TO_METERS
    deltas = np.diff(sample_weather.arc_info['distances_nm'])
    assert np.all(deltas > 0)

    # Ignore last segment; it can be shorter when the track length
    # is not an exact multiple of the grid spacing.
    typical_spacing = np.median(deltas[:-1] if len(deltas) > 1 else deltas)
    assert typical_spacing == pytest.approx(spacing_nm, rel=0.05)


def test_compute_ground_speed(sample_weather):
    ground_distance_m = 200.0 * NAUTICAL_MILES_TO_METERS
    altitude_m = 30000 * FEET_TO_METERS
    tas_ms = 200.0

    gs, heading, wind_u, wind_v = sample_weather.compute_ground_speed(
        ground_distance_m=ground_distance_m,
        altitude_m=altitude_m,
        tas_ms=tas_ms,
    )

    nm = ground_distance_m / NAUTICAL_MILES_TO_METERS
    expected_heading = float(sample_weather.get_heading_from_gd(nm))
    alt_ft = altitude_m / FEET_TO_METERS
    fl = float(
        sample_weather._validate_altitudes(
            np.array([alt_ft]), context='test_compute_ground_speed'
        )[0]
    )

    expected_u = sample_weather._interpolate_value(sample_weather.u_regridded, nm, fl)
    expected_v = sample_weather._interpolate_value(sample_weather.v_regridded, nm, fl)

    heading_rad = np.deg2rad(expected_heading)
    u_air = tas_ms * np.cos(heading_rad)
    v_air = tas_ms * np.sin(heading_rad)
    expected_gs = np.hypot(u_air + expected_u, v_air + expected_v)

    assert heading == pytest.approx(expected_heading)
    assert wind_u == pytest.approx(expected_u)
    assert wind_v == pytest.approx(expected_v)
    assert gs == pytest.approx(expected_gs)

    assert gs == pytest.approx(190.18250530837227, rel=1e-6)
    assert heading == pytest.approx(232.54310919606212, rel=1e-6)
    assert wind_u == pytest.approx(16.924026519286894, rel=1e-6)
    assert wind_v == pytest.approx(0.0, rel=1e-6)


def test_selects_valid_time_and_converts_pressure(ground_track):
    mission = Mission(
        origin='BOS',
        destination='ATL',
        aircraft_type='738',
        departure='2020-01-01T05:30:00',
        arrival='2020-01-01T07:30:00',
        load_factor=1.0,
    )

    valid_times = pd.date_range('2020-01-01', periods=3, freq='h')
    pressure_levels = np.array([1000.0, 900.0])
    lat = np.array([42.0, 41.5])
    lon = np.array([-71.0, -70.0])
    data = np.stack(
        [
            np.full(
                (len(pressure_levels), len(lat), len(lon)), fill_value=i, dtype=float
            )
            for i in range(len(valid_times))
        ]
    )

    ds = xr.Dataset(
        data_vars={
            't': (('valid_time', 'pressure_level', 'lat', 'lon'), data),
            'u': (('valid_time', 'pressure_level', 'lat', 'lon'), data),
            'v': (('valid_time', 'pressure_level', 'lat', 'lon'), -data),
        },
        coords={
            'valid_time': valid_times,
            'pressure_level': pressure_levels,
            'lat': lat,
            'lon': lon,
        },
    )

    weather = Weather(
        ds=ds,
        mission=mission,
        ground_track=ground_track,
        fl_min=0,
        fl_max=20,
        fl_spacing=10,
    )

    assert weather.valid_time_index == 5
    assert weather.ds['t'].values.mean() == pytest.approx(2.0)
    assert weather._pressure_coord_name() == 'pressure_level'

    pressure_values = weather.p_regridded.filter(like='NM_').to_numpy()
    assert pressure_values.min() > 90000.0
    assert pressure_values.max() < 110000.0


def test_time_dimension_and_lat_lon_fallback(ground_track):
    mission = Mission(
        origin='BOS',
        destination='ATL',
        aircraft_type='738',
        departure=iso_to_timestamp('2020-01-01 03:00:00'),
        arrival=iso_to_timestamp('2020-01-01 05:00:00'),
        load_factor=1.0,
    )

    times = pd.date_range('2020-01-01', periods=2, freq='h')
    level = np.array([950.0, 850.0])
    y = np.array([45.0, 44.5])
    x = np.array([-71.0, -70.5])

    first_slice = np.full((len(level), len(y), len(x)), 5.0)
    second_slice = np.full((len(level), len(y), len(x)), 8.0)
    data = np.stack([first_slice, second_slice])

    ds = xr.Dataset(
        data_vars={
            't': (('time', 'level', 'y', 'x'), data),
            'u': (('time', 'level', 'y', 'x'), data + 1),
            'v': (('time', 'level', 'y', 'x'), data + 2),
        },
        coords={'time': times, 'level': level, 'y': y, 'x': x},
    )

    weather = Weather(
        ds=ds,
        mission=mission,
        ground_track=ground_track,
        fl_min=0,
        fl_max=20,
        fl_spacing=10,
    )

    assert weather.valid_time_index == 3
    assert weather.ds['t'].values.mean() == pytest.approx(second_slice.mean())
    assert weather._lat_lon_names() == ('y', 'x')
    assert weather._pressure_coord_name() == 'level'


def test_validate_altitudes_and_supported_range(sample_weather):
    clipped = sample_weather._validate_altitudes(
        np.array([500.0, 40000.0]), context='unit-test'
    )
    assert clipped[0] == sample_weather.fl_min
    assert clipped[1] == sample_weather.fl_max
    assert sample_weather.get_supported_altitude_range_ft() == (
        sample_weather.fl_min * 100,
        sample_weather.fl_max * 100,
    )


def test_nearest_indices_and_interpolation_bounds(sample_weather):
    ascending = np.array([0.0, 10.0, 20.0])
    targets = np.array([1.0, 9.9, 20.0])
    assert sample_weather._nearest_indices_1d(ascending, targets).tolist() == [
        0,
        1,
        2,
    ]

    descending = ascending[::-1]
    assert sample_weather._nearest_indices_1d(descending, targets).tolist() == [
        2,
        1,
        0,
    ]

    min_val = sample_weather._interpolate_value(sample_weather.u_regridded, -5.0, 0)
    edge_min = sample_weather._interpolate_value(
        sample_weather.u_regridded, 0.0, sample_weather.fl_min
    )
    assert min_val == pytest.approx(edge_min)

    high_nm = sample_weather.total_nm + 50.0
    max_val = sample_weather._interpolate_value(
        sample_weather.u_regridded, high_nm, sample_weather.fl_max + 100
    )
    edge_max = sample_weather._interpolate_value(
        sample_weather.u_regridded, sample_weather.total_nm, sample_weather.fl_max
    )
    assert max_val == pytest.approx(edge_max)


def test_plot_variable_map_saves_file(sample_weather, tmp_path):
    import matplotlib

    matplotlib.use('Agg')

    out_file = tmp_path / 'u_plot.png'
    sample_weather.plot_variable_map('u', save_path=out_file, title='test plot')
    assert out_file.exists()
    assert out_file.stat().st_size > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

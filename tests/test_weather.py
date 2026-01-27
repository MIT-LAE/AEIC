import pytest

from AEIC.missions import Mission
from AEIC.trajectories.ground_track import GroundTrack
from AEIC.utils.helpers import iso_to_timestamp
from AEIC.weather.weather import Weather

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

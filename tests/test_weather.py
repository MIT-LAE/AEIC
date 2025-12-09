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
    return "weather/sample_weather_subset.nc"


@pytest.fixture(scope='session')
def sample_weather(ground_track, weather_dataset_path):
    return Weather(
        weather_data_path=str(weather_dataset_path),
        mission=sample_mission,
        ground_track=ground_track,
    )


def test_weather_init_with_bad_str(ground_track):
    with pytest.raises(FileNotFoundError):
        Weather(
            weather_data_path="no/file.nc",
            mission=sample_mission,
            ground_track=ground_track,
        )


def test_compute_ground_speed(sample_weather):
    ground_distance_m = 370400.0
    altitude_m = 9144.0
    tas_ms = 200.0

    gs = sample_weather.get_ground_speed(
        ground_distance=ground_distance_m,
        altitude=altitude_m,
        true_airspeed=tas_ms,
    )

    assert gs == pytest.approx(191.02855126751604, rel=1e-6)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

import numpy as np
import xarray as xr

from AEIC.missions import Mission
from AEIC.trajectories.ground_track import GroundTrack
from AEIC.utils.files import file_location


class Weather:
    """
    A class to query weather data variables and ground speed along
    ground track points.

    Parameters
    ----------
    weather_data_path : str
        Path to ERA5 Weather data NetCDF file
        File should contain variables: 't', 'u', 'v'
        with coordinates 'pressure_level', 'latitude', 'longitude',
        'valid_time' (optional)
    mission : Mission
        Mission object with origin, destination location
                as well as missions start time
    ground_track : GroundTrack object with waypoints along mission
    """

    def __init__(
        self, weather_data_path: str, mission: Mission, ground_track: GroundTrack
    ):
        self.ground_track = ground_track
        # Read weather file
        weather_ds = xr.open_dataset(file_location(weather_data_path))

        # If valid_hour exists, slice weather to get hour of departure
        if 'valid_time' in weather_ds.dims:
            weather_ds = weather_ds.isel(valid_time=mission.departure.hour)

        self.weather_ds = weather_ds

    def get_ground_speed(
        self,
        ground_distance: float,
        altitude: float,
        true_airspeed: float,
        azimuth: float = None,
    ) -> float:
        """
        Compute ground speed at a point along the mission.

        Parameters
        ----------
        ground_distance : float
            Distance flown along the ground track from the origin [meters].
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
        gt_point = self.ground_track.location(ground_distance)

        wind_u = self.weather_ds['u'].interp(
            pressure_level=self._altitude_to_pressure_level_hPa(altitude),
            latitude=gt_point.location.latitude,
            longitude=gt_point.location.longitude,
        )

        wind_v = self.weather_ds['v'].interp(
            pressure_level=self._altitude_to_pressure_level_hPa(altitude),
            latitude=gt_point.location.latitude,
            longitude=gt_point.location.longitude,
        )

        if azimuth is None:
            heading_rad = np.deg2rad(gt_point.azimuth)
        else:
            heading_rad = np.deg2rad(azimuth)

        u_air = true_airspeed * np.cos(heading_rad)
        v_air = true_airspeed * np.sin(heading_rad)

        ground_speed = float(np.hypot(u_air + wind_u, v_air + wind_v))
        return ground_speed

    def _altitude_to_pressure_level_hPa(self, altitude: float) -> float:
        """Convert altitude to pressure level."""
        pressure_hPa = 1013.25 * (1.0 - altitude / 44330.0) ** 5.255
        return pressure_hPa

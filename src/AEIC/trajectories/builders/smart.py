from dataclasses import dataclass, field

from AEIC.missions import Mission
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories import FlightPhase, GroundTrack, Trajectory
from AEIC.utils.files import file_location
from AEIC.utils.units import (
    FEET_TO_METERS,
)
from AEIC.weather.weather import Weather

from .base import Builder, Context, Options


@dataclass
class SmartOptions:
    """Additional options for the smart trajectory builder."""

    phases: dict[FlightPhase, int] = field(
        default_factory=lambda: {
            FlightPhase.CLIMB: 10,
            FlightPhase.CRUISE: 10,
            FlightPhase.DESCENT: 10,
        }
    )
    """Flight phases and the number of points per phase to be simulated."""

    fuel_LHV: float = 43.8e6  # J/kg
    """Lower heating value of the fuel used."""

    climb_type: str = 'max_roc'
    """Solution method used to perform climb."""

    cruise_type: str = 'const_CL'
    """Solution method used to perform cruise."""

    descent_type: str = 'const_TAS'
    """Solution method used to perform descent."""


class SmartContext(Context):
    """Context for smart trajectory builder."""

    def __init__(
        self,
        builder: 'SmartBuilder',
        ac_performance: PerformanceModel,
        mission: Mission,
        starting_mass: float | None,
    ):
        # The context constructor calculates all of the fixed information used
        # throughout the simulation by the trajectory builder.
        self.options = builder.options
        self.smart_options = builder.smart_options

        # Generate great circle ground track between departure and arrival
        # locations.
        ground_track = GroundTrack.great_circle(
            mission.origin_position.location, mission.destination_position.location
        )

        # Climb defined as starting 3000' above airport.
        self.clm_start_altitude = (
            mission.origin_position.altitude + 3000.0 * FEET_TO_METERS
        )

        # Maximum altitude in meters.
        max_alt: float = (
            ac_performance.model_info['General_Information']['max_alt_ft']
            * FEET_TO_METERS
        )

        # If starting altitude is above operating ceiling, set start altitude
        # to departure airport altitude.
        if self.clm_start_altitude >= max_alt:
            self.clm_start_altitude = mission.origin_position.altitude

        # Set descent altitude based on 3000' above arrival airport altitude;
        # clamp to aircraft operating ceiling if needed.
        self.des_end_altitude = (
            mission.destination_position.altitude + 3000.0 * FEET_TO_METERS
        )
        if self.des_end_altitude >= max_alt:
            self.des_end_altitude = max_alt

        # Determine whether takeoff procedures are being simulated;
        # initial altitude is origin altitude if so, otherwise clm_start_altitude
        phases = self.smart_options.phases
        takeoff_phases = [
            FlightPhase.IDLE_ORIGIN,
            FlightPhase.TAXI_ORIGIN,
            FlightPhase.TAKEOFF,
        ]
        if any([phase in self.phases for phase in takeoff_phases]):
            initial_altitude = mission.origin_position.altitude
        else:
            initial_altitude = self.clm_start_altitude

        # Initialize weather regridding when requested.
        self.weather: Weather | None = None
        if self.options.use_weather:
            mission_date = mission.departure.strftime('%Y%m%d')
            weather_path = file_location(
                f"{ac_performance.config.weather_data_dir}/{mission_date}.nc"
            )
            self.weather = Weather(
                weather_data_path=weather_path,
                mission=mission,
                ground_track=ground_track,
            )

        # Pass information to base context class constructor.
        super().__init__(
            builder,
            ac_performance,
            mission,
            ground_track,
            npoints=phases,
            initial_altitude=initial_altitude,
            starting_mass=starting_mass,
        )


class SmartBuilder(Builder):
    """Model for determining flight trajectories using the a 'smart' method.

    Args:
        options (Options): Base options for trajectory building.
        legacy_options (LegactyOptions): Builder-specific options for legacy
            trajectory builder.
    """

    CONTEXT_CLASS = SmartContext

    def __init__(
        self,
        options: Options = Options(),
        smart_options: SmartOptions = SmartOptions(),
        *args,
        **kwargs,
    ):
        super().__init__(options, *args, **kwargs)
        self.smart_options = smart_options

    def calc_starting_mass(self, **kwargs) -> float:
        """"""
        pass

    def fly_climb(self, traj: Trajectory, **kwargs) -> None:
        """"""
        pass

    def fly_cruise(self, traj: Trajectory, **kwargs):
        """"""
        pass

    def fly_descent(self, traj: Trajectory, **kwargs):
        """"""
        pass

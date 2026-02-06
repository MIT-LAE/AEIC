from dataclasses import dataclass

import numpy as np

from AEIC.config import config
from AEIC.missions import Mission
from AEIC.performance.models import LegacyPerformanceModel
from AEIC.performance.types import AircraftState, SimpleFlightRules
from AEIC.units import (
    FEET_TO_METERS,
    METERS_TO_FL,
    MINUTES_TO_SECONDS,
    NAUTICAL_MILES_TO_METERS,
)
from AEIC.weather import Weather

from .. import FlightPhase, GroundTrack, Trajectory
from .base import Builder, Context, Options


@dataclass
class LegacyOptions:
    """Additional options for the legacy trajectory builder."""

    frac_step_clm: float = 0.01
    """Climb step size as fraction of total climb altitude change."""

    frac_step_crz: float = 0.01
    """Cruise step size as fraction of total cruise ground distance."""

    frac_step_des: float = 0.01
    """Descent step size as fraction of total descent altitude change."""

    fuel_LHV: float = 43.8e6
    """Lower heating value of the fuel used (J/kg)."""


class LegacyContext(Context):
    """Context for legacy trajectory builder."""

    def __init__(
        self,
        builder: 'LegacyBuilder',
        ac_performance: LegacyPerformanceModel,
        mission: Mission,
        starting_mass: float | None,
    ):
        # The context constructor calculates all of the fixed information used
        # throughout the simulation by the trajectory builder.

        # Number of points in different flight phases. Last point of climb is
        # same as first point of cruise, last point of cruise is same as first
        # point of descent.
        n_climb = int(1 / builder.frac_step_clm)
        n_cruise = int(1 / builder.frac_step_crz)
        n_descent = int(1 / builder.frac_step_des + 1)

        # Generate great circle ground track between departure and arrival
        # locations. Allow the trajectory to step beyond the end of the ground
        # track to account for the fact that we just guess the point at which
        # we need to start the descent - depending on the exact airspeed values
        # returned by the performance model, we will either reach the ground
        # before or after the destination point. If the airspeeds are higher
        # than expected by the descent start point estimate, we will fly past
        # the destination, so will need to step along the ground track beyond the
        # final destination waypoint.
        ground_track = GroundTrack.great_circle(
            mission.origin_position.location,
            mission.destination_position.location,
            allow_overstep=True,
        )

        # Climb defined as starting 3000' above airport.
        self.clm_start_altitude = (
            mission.origin_position.altitude + 3000.0 * FEET_TO_METERS
        )

        # If starting altitude is above operating ceiling, set start altitude
        # to departure airport altitude.
        if self.clm_start_altitude >= ac_performance.maximum_altitude:
            self.clm_start_altitude = mission.origin_position.altitude

        # Cruise altitude is the operating ceiling - 7000 feet.
        self.crz_start_altitude = (
            ac_performance.maximum_altitude - 7000.0 * FEET_TO_METERS
        )

        # Ensure cruise altitude is above the starting altitude.
        if self.crz_start_altitude < self.clm_start_altitude:
            self.crz_start_altitude = self.clm_start_altitude

        # Prevent flying above aircraft ceiling. (NOTE: this will only trigger
        # due to random variables not currently implemented.)
        if self.crz_start_altitude > ac_performance.maximum_altitude:
            self.crz_start_altitude = ac_performance.maximum_altitude

        # In legacy trajectory, descent start altitude is equal to cruise
        # altitude.
        self.des_start_altitude = self.crz_start_altitude

        # Set descent altitude based on 3000' above arrival airport altitude;
        # clamp to aircraft operating ceiling if needed.
        self.des_end_altitude = (
            mission.destination_position.altitude + 3000.0 * FEET_TO_METERS
        )
        if self.des_end_altitude >= ac_performance.maximum_altitude:
            self.des_end_altitude = ac_performance.maximum_altitude

        if self.crz_start_altitude < self.clm_start_altitude:
            raise ValueError(
                "Departure airport + 3000ft should not be higher"
                "than start of cruise point"
            )
        if self.des_end_altitude > self.des_start_altitude:
            raise ValueError(
                "Arrival airport + 3000ft should not be higher thanend of cruise point"
            )
        self.descent_dist_approx = 18.23 * (
            self.des_start_altitude - self.des_end_altitude
        )
        if self.descent_dist_approx < 0:
            raise ValueError('Arrival airport should not be above cruise altitude')

        # Initialize weather regridding when requested.
        self.weather: Weather | None = None
        if builder.options.use_weather:
            self.weather = Weather(data_dir=config.weather.weather_data_dir)

        # Pass information to base context class constructor.
        super().__init__(
            builder,
            ac_performance,
            mission,
            ground_track,
            npoints={
                FlightPhase.CLIMB: n_climb,
                FlightPhase.CRUISE: n_cruise,
                FlightPhase.DESCENT: n_descent,
            },
            initial_altitude=self.clm_start_altitude,
            starting_mass=starting_mass,
        )


class LegacyBuilder(Builder):
    """Model for determining flight trajectories using the legacy method
    from AEIC v2.

    Args:
        options (Options): Base options for trajectory building.
        legacy_options (LegactyOptions): Builder-specific options for legacy
            trajectory builder.
    """

    CONTEXT_CLASS = LegacyContext

    def __init__(
        self,
        options: Options = Options(),
        legacy_options: LegacyOptions = LegacyOptions(),
    ):
        super().__init__(options)

        # Define discretization of each phase in steps as a percent of
        # the overall distance/altitude change
        self.frac_step_clm = legacy_options.frac_step_clm
        self.frac_step_crz = legacy_options.frac_step_crz
        self.frac_step_des = legacy_options.frac_step_des

        self.fuel_LHV = legacy_options.fuel_LHV

    def calc_starting_mass(self) -> float:
        """Calculates the starting mass using AEIC v2 methods.
        Sets both starting mass and non-reserve/hold/divert fuel mass."""

        perf = self.ac_performance.evaluate(
            AircraftState(
                altitude=self.crz_start_altitude,
                aircraft_mass='max',
            ),
            SimpleFlightRules.CRUISE,
        )

        # Figure out startingMass components per AEIC v2:
        #
        #      |   empty weight
        #      | + payload weight
        #      | + fuel weight
        #      | + fuel reserves weight
        #      | + fuel divert weight
        #      | + fuel hold weight
        #      | _______________________
        #        = Take-off weight

        # Payload.
        payload_mass = self.ac_performance.maximum_payload * self.mission.load_factor

        # Fuel needed (distance / velocity * fuel flow rate).
        approx_time = self.ground_track.total_distance / perf.true_airspeed
        fuel_mass = approx_time * perf.fuel_flow

        # Reserve fuel (assumed 5%).
        reserve_mass = fuel_mass * 0.05

        # Diversion and hold fuel per AEIC v2.
        if approx_time > 180 * MINUTES_TO_SECONDS:
            divert_dist = 200.0 * NAUTICAL_MILES_TO_METERS
            hold_time = 30 * MINUTES_TO_SECONDS
        else:
            divert_dist = 100.0 * NAUTICAL_MILES_TO_METERS
            hold_time = 45 * MINUTES_TO_SECONDS
        divert_mass = divert_dist / perf.true_airspeed * perf.fuel_flow
        hold_mass = hold_time * perf.fuel_flow

        starting_mass = (
            self.ac_performance.empty_mass
            + payload_mass
            + fuel_mass
            + reserve_mass
            + divert_mass
            + hold_mass
        )

        # Limit to MTOM if overweight.
        if starting_mass > self.ac_performance.maximum_mass:
            starting_mass = self.ac_performance.maximum_mass

        # Set fuel mass (for weight residual calculation).
        self.total_fuel_mass = fuel_mass

        return starting_mass

    def fly_climb(self, traj: Trajectory) -> None:
        self._fly_level_change(
            traj,
            SimpleFlightRules.CLIMB,
            0,
            np.linspace(self.clm_start_altitude, self.crz_start_altitude, self.n_climb),
        )

    def fly_descent(self, traj: Trajectory) -> None:
        self._fly_level_change(
            traj,
            SimpleFlightRules.DESCEND,
            self.n_climb + self.n_cruise,
            np.linspace(self.des_start_altitude, self.des_end_altitude, self.n_descent),
        )

    def _fly_level_change(
        self,
        traj: Trajectory,
        flight_rule: SimpleFlightRules,
        start_index: int,
        altitudes: np.ndarray,
    ) -> None:
        """Simulate climb and descent phases.

        Computes state over segments involving level changes using AEIC v2
        methods based on BADA-3 formulas."""

        # Final trajectory index to fill.
        end_index = start_index + len(altitudes) - 1

        # Start next segment at end of previous segment (fuel flow, TAS will be
        # replaced).
        if start_index > 0:
            traj.copy_point(start_index - 1, start_index)
        else:
            # Initial values for first call to performance model: replaced by
            # feasible values in first iteration below.
            traj.true_airspeed[0] = min(self.ac_performance.performance_table.tas)
            traj.rate_of_climb[0] = max(self.ac_performance.performance_table.rocd)

        # Regular steps in altitude between trajectory points.
        traj.altitude[start_index : end_index + 1] = altitudes
        traj.flight_level[start_index : end_index + 1] = altitudes * METERS_TO_FL

        for i in range(start_index, end_index + 1):
            # Determine aircraft performance at start of segment.
            perf = self.ac_performance.evaluate(
                AircraftState(
                    altitude=traj.altitude[i],
                    true_airspeed=traj.true_airspeed[i],
                    rate_of_climb=traj.rate_of_climb[i],
                    aircraft_mass=traj.aircraft_mass[i],
                ),
                flight_rule,
            )
            traj.fuel_flow[i] = perf.fuel_flow
            traj.true_airspeed[i] = perf.true_airspeed
            traj.rate_of_climb[i] = perf.rate_of_climb

            # No further processing needed for last point.
            if i == end_index:
                traj.heading[i] = traj.heading[i - 1]
                traj.ground_speed[i] = traj.ground_speed[i - 1]
                break

            # Calculate the forward true airspeed (used for ground speed).
            fwd_tas = np.sqrt(perf.true_airspeed**2 - perf.rate_of_climb**2)

            # Time to complete altitude change segment and total fuel burned.
            seg_time = (traj.altitude[i + 1] - traj.altitude[i]) / perf.rate_of_climb
            seg_fuel = perf.fuel_flow * seg_time

            # Ground speed, including weather effects if required.
            if self.weather is None:
                traj.ground_speed[i] = fwd_tas
            else:
                traj.ground_speed[i] = self.weather.get_ground_speed(
                    time=self.mission.departure,
                    gt_point=self.ground_track.location(traj.ground_distance[i]),
                    altitude=traj.altitude[i],
                    true_airspeed=fwd_tas,
                    azimuth=traj.azimuth[i],
                )
            traj.heading[i] = traj.azimuth[i]

            # Calculate distance along route travelled.
            dist = traj.ground_speed[i] * seg_time

            # Take step along great circle route.
            pt = self.ground_track.step(traj.ground_distance[i], dist)
            traj.longitude[i + 1] = pt.location.longitude
            traj.latitude[i + 1] = pt.location.latitude
            traj.azimuth[i + 1] = pt.azimuth

            # Account for acceleration/deceleration over the segment using
            # end-of-segment tas approximated using start of segment TAS, ROCD
            # and mass and end-of-segment altitude.
            perf_end = self.ac_performance.evaluate(
                AircraftState(
                    altitude=traj.altitude[i + 1],
                    true_airspeed=traj.true_airspeed[i],
                    rate_of_climb=traj.rate_of_climb[i],
                    aircraft_mass=traj.aircraft_mass[i],
                ),
                flight_rule,
            )

            kinetic_energy_chg = (
                0.5
                * traj.aircraft_mass[i]
                * (perf_end.true_airspeed**2 - perf.true_airspeed**2)
            )

            # Calculate fuel required for acceleration
            # NOTE: I have no idea where AEIC v2 got the efficiency of 0.15 from
            accel_fuel = kinetic_energy_chg / self.fuel_LHV / 0.15

            seg_fuel += accel_fuel

            # We cannot gain fuel by decelerating in a conventional fuel
            # aircraft.
            if seg_fuel < 0:
                seg_fuel = 0

            # Update the state vector.
            traj.fuel_mass[i + 1] = traj.fuel_mass[i] - seg_fuel
            traj.aircraft_mass[i + 1] = traj.aircraft_mass[i] - seg_fuel
            traj.ground_distance[i + 1] = traj.ground_distance[i] + dist
            traj.flight_time[i + 1] = traj.flight_time[i] + seg_time

    def fly_cruise(self, traj: Trajectory):
        """Simulate cruise phase.

        Computes state over cruise segment using AEIC v2 methods based on
        BADA-3 formulas
        """

        # Start cruise at end-of-climb position and mass (fuel flow, TAS will
        # be replaced).
        traj.copy_point(self.n_climb - 1, self.n_climb)
        start_dist = traj.ground_distance[self.n_climb - 1]

        # Cruise at constant altitude.
        traj.altitude[self.n_climb : self.n_climb + self.n_cruise] = (
            self.crz_start_altitude
        )
        traj.flight_level[self.n_climb : self.n_climb + self.n_cruise] = (
            self.crz_start_altitude * METERS_TO_FL
        )

        # Cruise end distance based on estimated descent distance.
        end_dist = self.ground_track.total_distance - self.descent_dist_approx

        # Cruise is discretized into ground distance steps.
        traj.ground_distance[self.n_climb : self.n_climb + self.n_cruise] = np.linspace(
            start_dist, end_dist, self.n_cruise
        )
        ground_distance_step = (end_dist - start_dist) / (self.n_cruise - 1)

        # Top of climb, entering cruise.
        traj.rate_of_climb[self.n_climb] = 0

        # Get fuel flow, ground speed, etc. for cruise segments.
        for i in range(self.n_climb, self.n_climb + self.n_cruise):
            if self.weather is not None:
                traj.ground_speed[i] = self.weather.get_ground_speed(
                    time=self.mission.departure,
                    gt_point=self.ground_track.location(traj.ground_distance[i]),
                    altitude=traj.altitude[i],
                    true_airspeed=traj.true_airspeed[i],
                    azimuth=traj.azimuth[i],
                )
                traj.heading[i] = traj.azimuth[i]
            else:
                traj.ground_speed[i] = traj.true_airspeed[i]
                traj.heading[i] = traj.azimuth[i]

            # Calculate time required to fly the segment.
            seg_time = ground_distance_step / traj.ground_speed[i]

            # Take step along great circle route.
            pt = self.ground_track.step(traj.ground_distance[i], ground_distance_step)
            traj.longitude[i + 1] = pt.location.longitude
            traj.latitude[i + 1] = pt.location.latitude
            traj.azimuth[i + 1] = pt.azimuth

            # Get fuel flow rate from performance model.
            perf = self.ac_performance.evaluate(
                AircraftState(
                    altitude=traj.altitude[i],
                    true_airspeed=traj.true_airspeed[i],
                    rate_of_climb=0,
                    aircraft_mass=traj.aircraft_mass[i],
                ),
                SimpleFlightRules.CRUISE,
            )

            # Calculate fuel burn in [kg] over the segment.
            seg_fuel = perf.fuel_flow * seg_time

            # Set aircraft state values.
            traj.true_airspeed[i + 1] = perf.true_airspeed
            traj.rate_of_climb[i + 1] = perf.rate_of_climb
            traj.fuel_flow[i + 1] = perf.fuel_flow
            traj.fuel_mass[i + 1] = traj.fuel_mass[i] - seg_fuel
            traj.aircraft_mass[i + 1] = traj.aircraft_mass[i] - seg_fuel

            traj.flight_time[i + 1] = traj.flight_time[i] + seg_time

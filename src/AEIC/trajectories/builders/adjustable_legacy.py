# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from AEIC.config import config
from AEIC.missions import Mission
from AEIC.performance.models import LegacyPerformanceModel
from AEIC.performance.models.legacy import ROCDFilter
from AEIC.performance.types import AircraftState, SimpleFlightRules
from AEIC.storage import FlightPhase
from AEIC.units import (
    FEET_TO_METERS,
    FL_TO_METERS,
    METERS_TO_FL,
    MINUTES_TO_SECONDS,
    NAUTICAL_MILES_TO_METERS,
)
from AEIC.weather import Weather

from .. import GroundTrack, Trajectory
from .base import Builder, Context, Options
from .legacy import LegacyOptions


class AdjustableLegacyContext(Context):
    """Context for adjustable legacy trajectory builder."""

    class AdjustmentFunction(Protocol):
        """Protocol for context adjustment functions.

        An adjustment function takes in the context, mission, performance model
        possibly additional optional keyword arguments and returns a float
        value."""

        def __call__(
            self,
            context: AdjustableLegacyContext,
            mission: Mission,
            performance: LegacyPerformanceModel,
            **kwargs,
        ) -> float: ...

    # A context adjustment is either an adjustment function, a fixed float
    # value, or None (in which case no adjustment is applied and the behavior
    # falls back to the standard legacy trajectory builder behavior).
    ContextAdjustment = AdjustmentFunction | float | None

    def __init__(
        self,
        builder: AdjustableLegacyBuilder,
        ac_performance: LegacyPerformanceModel,
        mission: Mission,
        starting_mass: float | None,
        climb_start_altitude: ContextAdjustment = None,
        cruise_altitude: ContextAdjustment = None,
        descent_end_altitude: ContextAdjustment = None,
        descent_distance: ContextAdjustment = None,
        reserve_fuel: ContextAdjustment = None,
        divert_distance: ContextAdjustment = None,
        hold_time: ContextAdjustment = None,
    ):
        """Adjustment parameters (a "simple" adjustment is one that takes no
        additional arguments beyond the context, mission and performance
        model):

        - `climb_start_altitude`: Simple adjustment for climb start altitude,
          which defaults to 3000' above departure airport altitude if not
          provided.

        - `cruise_altitude`: Simple adjustment for cruise altitude, which
           defaults to 7000' below aircraft operating ceiling if not provided.

        - `descent_end_altitude`: Simple adjustment for descent end altitude,
           which defaults to 3000' above arrival airport altitude if not
           provided.

        - `descent_distance`: Simple adjustment for descent distance, which
           defaults to a value based on the difference between cruise altitude
           and descent end altitude if not provided.

        - `reserve_fuel`: Adjustment for reserve fuel mass, which defaults to
           5% of total fuel mass if not provided (additional parameter:
           `fuel_mass`, total fuel mass consumed based on approximate flight
           time and nominal fuel flow).

        - `divert_distance`: Adjustment for diversion distance, which defaults
          to 200 NM if flight time is over 3 hours and 100 NM if flight time is
          under 3 hours if not provided (additional parameter: `approx_time`,
          approximate flight time based on distance and nominal cruise speed).

        - `hold_time`: Adjustment for hold time, which defaults to 30 minutes
          if flight time is over 3 hours and 45 minutes if flight time is under
          3 hours if not provided (additional parameter: `approx_time`,
          approximate flight time based on distance and nominal cruise speed).

        """

        # The context constructor calculates all of the fixed information used
        # throughout the simulation by the trajectory builder.

        # Number of points in different flight phases. Last point of climb is
        # same as first point of cruise, last point of cruise is same as first
        # point of descent.

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

        # Climb defined as starting 3000' above airport (adjustable).
        if climb_start_altitude is None:
            self.clm_start_altitude = (
                mission.origin_position.altitude + 3000.0 * FEET_TO_METERS
            )
        else:
            self.clm_start_altitude = self.apply_adjustment(
                climb_start_altitude, mission, ac_performance
            )

        # If starting altitude is above operating ceiling, set start altitude
        # to departure airport altitude.
        if self.clm_start_altitude >= ac_performance.maximum_altitude:
            self.clm_start_altitude = mission.origin_position.altitude

        # Cruise altitude is the operating ceiling - 7000 feet (adjustable).
        if cruise_altitude is None:
            self.crz_start_altitude = (
                ac_performance.maximum_altitude - 7000.0 * FEET_TO_METERS
            )
        else:
            self.crz_start_altitude = self.apply_adjustment(
                cruise_altitude, mission, ac_performance
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
        # clamp to aircraft operating ceiling if needed (adjustable).
        if descent_end_altitude is None:
            self.des_end_altitude = (
                mission.destination_position.altitude + 3000.0 * FEET_TO_METERS
            )
        else:
            self.des_end_altitude = self.apply_adjustment(
                descent_end_altitude, mission, ac_performance
            )

        if self.des_end_altitude >= ac_performance.maximum_altitude:
            self.des_end_altitude = ac_performance.maximum_altitude

        if self.crz_start_altitude < self.clm_start_altitude:
            raise ValueError(
                "Departure airport + 3000ft should not be higher "
                "than start of cruise point"
            )
        if self.des_end_altitude > self.des_start_altitude:
            raise ValueError(
                "Arrival airport + 3000ft should not be higher than end of cruise point"
            )
        if descent_distance is None:
            self.descent_dist_approx = 18.228347 * (
                self.des_start_altitude - self.des_end_altitude
            )
        else:
            self.descent_dist_approx = self.apply_adjustment(
                descent_distance, mission, ac_performance
            )
        if self.descent_dist_approx < 0:
            raise ValueError('Descent distance must be non-negative')

        # Initialize weather regridding when requested.
        self.weather: Weather | None = None
        if builder.options.use_weather:
            self.weather = Weather(
                data_dir=config.weather.weather_data_dir,
                file_resolution=config.weather.file_resolution,
                data_resolution=config.weather.data_resolution,
                file_format=config.weather.file_format,
            )

        # Save reserve fuel, divert distance and hold time adjustments for use
        # in starting mass calculation.
        self.reserve_fuel = reserve_fuel
        self.divert_distance = divert_distance
        self.hold_time = hold_time

        # Pass information to base context class constructor.
        super().__init__(
            builder,
            ac_performance,
            mission,
            ground_track,
            initial_altitude=self.clm_start_altitude,
            starting_mass=starting_mass,
        )

    def apply_adjustment(
        self,
        adjustment: ContextAdjustment,
        mission: Mission,
        performance: LegacyPerformanceModel,
        **kwargs,
    ) -> float:
        """Helper function to apply a context adjustment."""
        if adjustment is None:
            raise RuntimeError('Attempting to apply an empty ContextAdjustment.')
        elif isinstance(adjustment, Callable):
            return adjustment(self, mission, performance, **kwargs)
        else:
            return adjustment


class AdjustableLegacyBuilder(Builder):
    """Model for determining flight trajectories using the legacy method
    from AEIC v2.

    Args:
        options (Options): Base options for trajectory building.
        legacy_options (LegacyOptions): Builder-specific options for legacy
            trajectory builder.
    """

    CONTEXT_CLASS = AdjustableLegacyContext

    def __init__(
        self,
        options: Options = Options(),
        legacy_options: LegacyOptions = LegacyOptions(),
    ):
        super().__init__(options)

        # Altitude step to use in climb and descent phases of flight.
        self.altitude_step = legacy_options.altitude_step

        # Ground distance step to use in cruise phase.
        self.cruise_step = legacy_options.cruise_step

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

        lowest_cruise_altitude = (
            min(self.ac_performance.performance_table(ROCDFilter.ZERO).fl)
            * FL_TO_METERS
        )

        perf_low = self.ac_performance.evaluate(
            AircraftState(
                altitude=lowest_cruise_altitude,
                aircraft_mass='min',
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
        if self.reserve_fuel is None:
            reserve_mass = fuel_mass * 0.05
        else:
            reserve_mass = self.apply_adjustment(
                self.reserve_fuel,
                self.mission,
                self.ac_performance,
                fuel_mass=fuel_mass,
            )

        # Diversion and hold fuel per AEIC v2.
        # TODO: ADJUSTMENT POINT
        if self.divert_distance is None:
            if approx_time > 180 * MINUTES_TO_SECONDS:
                divert_dist = 200.0 * NAUTICAL_MILES_TO_METERS
            else:
                divert_dist = 100.0 * NAUTICAL_MILES_TO_METERS
        else:
            divert_dist = self.apply_adjustment(
                self.divert_distance,
                self.mission,
                self.ac_performance,
                approx_time=approx_time,
            )
        if self.hold_time is None:
            if approx_time > 180 * MINUTES_TO_SECONDS:
                hold_time = 30 * MINUTES_TO_SECONDS
            else:
                hold_time = 45 * MINUTES_TO_SECONDS
        else:
            hold_time = self.apply_adjustment(
                self.hold_time,
                self.mission,
                self.ac_performance,
                approx_time=approx_time,
            )
        divert_mass = divert_dist / perf.true_airspeed * perf.fuel_flow
        hold_mass = hold_time * perf_low.fuel_flow

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
        """Simulate the climb phase of a trajectory.

        Delegates to :meth:`_fly_level_change` with climb flight rules, integrating
        from the end of the previous phase up to the cruise-start altitude.
        """
        self._fly_level_change(
            traj,
            FlightPhase.CLIMB,
            SimpleFlightRules.CLIMB,
            self.crz_start_altitude,
        )

    def fly_descent(self, traj: Trajectory) -> None:
        """Simulate the descent phase of a trajectory.

        Delegates to :meth:`_fly_level_change` with descent flight rules,
        integrating from the cruise altitude down to the descent-end altitude.
        The final per-phase point count is decremented by one to account for
        the shared boundary point with the next phase.
        """
        self._fly_level_change(
            traj,
            FlightPhase.DESCENT,
            SimpleFlightRules.DESCEND,
            self.des_end_altitude,
        )
        if traj.n_descent > 0:
            traj.n_descent -= 1

    def _fly_level_change(
        self,
        traj: Trajectory,
        flight_phase: FlightPhase,
        flight_rule: SimpleFlightRules,
        final_altitude: float,
    ) -> None:
        """Simulate climb and descent phases.

        Computes state over segments involving level changes using AEIC v2
        methods based on BADA-3 formulas."""

        traj.set_phase(flight_phase)

        # Start climb at overall flight start point and start descent at end of
        # cruise point.
        if flight_phase == FlightPhase.CLIMB:
            pt = self._start_point(traj)
            traj.append(pt)
        else:
            pt = traj.make_point(-1)

        # Flight phase is discretized into constant altitude steps with a
        # possible extra "short step" at the end to reach the target altitude.
        altitudes = np.arange(
            pt.altitude,
            final_altitude,
            self.altitude_step
            if flight_phase == FlightPhase.CLIMB
            else -self.altitude_step,
        )
        if len(altitudes) == 0:
            return
        if (flight_phase == FlightPhase.CLIMB and altitudes[-1] < final_altitude) or (
            flight_phase == FlightPhase.DESCENT and altitudes[-1] > final_altitude
        ):
            altitudes = np.append(altitudes, final_altitude)

        # Loop over altitude change segments.
        for start_altitude, end_altitude in zip(altitudes[:-1], altitudes[1:]):
            # Determine aircraft performance at start of segment.
            perf = self.ac_performance.evaluate(
                AircraftState(
                    altitude=start_altitude,  # type: ignore
                    true_airspeed=pt.true_airspeed,  # type: ignore
                    rate_of_climb=pt.rate_of_climb,  # type: ignore
                    aircraft_mass=pt.aircraft_mass,  # type: ignore
                ),
                flight_rule,
            )

            fwd_tas = np.sqrt(perf.true_airspeed**2 - perf.rate_of_climb**2)

            # Time to complete altitude change segment and total fuel burned.
            seg_time = (end_altitude - start_altitude) / perf.rate_of_climb
            seg_fuel = perf.fuel_flow * seg_time

            # Ground speed, including weather effects if required.
            if self.weather is None:
                pt.ground_speed = fwd_tas
            else:
                pt.ground_speed = self.weather.get_ground_speed(
                    time=self.mission.departure,
                    gt_point=self.ground_track.location(pt.ground_distance),
                    altitude=start_altitude,
                    true_airspeed=fwd_tas,
                    azimuth=pt.azimuth,
                )
            pt.heading = pt.azimuth

            pt.altitude = end_altitude
            pt.flight_level = pt.altitude * METERS_TO_FL

            # Calculate distance along route travelled.
            dist = pt.ground_speed * seg_time

            # Take step along great circle route.
            gpt = self.ground_track.step(pt.ground_distance, dist)
            pt.longitude = gpt.location.longitude
            pt.latitude = gpt.location.latitude
            pt.azimuth = gpt.azimuth

            # Calculate fuel required for acceleration.
            match flight_phase:
                case FlightPhase.CLIMB:
                    # Account for acceleration/deceleration over the segment
                    # using end-of-segment tas approximated using start of
                    # segment TAS, ROCD and mass and end-of-segment altitude.
                    perf_end = self.ac_performance.evaluate(
                        AircraftState(
                            altitude=end_altitude,  # type: ignore
                            true_airspeed=pt.true_airspeed,  # type: ignore
                            rate_of_climb=pt.rate_of_climb,  # type: ignore
                            aircraft_mass=pt.aircraft_mass,  # type: ignore
                        ),
                        flight_rule,
                    )

                    kinetic_energy_chg = (
                        0.5
                        * pt.aircraft_mass
                        * (perf_end.true_airspeed**2 - perf.true_airspeed**2)
                    )

                    # NOTE: I have no idea where AEIC v2 got the efficiency of
                    # 0.15 from.
                    # TODO: ADJUSTMENT POINT
                    efficiency = 0.15
                    accel_fuel = kinetic_energy_chg / self.fuel_LHV / efficiency
                    seg_fuel += accel_fuel
                case FlightPhase.DESCENT:
                    # For descent, assumed fuel flow is essentially just engine
                    # at idle.
                    pass

            # We cannot gain fuel by decelerating in a conventional fuel
            # aircraft.
            if seg_fuel < 0:
                seg_fuel = 0

            # Update the state vector.
            pt.fuel_mass -= seg_fuel
            pt.aircraft_mass -= seg_fuel
            pt.ground_distance += dist

            pt.flight_time += seg_time

            traj.fuel_flow[-1] = perf.fuel_flow
            traj.true_airspeed[-1] = perf.true_airspeed
            traj.rate_of_climb[-1] = perf.rate_of_climb

            traj.append(pt)

    def fly_cruise(self, traj: Trajectory):
        """Simulate cruise phase.

        Computes state over cruise segment using AEIC v2 methods based on
        BADA-3 formulas."""

        traj.set_phase(FlightPhase.CRUISE)

        # Start cruise at end-of-climb position and mass (fuel flow, TAS will
        # be replaced).
        pt = traj.make_point(-1)

        # Cruise at constant altitude.
        pt.altitude = self.crz_start_altitude
        pt.flight_level = self.crz_start_altitude * METERS_TO_FL

        # Cruise end distance based on estimated descent distance.
        end_dist = self.ground_track.total_distance - self.descent_dist_approx

        # If there is not enough distance to fly before starting descent, skip
        # cruise phase.
        if pt.ground_distance >= end_dist:
            return

        # Cruise is discretized into ground distance steps with a possible
        # extra "short step" at the end to reach the target distance.
        distances = np.arange(pt.ground_distance, end_dist, self.cruise_step)
        if distances[-1] != end_dist:
            distances = np.append(distances, end_dist)

        # Top of climb, entering cruise.
        pt.rate_of_climb = 0

        # Get fuel flow, ground speed, etc. for cruise segments.
        for distance in distances[1:]:
            # Get fuel flow rate from performance model.
            perf = self.ac_performance.evaluate(
                AircraftState(
                    altitude=pt.altitude,
                    true_airspeed=pt.true_airspeed,  # type: ignore
                    rate_of_climb=0,
                    aircraft_mass=pt.aircraft_mass,  # type: ignore
                ),
                SimpleFlightRules.CRUISE,
            )

            if self.weather is not None:
                pt.ground_speed = self.weather.get_ground_speed(
                    time=self.mission.departure,
                    gt_point=self.ground_track.location(distance),
                    altitude=pt.altitude,
                    true_airspeed=perf.true_airspeed,
                    azimuth=pt.azimuth,
                )
            else:
                pt.ground_speed = perf.true_airspeed
            pt.heading = pt.azimuth

            # Take step along great circle route.
            gpt = self.ground_track.location(distance)
            pt.longitude = gpt.location.longitude
            pt.latitude = gpt.location.latitude
            pt.azimuth = gpt.azimuth

            # Calculate time required to fly the segment.
            ground_distance_step = distance - pt.ground_distance
            pt.ground_distance = distance
            seg_time = ground_distance_step / pt.ground_speed

            # Calculate fuel burn in [kg] over the segment.
            seg_fuel = perf.fuel_flow * seg_time

            # Set aircraft state values.
            pt.fuel_mass -= seg_fuel
            pt.aircraft_mass -= seg_fuel
            pt.flight_time += seg_time

            # Store performance data for point at beginning of this step.
            traj.true_airspeed[-1] = perf.true_airspeed
            traj.rate_of_climb[-1] = perf.rate_of_climb
            traj.fuel_flow[-1] = perf.fuel_flow

            traj.append(pt)

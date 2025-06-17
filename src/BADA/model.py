import numpy as np

# from scipy.integrate import cumulative_trapezoid
from abc import ABC, abstractmethod

from utils.helpers import mps_to_knots, meters_to_feet
from .fuel_burn_base import BaseFuelBurnModel
from utils.custom_types import FloatOrNDArray
from .aircraft_parameters import Bada3AircraftParameters
from utils.consts import g0
from utils.standard_atmosphere import (
    temperature_at_altitude_isa_bada4,
    calculate_air_density,
    pressure_at_altitude_isa_bada4,
)


class Bada3EngineModel(ABC):
    def __init__(self, aircraft_parameters: Bada3AircraftParameters):
        self.aircraft_parameters = aircraft_parameters

    @abstractmethod
    def calculate_specific_fuel_consumption(self, *args, **kwargs) -> FloatOrNDArray:
        pass

    @abstractmethod
    def calculate_nominal_fuel_flow(
        self, thrust, v_tas, *args, **kwargs
    ) -> FloatOrNDArray:
        pass

    @abstractmethod
    def calculate_cruise_fuel_flow(
        self, thrust, v_tas, *args, **kwargs
    ) -> FloatOrNDArray:
        pass

    @abstractmethod
    def calculate_max_climb_thrust_isa(
        self, altitude, v_tas, *args, **kwargs
    ) -> FloatOrNDArray:
        pass

    def calculate_max_climb_thrust(
        self,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        temperature: FloatOrNDArray,
    ) -> FloatOrNDArray:
        """Calculate maximum climb thrust for non-ISA conditions, using equation (3.7-4)

        Parameters
        ----------
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        temperature : Union[float, NDArray]
            Temperature [K].

        Returns
        -------
        Union[float, NDArray]
            Maximum climb thrust [N].
        """

        delta_temperature = temperature - temperature_at_altitude_isa_bada4(altitude)
        delta_temperature_eff = delta_temperature - self.aircraft_parameters['c_tc4']

        return self.calculate_max_climb_thrust_isa(altitude, v_tas) * (
            1
            - np.clip(
                delta_temperature_eff * np.maximum(0, self.aircraft_parameters['c_tc5']),
                0,
                0.4,
            )
        )

    def calculate_max_cruise_thrust(
        self,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        temperature: FloatOrNDArray,
    ) -> FloatOrNDArray:
        """Calculate maximum cruise thrust, using equation (3.7-8)

        Parameters
        ----------
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        temperature : Union[float, NDArray]
            Temperature [K].

        Returns
        -------
        Union[float, NDArray]
            Maximum cruise thrust [N].
        """

        return (
            self.calculate_max_climb_thrust(altitude, v_tas, temperature)
            * self.aircraft_parameters.c_tcr
        )

    def calculate_descent_thrust_high(
        self,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        temperature: FloatOrNDArray,
    ) -> FloatOrNDArray:
        """Calculate descent thrust for high altitudes, using equation (3.7-9)

        Parameters
        ----------
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        temperature : Union[float, NDArray]
            Temperature [K].

        Returns
        -------
        Union[float, NDArray]
            Descent thrust [N].
        """
        return self.aircraft_parameters['c_tdes_high'] * self.calculate_max_climb_thrust(
            altitude, v_tas, temperature
        )

    def calculate_descent_thrust_low(
        self,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        temperature: FloatOrNDArray,
    ) -> FloatOrNDArray:
        """Calculate descent thrust for low altitudes, using equation (3.7-10)

        Parameters
        ----------
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        temperature : Union[float, NDArray]
            Temperature [K].

        Returns
        -------
        Union[float, NDArray]
            Descent thrust [N].
        """
        return self.aircraft_parameters.c_tdes_low * self.calculate_max_climb_thrust(
            altitude, v_tas, temperature
        )

    def calculate_descent_thrust_app(
        self,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        temperature: FloatOrNDArray,
    ) -> FloatOrNDArray:
        """Calculate descent thrust for approach conditions, using equation (3.7-11)

        Parameters
        ----------
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        temperature : Union[float, NDArray]
            Temperature [K].

        Returns
        -------
        Union[float, NDArray]
            Descent thrust [N].
        """
        return self.aircraft_parameters.c_tdes_app * self.calculate_max_climb_thrust(
            altitude, v_tas, temperature
        )

    def calculate_descent_thrust_land(
        self,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        temperature: FloatOrNDArray,
    ) -> FloatOrNDArray:
        """Calculate descent thrust for landing conditions, using equation (3.7-12)

        Parameters
        ----------
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        temperature : Union[float, NDArray]
            Temperature [K].

        Returns
        -------
        Union[float, NDArray]
            Descent thrust [N].
        """
        return self.aircraft_parameters.c_tdes_ld * self.calculate_max_climb_thrust(
            altitude, v_tas, temperature
        )


class Bada3JetEngineModel(Bada3EngineModel):

    def calculate_specific_fuel_consumption(
        self, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """Calculate specific fuel consumption for jet, using equation (3.9-1)

        Parameters
        ----------
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Specific fuel consumption [kg/N/s].
        """
        return (
            self.aircraft_parameters['c_f1']
            * (1 + mps_to_knots(v_tas) / self.aircraft_parameters['c_f2'])
            / (60 * 1000)
        )

    def calculate_nominal_fuel_flow(
        self, thrust: FloatOrNDArray, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """Calculate nominal fuel flow for jet, using equation (3.9-3)

        Parameters
        ----------
        thrust : Union[float, NDArray]
            Thrust [N].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Nominal fuel flow [kg/s].
        """
        return self.calculate_specific_fuel_consumption(v_tas) * thrust

    def calculate_cruise_fuel_flow(
        self, thrust: FloatOrNDArray, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """Calculate cruise fuel flow for jet, using equation (3.9-6)

        Parameters
        ----------
        thrust : Union[float, NDArray]
            Thrust [N].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Cruise fuel flow [kg/s].
        """
        return (
            self.calculate_specific_fuel_consumption(v_tas)
            * thrust
            * self.aircraft_parameters['c_fcr']
        )

    def calculate_max_climb_thrust_isa(
        self, altitude: FloatOrNDArray, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """Calculate maximum climb thrust for ISA conditions, using equation (3.7-1)

        Parameters
        ----------
        altitude : Union[float, NDArray]
            Altitude [m].

        v_tas : Union[float, NDArray]
            True airspeed [m/s]. UNUSED - only for consistency with other engine models.

        Returns
        -------
        Union[float, NDArray]
            Maximum climb thrust [N].
        """
        altitude_ft = meters_to_feet(altitude)
        return self.aircraft_parameters['c_tc1'] * (
            1
            - altitude_ft / self.aircraft_parameters['c_tc2']
            + self.aircraft_parameters['c_tc3'] * altitude_ft**2
        )


class Bada3TurbopropEngineModel(Bada3EngineModel):

    def calculate_specific_fuel_consumption(
        self, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """Calculate specific fuel consumption for turboprop, using equation (3.9-2)

        Parameters
        ----------
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Specific fuel consumption [kg/N/s].
        """
        return (
            self.aircraft_parameters.c_f1
            * (1 - mps_to_knots(v_tas) / self.aircraft_parameters.c_f2)
            * (mps_to_knots(v_tas) / 1000)
            / (60 * 1000)
        )

    def calculate_nominal_fuel_flow(
        self, thrust: FloatOrNDArray, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """Calculate nominal fuel flow for turboprop, using equation (3.9-3)

        Parameters
        ----------
        thrust : Union[float, NDArray]
            Thrust [N].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Nominal fuel flow [kg/s].
        """
        return self.calculate_specific_fuel_consumption(v_tas) * thrust

    def calculate_cruise_fuel_flow(
        self, thrust: FloatOrNDArray, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """Calculate cruise fuel flow for turboprop, using equation (3.9-6)

        Parameters
        ----------
        thrust : Union[float, NDArray]
            Thrust [N].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Cruise fuel flow [kg/s].
        """
        return (
            self.calculate_specific_fuel_consumption(v_tas)
            * thrust
            * self.aircraft_parameters.c_fcr
        )

    def calculate_max_climb_thrust_isa(
        self, altitude: FloatOrNDArray, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """Calculate maximum climb thrust for ISA conditions, using equation (3.7-2)

        Parameters
        ----------
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Maximum climb thrust [N].
        """
        altitude_ft = meters_to_feet(altitude)
        v_tas_kts = mps_to_knots(v_tas)

        return (self.aircraft_parameters.c_tc1 / v_tas_kts) * (
            1 - altitude_ft / self.aircraft_parameters.c_tc2
        ) + self.aircraft_parameters.c_tc3


class Bada3PistonEngineModel(Bada3EngineModel):

    def calculate_specific_fuel_consumption(self) -> None:
        """
        Returns
        -------
        None - thrust specific fuel consumption is not defined for piston engines.
        """

        return None

    def calculate_nominal_fuel_flow(self, thrust, v_tas) -> FloatOrNDArray:
        """

        Parameters
        ----------
        thrust : Union[float, NDArray]
            Consistency Placeholder for thrust.
        v_tas : Union[float, NDArray]
            Consistency Placeholder for true airspeed.

        Returns
        -------
        Union[float, NDArray] - nominal fuel flow for piston engines.
        """
        return self.aircraft_parameters.c_f1

    def calculate_cruise_fuel_flow(self, thrust, v_tas) -> FloatOrNDArray:
        """

        Parameters
        ----------
        thrust : Union[float, NDArray]
            Consistency Placeholder for thrust.
        v_tas : Union[float, NDArray]
            Consistency Placeholder for true airspeed.

        Returns
        -------
        Union[float, NDArray] - cruise fuel flow for piston engines.
        """
        return self.aircraft_parameters.c_f1 * self.aircraft_parameters.c_fcr

    def calculate_max_climb_thrust_isa(
        self, altitude: FloatOrNDArray, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """
        Calculate maximum climb thrust for ISA conditions, using equation (3.7-3)

        Parameters
        ----------
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Maximum climb thrust [N].
        """
        altitude_ft = meters_to_feet(altitude)
        v_tas_kts = mps_to_knots(v_tas)

        return (
            self.aircraft_parameters.c_tc1
            * (1 - altitude_ft / self.aircraft_parameters.c_tc2)
            + self.aircraft_parameters.c_tc3 / v_tas_kts
        )


class Bada3FuelBurnModel(BaseFuelBurnModel):
    """BADA3 fuel burn model class."""

    def __init__(self, aircraft_parameters: Bada3AircraftParameters):
        self.aircraft_parameters = aircraft_parameters
        self.create_engine_model()

    def create_engine_model(self) -> Bada3EngineModel:
        engine_type = self.aircraft_parameters.engine_type
        if engine_type == "Jet":
            self.engine_model = Bada3JetEngineModel(self.aircraft_parameters)

        elif engine_type == "Turboprop":
            self.engine_model = Bada3TurbopropEngineModel(self.aircraft_parameters)

        elif engine_type == "Piston":
            self.engine_model = Bada3PistonEngineModel(self.aircraft_parameters)
        elif engine_type == "Electric":
            raise NotImplementedError("Electric engine model not implemented.")
        else:
            raise ValueError(f"Engine type not recognised: {engine_type}")

    def calculate_cl(
        self, mass: FloatOrNDArray, rho: FloatOrNDArray, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """Calculate lift coefficient using equation (3.6-1)

        Parameters
        ----------
        mass : Union[float, NDArray]
            Aircraft mass [kg].
        rho : Union[float, NDArray]
            Air density [kg/m^3].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Lift coefficient.
        """
        return 2 * mass * g0 / (rho * self.aircraft_parameters.S_ref * v_tas**2)

    def calculate_cd(self, cl: FloatOrNDArray) -> FloatOrNDArray:
        """
        Calculate drag coefficient using equation (3.6-2)

        Parameters
        ----------
        cl : Union[float, NDArray]
            Lift coefficient.

        Returns
        -------
        Union[float, NDArray]
            Drag coefficient.
        """
        return self.aircraft_parameters.c_d0cr + self.aircraft_parameters.c_d2cr * cl**2

    def calculate_drag(
        self, cd: FloatOrNDArray, rho: FloatOrNDArray, v_tas: FloatOrNDArray
    ) -> FloatOrNDArray:
        """
        Calculate drag using equation (3.6-5)

        Parameters
        ----------
        cd : Union[float, NDArray]
            Drag coefficient.
        rho : Union[float, NDArray]
            Air density [kg/m^3].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].

        Returns
        -------
        Union[float, NDArray]
            Drag [N].
        """
        return 0.5 * rho * self.aircraft_parameters.S_ref * v_tas**2 * cd

    def calculate_thrust_by_total_energy(
        self,
        drag: FloatOrNDArray,
        mass: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        rocd: FloatOrNDArray,
        acceleration: FloatOrNDArray,
    ) -> FloatOrNDArray:
        """
        Calculate thrust by total energy method, i.e. equation (3.2-1)

        Parameters
        ----------
        drag : Union[float, NDArray]
            Drag [N].
        mass : Union[float, NDArray]
            Aircraft mass [kg].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        rocd : Union[float, NDArray]
            Rate of climb or descent [m/s].
        acceleration : Union[float, NDArray]
            Acceleration [m/s^2].

        Returns
        -------
        Union[float, NDArray]
            Thrust [N].
        """
        return drag + mass * (g0 * (1 / v_tas) * rocd + acceleration)

    def calculate_thrust(
        self,
        mass: FloatOrNDArray,
        temperature: FloatOrNDArray,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        rocd: FloatOrNDArray,
        acceleration: FloatOrNDArray,
        in_cruise: FloatOrNDArray,
    ) -> FloatOrNDArray:
        """
        Calculate thrust from total energy but correct extremes using max and descent thrusts

        Parameters
        ----------
        mass : Union[float, NDArray]
            Aircraft mass [kg].
        temperature : Union[float, NDArray]
            Temperature [K].
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        rocd : Union[float, NDArray]
            Rate of climb or descent [m/s].
        acceleration : Union[float, NDArray]
            Acceleration [m/s^2].
        in_cruise : Union[float, NDArray]
            Flag indicating whether aircraft is in cruise.

        Returns
        -------
        Union[float, NDArray]
            Thrust [N].
        """
        pressure = pressure_at_altitude_isa_bada4(altitude)
        rho = calculate_air_density(pressure, temperature)
        cl = self.calculate_cl(mass, rho, v_tas)
        cd = self.calculate_cd(cl)
        drag = self.calculate_drag(cd, rho, v_tas)
        thrust = self.calculate_thrust_by_total_energy(
            drag, mass, v_tas, rocd, acceleration
        )

        max_climb_thrust = self.engine_model.calculate_max_climb_thrust(
            altitude, v_tas, temperature
        )
        max_cruise_thrust = self.engine_model.calculate_max_cruise_thrust(
            altitude, v_tas, temperature
        )
        max_thrust = np.where(in_cruise, max_cruise_thrust, max_climb_thrust)
        # where thrust above max thrust, assign max thrust
        thrust = np.where(thrust > max_thrust, max_thrust, thrust)

        descent_thrust_high = self.engine_model.calculate_descent_thrust_high(
            altitude, v_tas, temperature
        )
        descent_thrust_low = self.engine_model.calculate_descent_thrust_low(
            altitude, v_tas, temperature
        )
        descent_thrust = np.where(
            meters_to_feet(altitude) > self.aircraft_parameters.h_p_des,
            descent_thrust_high,
            descent_thrust_low,
        )

        # where thrust below 0, assign descent thrust
        thrust = np.where(thrust < 0, descent_thrust, thrust)

        return thrust

    def calculate_specific_ground_range(
        self,
        mass: FloatOrNDArray,
        temperature: FloatOrNDArray,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        rocd: FloatOrNDArray,
        acceleration: FloatOrNDArray,
        in_cruise: FloatOrNDArray,
        groundspeed: FloatOrNDArray,
    ) -> FloatOrNDArray:
        """
        Calculate specific ground range

        Parameters
        ----------
        mass : Union[float, NDArray]
            Aircraft mass [kg].
        temperature : Union[float, NDArray]
            Temperature [K].
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        rocd : Union[float, NDArray]
            Rate of climb or descent [m/s].
        acceleration : Union[float, NDArray]
            Acceleration [m/s^2].
        in_cruise : Union[float, NDArray]
            Flag indicating whether aircraft is in cruise.
        groundspeed : Union[float, NDArray]
            Ground speed [m/s].


        Returns
        -------
        Union[float, NDArray]
            Specific ground range [m/kg]
        """

        thrust = self.calculate_thrust(
            mass, temperature, altitude, v_tas, rocd, acceleration, in_cruise
        )

        fuel_flow = self.engine_model.calculate_nominal_fuel_flow(thrust, v_tas)
        fuel_flow_cruise = self.engine_model.calculate_cruise_fuel_flow(thrust, v_tas)

        fuel_flow = np.where(in_cruise, fuel_flow_cruise, fuel_flow)

        return np.divide(
            groundspeed, fuel_flow, out=np.zeros_like(groundspeed), where=fuel_flow != 0
        )

    def iterate_flight_simulation_constant_initial_mass(
        self,
        temperature: FloatOrNDArray,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        rocd: FloatOrNDArray,
        acceleration: FloatOrNDArray,
        in_cruise: FloatOrNDArray,
        groundspeed: FloatOrNDArray,
        segment_distance: FloatOrNDArray,
        initial_mass: float,
        n_iter: int = 10,
    ) -> FloatOrNDArray:
        """
        Iterate flight simulation for constant initial mass

        Parameters
        ----------

        temperature : Union[float, NDArray]
            Temperature [K].
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        rocd : Union[float, NDArray]
            Rate of climb or descent [m/s].
        acceleration : Union[float, NDArray]
            Acceleration [m/s^2].
        in_cruise : Union[float, NDArray]
            Flag indicating whether aircraft is in cruise.
        groundspeed : Union[float, NDArray]
            Ground speed [m/s].
        segment_distance : Union[float, NDArray]
            Segment distance [m].
        initial_mass : float
            Initial mass [kg].
        n_iter : int, optional
            Number of iterations, by default 10



        Returns
        -------
        Union[float, NDArray]
            mass vector [kg]
        """

        mass = np.full(len(groundspeed), initial_mass)

        specific_ground_range = self.calculate_specific_ground_range(
            mass,
            temperature,
            altitude,
            v_tas,
            rocd,
            acceleration,
            in_cruise,
            groundspeed,
        )

        mass = self.update_mass_vector(mass, specific_ground_range, segment_distance)

        old_final_mass = mass[-1].copy()

        for i in range(n_iter - 1):
            specific_ground_range = self.calculate_specific_ground_range(
                mass,
                temperature,
                altitude,
                v_tas,
                rocd,
                acceleration,
                in_cruise,
                groundspeed,
            )

            mass = self.update_mass_vector(
                mass, specific_ground_range, segment_distance
            )

            final_mass_pct_change = (
                np.abs(mass[-1] - old_final_mass) / old_final_mass
            ) * 100

            if final_mass_pct_change < 0.01:
                return mass

            old_final_mass = mass[-1].copy()
        return mass

    def iterate_flight_simulation_constant_final_mass(
        self,
        temperature: FloatOrNDArray,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        rocd: FloatOrNDArray,
        acceleration: FloatOrNDArray,
        in_cruise: FloatOrNDArray,
        groundspeed: FloatOrNDArray,
        segment_distance: FloatOrNDArray,
        final_mass: float,
        n_iter: int = 10,
    ) -> FloatOrNDArray:
        """
        Iterate flight simulation for constant initial mass

        Parameters
        ----------

        temperature : Union[float, NDArray]
            Temperature [K].
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        rocd : Union[float, NDArray]
            Rate of climb or descent [m/s].
        acceleration : Union[float, NDArray]
            Acceleration [m/s^2].
        in_cruise : Union[float, NDArray]
            Flag indicating whether aircraft is in cruise.
        groundspeed : Union[float, NDArray]
            Ground speed [m/s].
        segment_distance : Union[float, NDArray]
            Segment distance [m].
        final_mass : float
            Final mass [kg].
        n_iter : int, optional
            Number of iterations, by default 10



        Returns
        -------
        Union[float, NDArray]
            mass vector [kg]
        """

        mass = np.full(len(groundspeed), final_mass)

        specific_ground_range = self.calculate_specific_ground_range(
            mass,
            temperature,
            altitude,
            v_tas,
            rocd,
            acceleration,
            in_cruise,
            groundspeed,
        )

        mass = self.update_mass_vector_backward(
            mass, specific_ground_range, segment_distance
        )

        old_initial_mass = mass[0].copy()

        for i in range(n_iter - 1):
            specific_ground_range = self.calculate_specific_ground_range(
                mass,
                temperature,
                altitude,
                v_tas,
                rocd,
                acceleration,
                in_cruise,
                groundspeed,
            )

            mass = self.update_mass_vector_backward(
                mass, specific_ground_range, segment_distance
            )

            initial_mass_pct_change = (
                np.abs(mass[0] - old_initial_mass) / old_initial_mass
            ) * 100

            if initial_mass_pct_change < 0.01:
                return mass

            old_initial_mass = mass[0].copy()
        return mass

    def iterate_flight_simulation_fuel_burn_dependent_initial_mass_rf_fraction(
        self,
        temperature: FloatOrNDArray,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        rocd: FloatOrNDArray,
        acceleration: FloatOrNDArray,
        in_cruise: FloatOrNDArray,
        groundspeed: FloatOrNDArray,
        segment_distance: FloatOrNDArray,
        initial_mass_estimate: float,
        mtow: float,
        oew: float,
        mpl: float,
        load_factor: float,
        reserve_fuel_fraction: float,
        n_iter: int = 10,
    ) -> FloatOrNDArray:
        """
        Iterate flight simulation for fuel burn dependent initial mass where reserve fuel is defined by fraction of fuel burn

        Parameters
        ----------

        temperature : Union[float, NDArray]
            Temperature [K].
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        rocd : Union[float, NDArray]
            Rate of climb or descent [m/s].
        acceleration : Union[float, NDArray]
            Acceleration [m/s^2].
        in_cruise : Union[float, NDArray]
            Flag indicating whether aircraft is in cruise.
        groundspeed : Union[float, NDArray]
            Ground speed [m/s].
        segment_distance : Union[float, NDArray]
            Segment distance [m].
        initial_mass_estimate : float
            Initial mass estimate [kg].
        mtow : float
            Maximum takeoff weight [kg].
        oew : float
            Operating empty weight [kg].
        mpl : float
            Maximum payload [kg].
        load_factor : float
            Load factor [-].
        reserve_fuel_fraction : float
            Fraction of fuel burn to reserve fuel.
        n_iter : int, optional
            Number of iterations, by default 10


        Returns
        -------
        Union[float, NDArray]
            mass vector [kg]
        """

        mass = np.full(len(groundspeed), initial_mass_estimate)

        specific_ground_range = self.calculate_specific_ground_range(
            mass,
            temperature,
            altitude,
            v_tas,
            rocd,
            acceleration,
            in_cruise,
            groundspeed,
        )

        mass = self.update_mass_vector(mass, specific_ground_range, segment_distance)

        old_final_mass = mass[-1].copy()

        for i in range(n_iter):
            specific_ground_range = self.calculate_specific_ground_range(
                mass,
                temperature,
                altitude,
                v_tas,
                rocd,
                acceleration,
                in_cruise,
                groundspeed,
            )

            mass = self.update_mass_vector(
                mass, specific_ground_range, segment_distance
            )

            fuel_burn = mass[0] - mass[-1]

            initial_mass = np.min(
                (
                    oew + mpl * load_factor + fuel_burn * (1 + reserve_fuel_fraction),
                    mtow,
                )
            )

            mass[0] = initial_mass

            final_mass_pct_change = (
                np.abs(mass[-1] - old_final_mass) / old_final_mass
            ) * 100

            if final_mass_pct_change < 0.01:
                return mass

            old_final_mass = mass[-1].copy()

        return mass

    def iterate_flight_simulation_fuel_burn_dependent_initial_mass_rf_value(
        self,
        temperature: FloatOrNDArray,
        altitude: FloatOrNDArray,
        v_tas: FloatOrNDArray,
        rocd: FloatOrNDArray,
        acceleration: FloatOrNDArray,
        in_cruise: FloatOrNDArray,
        groundspeed: FloatOrNDArray,
        segment_distance: FloatOrNDArray,
        initial_mass_estimate: float,
        mtow: float,
        oew: float,
        mpl: float,
        load_factor: float,
        reserve_fuel: float,
        n_iter: int = 10,
    ) -> FloatOrNDArray:
        """
        Iterate flight simulation for fuel burn dependent initial mass where reserve fuel is defined by value

        Parameters
        ----------

        temperature : Union[float, NDArray]
            Temperature [K].
        altitude : Union[float, NDArray]
            Altitude [m].
        v_tas : Union[float, NDArray]
            True airspeed [m/s].
        rocd : Union[float, NDArray]
            Rate of climb or descent [m/s].
        acceleration : Union[float, NDArray]
            Acceleration [m/s^2].
        in_cruise : Union[float, NDArray]
            Flag indicating whether aircraft is in cruise.
        groundspeed : Union[float, NDArray]
            Ground speed [m/s].
        segment_distance : Union[float, NDArray]
            Segment distance [m].
        initial_mass_estimate : float
            Initial mass estimate [kg].
        mtow : float
            Maximum takeoff weight [kg].
        oew : float
            Operating empty weight [kg].
        mpl : float
            Maximum payload [kg].
        load_factor : float
            Load factor [-].
        reserve_fuel : float
            Reserve fuel [kg].
        n_iter : int, optional
            Number of iterations, by default 10


        Returns
        -------
        Union[float, NDArray]
            mass vector [kg]
        """

        mass = np.full(len(groundspeed), initial_mass_estimate)

        specific_ground_range = self.calculate_specific_ground_range(
            mass,
            temperature,
            altitude,
            v_tas,
            rocd,
            acceleration,
            in_cruise,
            groundspeed,
        )

        mass = self.update_mass_vector(mass, specific_ground_range, segment_distance)

        old_final_mass = mass[-1].copy()

        for i in range(n_iter):
            specific_ground_range = self.calculate_specific_ground_range(
                mass,
                temperature,
                altitude,
                v_tas,
                rocd,
                acceleration,
                in_cruise,
                groundspeed,
            )

            mass = self.update_mass_vector(
                mass, specific_ground_range, segment_distance
            )

            fuel_burn = mass[0] - mass[-1]

            initial_mass = np.min(
                (
                    oew + mpl * load_factor + fuel_burn + reserve_fuel,
                    mtow,
                )
            )

            mass[0] = initial_mass

            final_mass_pct_change = (
                np.abs(mass[-1] - old_final_mass) / old_final_mass
            ) * 100

            if final_mass_pct_change < 0.01:

                return mass

            old_final_mass = mass[-1].copy()

        return mass

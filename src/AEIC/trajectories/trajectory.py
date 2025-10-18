import numpy as np
import xarray as xr
from pyproj import Geod

from AEIC.performance_model import PerformanceModel
from utils import airports


class Trajectory:
    """Parent class for all trajectory implementations in AEIC. Contains overall
    ``fly_flight`` logic.

    Args:
        ac_performance (PerformanceModel): Performance model used for trajectory
            simulation.
        mission (_type_): Data dictating the mission to be flown (departure/arrival
            info, etc.).
        optimize_traj (bool): (Currently unimplemented) Flag controlling whether the
            nominal trajectory undergoes horizontal, vertical, and speed optimization
            during simulation.
        iterate_mass (bool): Flag controlling whether starting mass is iterated on such
            that the remaining fuel (non-reserve) is close to 0 at the arrival airport.
        startMass (float, optional): Starting mass of the aircraft; leave as default to
            calculate starting mass during simulation. Defaults to -1.
        max_mass_iters (int, optional): Maximum number of mass iterations (if used).
            Defaults to 5.
        mass_iter_reltol (float, optional): Desired relative tolerance for mass
            iteration. Defaults to 1e-2.

    Attributes:
        name (str): Identifier for the flight; format is
            ``departureAirport_arrivalAirport_ACCode``.
        ac_performance (PerformanceModel): Performance model used for trajectory
            simulation.
        dep_lon_lat_alt (list[float]): Longitude, latitude, and altitude of the
            departure airport.
        arr_lon_lat_alt (list[float]): Longitude, latitude, and altitude of the arrival
            airport.
        start_time (str): Departure date and time.
        end_time (str): Nominal arrival date and time.
        gc_distance (float): (TODO: Change from naut. miles to meters) Great circle
            distance between departure and arrival airports.
        load_factor (float): Load factor of the flight relative to maximum payload.
        optimize_traj (bool): (Currently unimplemented) Flag controlling whether the
            nominal trajectory undergoes horizontal, vertical, and speed optimization
            during simulation.
        iter_mass (bool): Flag controlling whether starting mass is iterated on such
            that the remaining fuel (non-reserve) is close to 0 at the arrival airport.
        max_mass_iters (int): Maximum number of mass iterations (if used). Defaults to
            5.
        mass_iter_reltol (float): Desired relative tolerance for mass iteration.
            Defaults to 1e-2.
        mass_converged (bool): ``True`` if mass iteration was converged, ``False`` if
            not. ``None`` if not iterating mass.
        starting_mass (float): Starting mass of the aircraft.
        fuel_mass (float): Total fuel mass loaded onto the aircraft.
        NClm (int): Number of points simulated in climb.
        NCrz (int): Number of points simulated in cruise.
        NDes (int): Number of points simulated in descent.
        Ntot (int): Number of points simulated across total mission.
        clm_start_altitude (float): Climb starting altitude.
        crz_start_altitude (float): Cruise starting altitude.
        des_start_altitude (float): Descent starting altitude.
        des_end_altitude (float): End-of-descent altitude.
        traj_data (xr.Dataset): xarray Dataset containing all trajectory data.
            Only defined when ``fly_flight`` is called. Variables are:\n
            ``fuelFlow``: fuel flow rate.\n
            ``acMass``: aircraft mass.\n
            ``fuelMass``: fuel mass.\n
            ``groundDist``: total ground distance covered.\n
            ``altitude``: current pressure altitude.\n
            ``FLs``: flight level equivalent of the pressure altitude.\n
            ``rocs``: rate of climb/descent.\n
            ``flightTime``: elapsed flight time.\n
            ``latitude``: current latitude.\n
            ``longitude``: current longitude.\n
            ``azimuth``: azimuth angle.\n
            ``heading``: current heading.\n
            ``tas``: true airspeed.\n
            ``groundSpeed``: ground speed.\n
            ``FL_weight``: weighting used in linear interpolation over flight levels.
    """

    def __init__(
        self,
        ac_performance: PerformanceModel,
        mission,
        optimize_traj: bool,
        iterate_mass: bool,
        startMass: float = -1,
        max_mass_iters: int = 5,
        mass_iter_reltol: float = 1e-2,
    ) -> None:
        # Save A/C performance model and the mission to be flown
        # NOTE: Currently assume that `mission` comes in as a dictionary with
        # the format of a single flight
        # in `src/missions/sample_missions_10.json`. We also assume that
        # Load Factor for the flight will be
        # included in the mission object.
        self.name = f'{mission.origin}_{mission.destination}_{mission.aircraft_type}'
        self.ac_performance = ac_performance

        # Save airport locations and dep/arr times; lat/long in degrees
        ori_airport_data = airports.airports[mission.origin]
        des_airport_data = airports.airports[mission.destination]
        self.dep_lon_lat_alt = [
            ori_airport_data.longitude,
            ori_airport_data.latitude,
            ori_airport_data.elevation,
        ]
        self.arr_lon_lat_alt = [
            des_airport_data.longitude,
            des_airport_data.latitude,
            des_airport_data.elevation,
        ]

        self.start_time = mission.departure
        self.end_time = mission.arrival

        # Convert gc distance from km to meters
        self.gc_distance = mission.distance * 1e3
        self.geod = Geod(ellps="WGS84")

        # Get load factor from mission object
        self.load_factor = 1.0  # FIXME: mission.seat_capacity

        # Controls whether or not route optimization is performed
        # NOTE: This currently does nothing
        self.optimize_traj = optimize_traj

        # Control whether or not starting mass is iterated on
        self.iter_mass = iterate_mass
        self.max_mass_iters = max_mass_iters
        self.mass_iter_reltol = mass_iter_reltol
        self.mass_converged = None

        # Allow user to specify starting mass if desired
        self.starting_mass = startMass

        # Initialize a non-reserve, non-divert/hold fuel mass for mass residual
        # calculation
        self.fuel_mass = None

        # Initialize values for number of simulated points per segment
        # (to be defined in child classes)
        self.NClm = None
        self.NCrz = None
        self.NDes = None
        self.Ntot = None

        # Initialize important altitudes (clm = climb, crz = cruise, des = descent)
        self.clm_start_altitude = None
        self.crz_start_altitude = None
        self.des_start_altitude = None
        self.des_end_altitude = None

    def fly_flight(self, **kwargs) -> None:
        """Top-level function that initiates and runs flights.

        Args:
            kwargs: Additional parameters needed by the specific type of trajectory
                being used.
        """
        # Initialize dataset with point dimension
        self.traj_data = xr.Dataset(coords={'point': np.arange(self.Ntot)})

        # Define variable metadata
        var_metadata = {
            'fuelFlow': {'description': 'Fuel flow rate', 'units': 'kg/s'},
            'acMass': {'description': 'Aircraft mass', 'units': 'kg'},
            'fuelMass': {'description': 'Fuel mass remaining', 'units': 'kg'},
            'groundDist': {'description': 'Ground distance traveled', 'units': 'm'},
            'altitude': {'description': 'Altitude above sea level', 'units': 'm'},
            'FLs': {'description': 'Flight level', 'units': 'FL'},
            'rocs': {'description': 'Rate of climb/descent', 'units': 'm/s'},
            'flightTime': {'description': 'Flight time elapsed', 'units': 's'},
            'latitude': {'description': 'Latitude', 'units': 'degrees'},
            'longitude': {'description': 'Longitude', 'units': 'degrees'},
            'azimuth': {'description': 'Azimuth angle', 'units': 'degrees'},
            'heading': {'description': 'Aircraft heading', 'units': 'degrees'},
            'tas': {'description': 'True airspeed', 'units': 'm/s'},
            'groundSpeed': {'description': 'Ground speed', 'units': 'm/s'},
            'FL_weight': {
                'description': 'Flight level weight factor',
                'units': 'dimensionless',
            },
        }

        # Add all data variables with metadata
        for var_name, metadata in var_metadata.items():
            self.traj_data[var_name] = ('point', np.zeros(self.Ntot, dtype=np.float64))
            self.traj_data[var_name].attrs['description'] = metadata['description']
            self.traj_data[var_name].attrs['units'] = metadata['units']

        self.traj_data.attrs['title'] = 'Aircraft trajectory data'
        self.traj_data.attrs['description'] = '1D trajectory over mission points'

        if self.starting_mass < 0:
            self.calc_starting_mass(**kwargs)

        # Trajectory optimization
        if self.optimize_traj:
            # Will be implemented in a future version
            pass

        if self.iter_mass:
            self.mass_converged = False

            for m_iter in range(self.max_mass_iters):
                mass_res = self.fly_flight_iteration(**kwargs)

                # Keep the calculated trajectory if the mass is sufficiently small
                if abs(mass_res) < self.mass_iter_reltol:
                    self.mass_converged = True
                    break

                # Perform a `dumb` correction of the starting mass
                self.starting_mass = self.starting_mass - (mass_res * self.fuel_mass)

            if not self.mass_converged:
                print(
                    "Mass iteration failed to converge; final residual "
                    f"{mass_res * 100}% > {self.mass_iter_reltol * 100}%"
                )

        else:
            self.fly_flight_iteration(**kwargs)

    def fly_flight_iteration(self, **kwargs):
        """Function for running a single flight iteration. In non-weight-iterating
        mode, only runs once. `kwargs` used to pass in relevent optimization variables
        in applicable cases.

        Args:
            kwargs: Additional parameters needed by the specific type of trajectory
                being used.

        Returns:
            (float) Difference in fuel burned and calculated required fuel mass.
        """
        self.current_mass = self.starting_mass

        # Initialize all data variables to NaN
        for field in self.traj_data.data_vars:
            self.traj_data[field].values[:] = np.nan

        # Set initial values
        self.traj_data['flightTime'].values[0] = 0
        self.traj_data['acMass'].values[0] = self.starting_mass
        self.traj_data['fuelMass'].values[0] = self.fuel_mass
        self.traj_data['groundDist'].values[0] = 0
        self.traj_data['altitude'].values[0] = self.clm_start_altitude

        # Calculate lat, lon, heading of initial point
        # Get great circle trajectory in lat,lon points
        lon_dep, lat_dep, _ = self.dep_lon_lat_alt
        lon_arr, lat_arr, _ = self.arr_lon_lat_alt
        # lat_lon_trajectory = self.geod.npts(
        #                   lon_dep, lat_dep, lon_arr, lat_arr, self.Ntot)
        self.traj_data['latitude'].values[0] = lat_dep
        self.traj_data['longitude'].values[0] = lon_dep
        self.traj_data['azimuth'].values[0], _, _ = self.geod.inv(
            lon_dep, lat_dep, lon_arr, lat_arr
        )

        # Fly the climb, cruise, descent segments in order
        self.climb(**kwargs)
        self.cruise(**kwargs)
        self.descent(**kwargs)

        # Calculate weight residual normalized by fuel_mass
        fuelBurned = self.starting_mass - self.traj_data['acMass'].values[-1]
        mass_residual = (self.fuel_mass - fuelBurned) / self.fuel_mass

        return mass_residual

    ############################################################
    # UNIVERSAL TRAJECTORY FUNCTIONS - TO BE DEFINED PER MODEL #
    ############################################################
    def climb(self):
        pass

    def cruise(self):
        pass

    def descent(self):
        pass

    def calc_starting_mass(self):
        pass

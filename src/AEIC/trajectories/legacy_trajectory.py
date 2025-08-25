import numpy as np
from numpy.typing import NDArray

from AEIC.performance_model import PerformanceModel
from AEIC.trajectories.trajectory import Trajectory
from utils.helpers import filter_order_duplicates
from utils.units import FEET_TO_METERS, METERS_TO_FEET, NAUTICAL_MILES_TO_METERS
from utils.weather_utils import compute_ground_speed


class LegacyTrajectory(Trajectory):
    """Model for determining flight trajectories using the legacy method
    from AEIC v2. Contains all attributes listed in Trajectory plus the
    ones listed here.

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
        pctStepClm (float, optional): Climb step size as percentage of total climb
            altitude change. Defaults to 0.01.
        pctStepCrz (float, optional): Cruise step size as percentage of total cruise
            ground distance. Defaults to 0.01.
        pctStepDes (float, optional): Descent step size as percentage of total descent
            altitude change. Defaults to 0.01.
        fuel_LHV (float, optional): Lower heating value of the fuel used. Defaults to
            43.8e6.

    Attributes:
        crz_FLs (list[float]): Flight levels in the performance data bounding the
            constant altitude cruise FL.
        crz_FL_inds (list[int]): Indices of the flight levels bounding the constant
            altitude cruise FL.
        zero_roc_mask (NDArray[bool]): mask on the rate-of-climb data; True only where
            ROC = 0.
    """

    def __init__(
        self,
        ac_performance: PerformanceModel,
        mission,
        optimize_traj: bool,
        iterate_mass: bool,
        startMass: float = -1,
        pctStepClm: float = 0.01,
        pctStepCrz: float = 0.01,
        pctStepDes: float = 0.01,
        fuel_LHV: float = 43.8e6,
    ):
        super().__init__(
            ac_performance, mission, optimize_traj, iterate_mass, startMass=startMass
        )

        # Define discretization of each phase in steps as a percent of
        # the overall distance/altitude change
        self.pctStepClm = pctStepClm
        self.pctStepCrz = pctStepCrz
        self.pctStepDes = pctStepDes

        self.NClm = int(1 / self.pctStepClm + 1)
        self.NCrz = int(1 / self.pctStepCrz + 1)
        self.NDes = int(1 / self.pctStepDes + 1)
        self.Ntot = self.NClm + self.NCrz + self.NDes

        self.fuel_LHV = fuel_LHV

        # Climb defined as starting 3000' above airport
        self.clm_start_altitude = self.dep_lon_lat_alt[-1] + 3000.0 * FEET_TO_METERS

        # Max alt should be changed to meters
        max_alt = (
            ac_performance.model_info['General_Information']['max_alt_ft']
            * FEET_TO_METERS
        )

        # Check if starting altitude is above operating ceiling;
        # if true, set start altitude to
        # departure airport altitude
        if self.clm_start_altitude >= max_alt:
            self.clm_start_altitude = self.dep_lon_lat_alt[-1]

        # Cruise altitude is the operating ceiling - 7000 feet
        self.crz_start_altitude = max_alt - 7000.0 * FEET_TO_METERS

        # Ensure cruise altitude is above the starting altitude
        if self.crz_start_altitude < self.clm_start_altitude:
            self.crz_start_altitude = self.clm_start_altitude

        # Prevent flying above A/C ceiling (NOTE: this will only trigger due to random
        # variables not currently implemented)
        if self.crz_start_altitude > max_alt:
            self.crz_start_altitude = max_alt

        # In legacy trajectory, descent start altitude is equal to cruise altitude
        self.des_start_altitude = self.crz_start_altitude

        # Set descent altitude based on 3000' above arrival airport altitude;
        # clamp to A/C operating
        # ceiling if needed
        self.des_end_altitude = self.arr_lon_lat_alt[-1] + 3000.0 * FEET_TO_METERS
        if self.des_end_altitude >= max_alt:
            self.des_end_altitude = max_alt

        # Save relevant flight levels
        self.crz_FL = self.crz_start_altitude * METERS_TO_FEET / 100
        self.clm_FL = self.clm_start_altitude * METERS_TO_FEET / 100
        self.des_FL = self.des_start_altitude * METERS_TO_FEET / 100
        self.end_FL = self.des_end_altitude * METERS_TO_FEET / 100

        # Get the relevant bounding flight levels for cruise based on performance data
        self.__calc_crz_FLs()

        # Get the indices for 0-ROC performance
        self.__get_zero_roc_index()

    def climb(self):
        """Function called by ``fly_flight_iteration()`` to simulate climb"""
        dAlt = (self.crz_start_altitude - self.clm_start_altitude) / (self.NClm - 1)
        if dAlt < 0:
            raise ValueError(
                "Departure airport + 3000ft should not be higher"
                "than start of cruise point"
            )

        alts = np.linspace(self.clm_start_altitude, self.crz_start_altitude, self.NClm)
        self.traj_data['altitude'][0 : self.NClm] = alts

        self.__legacy_climb()

    def cruise(self):
        """Function called by ``fly_flight_iteration()`` to simulate cruise"""
        # Start cruise at end-of-climb position and mass (fuel flow, TAS will be
        # replaced)
        for field in self.traj_data.dtype.names:
            self.traj_data[field][self.NClm] = self.traj_data[field][self.NClm - 1]

        # Cruise at constant altitude
        self.traj_data['altitude'][self.NClm : self.NClm + self.NCrz] = (
            self.crz_start_altitude
        )

        descent_dist_approx = 18.23 * (self.des_start_altitude - self.des_end_altitude)

        if descent_dist_approx < 0:
            raise ValueError('Arrival airport should not be above cruise altitude')

        cruise_start_distance = self.traj_data['groundDist'][self.NClm - 1]
        cruise_dist_approx = (
            self.gc_distance * NAUTICAL_MILES_TO_METERS
            - cruise_start_distance
            - descent_dist_approx
        )

        # Cruise is discretized into ground distance steps
        cruise_end_distance = cruise_start_distance + cruise_dist_approx
        cruise_distance_values = np.linspace(
            cruise_start_distance, cruise_end_distance, self.NCrz
        )
        self.traj_data['groundDist'][self.NClm : self.NClm + self.NCrz] = (
            cruise_distance_values
        )

        # Get distance step size
        dGD = cruise_dist_approx / (self.NCrz - 1)

        self.__legacy_cruise(dGD)

    def descent(self):
        """Function called by ``fly_flight_iteration()`` to simulate descent"""
        # Start descent at end-of-cruise position and mass (fuel flow, TAS will be
        # replaced)
        for field in self.traj_data.dtype.names:
            self.traj_data[field][self.NClm + self.NCrz] = self.traj_data[field][
                self.NClm + self.NCrz - 1
            ]

        dAlt = (self.des_end_altitude - self.des_start_altitude) / (self.NDes)
        if dAlt > 0:
            raise ValueError(
                "Arrival airport + 3000ft should not be higher thanend of cruise point"
            )

        alts = np.linspace(self.des_start_altitude, self.des_end_altitude, self.NDes)
        startN = self.NClm + self.NCrz
        endN = startN + self.NDes
        self.traj_data['altitude'][startN:endN] = alts

        self.__legacy_descent()

    def calc_starting_mass(self):
        """Calculates the starting mass using AEIC v2 methods.
        Sets both starting mass and non-reserve/hold/divert fuel mass"""
        # Use the highest value of mass per AEIC v2 method
        mass_ind = [len(self.ac_performance.performance_table_cols[-1]) - 1]

        subset_performance = self.ac_performance.performance_table[
            np.ix_(
                self.crz_FL_inds,
                # axis 0: flight levels
                np.arange(self.ac_performance.performance_table.shape[1]),
                # axis 1: all TAS's
                np.where(self.zero_roc_mask)[0],
                # axis 2: ROC ≈ 0
                mass_ind,
                # axis 3: high mass value
            )
        ]

        non_zero_mask = np.any(subset_performance != 0, axis=(0, 2, 3))
        non_zero_perf = subset_performance[:, non_zero_mask, :, :]
        crz_tas = np.array(self.ac_performance.performance_table_cols[1])[non_zero_mask]

        # At this point, we should have a (2, 2, 1, 1)-shape matrix of
        # fuel flow in (FL, TAS, --, --)
        # where there should only be two non-0 values in the FL and TAS dimensions.
        # Isolate this matrix:
        if np.shape(non_zero_perf) != (2, 2, 1, 1):
            raise ValueError('Performance is overdefined for legacy methods')

        twoByTwoPerf = non_zero_perf[:, :, 0, 0]
        ff_mat = twoByTwoPerf[twoByTwoPerf != 0.0]
        if np.shape(ff_mat) != (2,):
            raise ValueError(
                "Mass estimation fuel flow matrix does not have the"
                "required dimensions (Expected: (2,); Recieved: "
                f"{np.shape(ff_mat)})"
            )

        # Now perform the necessary interpolations in TAS and fuel flow
        FL_weighting = (self.crz_FL - self.crz_FLs[0]) / (
            self.crz_FLs[1] - self.crz_FLs[0]
        )
        dfuelflow = ff_mat[1] - ff_mat[0]
        dTAS = crz_tas[1] - crz_tas[0]

        fuelflow = ff_mat[0] + dfuelflow * FL_weighting
        tas = crz_tas[0] + dTAS * FL_weighting

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

        # Empty mass per BADA-3 (low mass / 1.2)
        emptyMass = self.ac_performance.performance_table_cols[-1][0] / 1.2

        # Payload
        payloadMass = (
            self.ac_performance.model_info['General_Information']['max_payload_kg']
            * self.load_factor
        )

        # Fuel Needed (distance / velocity * fuel flow rate)
        approxTime = self.gc_distance * NAUTICAL_MILES_TO_METERS / tas
        fuelMass = approxTime * fuelflow

        # Reserve fuel (assumed 5%)
        reserveMass = fuelMass * 0.05

        # Diversion fuel per AEIC v2
        if approxTime / 60 > 180:  # > 180 minutes
            divertMass = 200.0 * NAUTICAL_MILES_TO_METERS / tas * fuelflow
            holdMass = 30 * 60 * tas  # 30 min; using cruise ff here
        else:
            divertMass = 100.0 * NAUTICAL_MILES_TO_METERS / tas * fuelflow
            holdMass = 45 * 60 * tas  # 30 min; using cruise ff here

        self.starting_mass = (
            emptyMass + payloadMass + fuelMass + reserveMass + divertMass + holdMass
        )

        # Limit to MTOM if overweight
        if self.starting_mass > self.ac_performance.performance_table_cols[-1][-1]:
            self.starting_mass = self.ac_performance.performance_table_cols[-1][-1]

        # Set fuel mass (for weight residual calculation)
        self.fuel_mass = fuelMass

    ###################
    # PRIVATE METHODS #
    ###################
    def __calc_crz_FLs(self):
        """Get the bounding cruise flight levels (for which data exists)"""
        # Get the two flight levels in data closest to the cruise FL
        self.crz_FL_inds = self.__search_flight_levels_ind(self.crz_FL)
        self.crz_FLs = np.array(self.ac_performance.performance_table_cols[0])[
            self.crz_FL_inds
        ]

    def __search_flight_levels_ind(self, FL: float) -> list[float]:
        """Searches the valid flight levels in the performance model for the indices
        bounding a known FL value.

        Args:
            FL (float): Flight level of interest.

        Returns:
            (list[int]) List containing the indices of the FLs in the performance data
                that bound the given FL.
        """
        FL_ind_high = np.searchsorted(self.ac_performance.performance_table_cols[0], FL)

        if FL_ind_high == 0:
            raise ValueError(
                f"Aircraft is trying to fly below minimum cruise altitude(FL {FL:.2f})"
            )
        if FL_ind_high == len(self.ac_performance.performance_table_cols[0]):
            raise ValueError(
                f"Aircraft is trying to fly above maximum cruise altitude(FL {FL:.2f})"
            )

        FL_ind_low = FL_ind_high - 1
        FLs = [FL_ind_low, FL_ind_high]
        return FLs

    def __search_TAS_ind(self, tas: float) -> list[float]:
        """Searches the valid tas in the performance model for the indices bounding a
        known tas value.

        Args:
            tas (float): TAS value of interest.

        Returns:
            (list[int]) List containing the indices of the TAS values in performance
                data that bound the given TAS.
        """
        tas_ind_high = np.searchsorted(
            self.ac_performance.performance_table_cols[1], tas
        )

        if tas_ind_high == 0:
            raise ValueError(
                f"Aircraft is trying to fly below minimum cruise altitude(FL {tas:.2f})"
            )
        if tas_ind_high == len(self.ac_performance.performance_table_cols[1]):
            raise ValueError(
                f"Aircraft is trying to fly above maximum cruise altitude(FL {tas:.2f})"
            )

        tas_ind_low = tas_ind_high - 1
        tass = [tas_ind_low, tas_ind_high]
        return tass

    def __search_mass_ind(self, mass: float) -> list[float]:
        """Searches the valid mass values in the performance model for the indices
        bounding a known mass value.

        Args:
            mass (float): Mass value of interest.

        Returns:
            (list[int]) List containing the indices of the TAS values in performance
                data that bound the given TAS.
        """
        mass_ind_high = np.searchsorted(
            self.ac_performance.performance_table_cols[-1], mass
        )

        if mass_ind_high == 0:
            raise ValueError('Aircraft is trying to fly below minimum mass')
        if mass_ind_high == len(self.ac_performance.performance_table_cols[0]):
            raise ValueError('Aircraft is trying to fly above maximum mass')

        mass_ind_low = mass_ind_high - 1
        masses = [mass_ind_low, mass_ind_high]
        return masses

    def __get_zero_roc_index(self, roc_zero_tol: float = 1e-6) -> None:
        """Get the index along the ROC axis of performance where ROC == 0

        Args:
            roc_zero_tol (float, optional): Tolerance at which ROC is considered to be
                0. Defaults to 1e-6.
        """
        self.zero_roc_mask = (
            np.abs(np.array(self.ac_performance.performance_table_cols[2]))
            < roc_zero_tol
        )

    def __calc_FL_interp_vals(
        self, i: int, alt: float, roc_perf: NDArray[np.float64]
    ) -> tuple[float, ...]:
        """Computes the state values that depend only on flight level. These include
        true airspeed (TAS) and fuel flow (ff). Rate of climb (roc) is also only
        dependent on FL in descent.

        Args:
            i (int): Index of the current point in the ``traj_data`` array.
            alt (float): Altitude at which to calculate interpolated values.
            roc_perf (NDArray[np.float64]): Performance data restricted to the relevant
                rates of climb.

        Returns:
            Tuple[float]: TAS, fuel flow rate, and rate of climb at the specified
                altitude
        """
        FL = alt * METERS_TO_FEET / 100
        self.traj_data['FLs'][i] = FL
        FL_inds = self.__search_flight_levels_ind(FL)
        bounding_fls = np.array(self.ac_performance.performance_table_cols[0])[FL_inds]

        # Construct interpolation weightings
        fl_weighting = (FL - bounding_fls[0]) / (bounding_fls[1] - bounding_fls[0])
        self.traj_data['FL_weight'][i] = fl_weighting

        # Filter to bounding flight levels
        pos_roc_fl_reduced_perf = roc_perf[
            np.ix_(
                FL_inds,  # axis 0: flight levels
                np.arange(roc_perf.shape[1]),  # axis 1: all TAS's
                np.arange(roc_perf.shape[2]),  # axis 2: all positive ROC
                np.arange(roc_perf.shape[3]),  # axis 3: mass value
            )
        ]

        # The the collapsed indices and values of all non-0 fuel flow
        # and TAS values in the filtered performance data
        non_zero_ff_inds = np.nonzero(pos_roc_fl_reduced_perf)
        non_zero_ff_vals = pos_roc_fl_reduced_perf[non_zero_ff_inds]

        non_zero_tas_inds = non_zero_ff_inds[1]
        non_zero_tas_vals = np.array(self.ac_performance.performance_table_cols[1])[
            non_zero_tas_inds
        ]

        # ROC will only be valid in descent
        non_zero_roc_inds = non_zero_ff_inds[2]
        non_zero_roc_vals = np.array(self.ac_performance.performance_table_cols[2])[
            non_zero_roc_inds
        ]

        # Remove duplicate entries; we should have 2 entries in each
        # corresponding to the two bounding flight levels
        tas_vals = filter_order_duplicates(non_zero_tas_vals)
        ff_vals = filter_order_duplicates(non_zero_ff_vals)

        roc_vals = filter_order_duplicates(non_zero_roc_vals)

        # Interpolate to get TAS and fuel flow
        if len(tas_vals) == 1:
            tas_interp = tas_vals[0]
        else:
            tas_interp = tas_vals[0] + fl_weighting * (tas_vals[1] - tas_vals[0])

        if len(ff_vals) == 1:
            ff_interp = ff_vals[0]
        else:
            ff_interp = ff_vals[0] + fl_weighting * (ff_vals[1] - ff_vals[0])

        if len(roc_vals) == 1:
            roc_interp = roc_vals[0]
        else:
            roc_interp = roc_vals[0] + fl_weighting * (roc_vals[1] - roc_vals[0])

        return tas_interp, ff_interp, roc_interp

    def __calc_tas_crz(
        self, i: int, alt: float, roc_perf: NDArray[np.float64]
    ) -> tuple[float, ...]:
        """Computes the TAS that depend only on flight level for cruise

        Args:
            i (int): Index of current point in ``traj_data``.
            alt (float): Current altitude.
            roc_perf (NDArray[np.float64]): Performance data restricted to 0 rate of
                climb.

        Returns:
            Tuple[float]: Interpolated TAS and weighting used in linear interpolation.
        """
        FL = alt * METERS_TO_FEET / 100
        self.traj_data['FLs'][i] = FL

        # Construct interpolation weightings
        fl_weighting = (FL - self.crz_FLs[0]) / (self.crz_FLs[1] - self.crz_FLs[0])

        # The the collapsed indices and values of all non-0 fuel flow
        # and TAS values in the filtered performance data
        non_zero_ff_inds = np.nonzero(roc_perf)

        non_zero_tas_inds = non_zero_ff_inds[1]
        non_zero_tas_vals = np.array(self.ac_performance.performance_table_cols[1])[
            non_zero_tas_inds
        ]

        # Remove duplicate entries; we should have 2 entries in each
        # corresponding to the two bounding flight levels
        tas_vals = filter_order_duplicates(non_zero_tas_vals)

        # Interpolate to get TAS
        if len(tas_vals) == 1:
            tas_interp = tas_vals[0]
        else:
            tas_interp = tas_vals[0] + fl_weighting * (tas_vals[1] - tas_vals[0])

        return tas_interp, fl_weighting

    def __calc_roc_climb(
        self,
        i: int,
        FL: float,
        seg_start_mass: float,
        roc_perf: NDArray[np.float64],
        rocs: NDArray[np.float64],
    ) -> float:
        """Calculates rate of climb (roc) given flight level and mass given a
        subset of overall performance data (limited to roc > 0 or roc < 0 in
        ``roc_perf``).

        Args:
            i (int): Index of current point in ``traj_data``.
            FL (float): Current flight level.
            seg_start_mass (float): Starting mass of the climb segment.
            roc_perf (NDArray[np.float64]): Performance data (fuel flow rate) restricted
                to either positive or negative rate of climb.
            rocs (NDArray[np.float64]): Unfiltered list of rate of climb values.

        Returns:
            float: Rate of climb at the point of interest.
        """
        # Get bounding flight levels
        FL_inds = self.__search_flight_levels_ind(FL)
        bounding_fls = np.array(self.ac_performance.performance_table_cols[0])[FL_inds]

        # Get bounding mass values
        mass_inds = self.__search_mass_ind(seg_start_mass)
        bounding_mass = np.array(self.ac_performance.performance_table_cols[3])[
            mass_inds
        ]

        # Filter to bounding values
        pos_roc_reduced_perf = roc_perf[
            np.ix_(
                FL_inds,  # axis 0: flight levels
                np.arange(roc_perf.shape[1]),  # axis 1: all TAS's
                np.arange(roc_perf.shape[2]),  # axis 2: all zero ROC
                mass_inds,  # axis 3: mass value
            )
        ]

        non_zero_ff_inds = np.nonzero(pos_roc_reduced_perf)
        rocs = rocs[non_zero_ff_inds[2]]
        fls = bounding_fls[non_zero_ff_inds[0]]
        masses = bounding_mass[non_zero_ff_inds[3]]

        # Prepare ROC matrix
        roc_mat = np.full((len(bounding_fls), len(bounding_mass)), np.nan)

        # Fill ROC matrix
        for kk in range(len(rocs)):
            ii = np.where(bounding_fls == fls[kk])[0][0]
            jj = np.where(bounding_mass == masses[kk])[0][0]
            roc_mat[ii, jj] = rocs[kk]

        # Get the precomputed flight-level interpolation weighting
        fl_weight = self.traj_data['FL_weight'][i]

        # Calculate the mass-based interpolation weighting
        mass_weight = (seg_start_mass - bounding_mass[0]) / (
            bounding_mass[1] - bounding_mass[0]
        )

        # Perform bilinear interpolation
        roc_1 = roc_mat[0, 0] + (roc_mat[1, 0] - roc_mat[0, 0]) * fl_weight
        roc_2 = roc_mat[0, 1] + (roc_mat[1, 1] - roc_mat[0, 1]) * fl_weight

        roc = roc_1 + (roc_2 - roc_1) * mass_weight

        return roc

    def __calc_ff_cruise(
        self, i: int, seg_start_mass: float, crz_perf: NDArray[np.float64]
    ) -> float:
        """Calculates fuel flow rate in cruise using given flight level and
        mass given a subset of overall performance data (limited to roc = 0).

        Args:
            i (int): Index of current point in ``traj_data``.
            seg_start_mass (float): Starting mass of the climb segment.
            crz_perf (NDArray[np.float64]): Performance data limited to ROC=0.

        Returns:
            float: Cruise fuel flow rate at the point of interest.
        """
        # Get bounding mass values
        mass_inds = self.__search_mass_ind(seg_start_mass)

        bounding_mass = np.array(self.ac_performance.performance_table_cols[3])[
            mass_inds
        ]

        # Filter to bounding values
        crz_perf_reduced = crz_perf[
            np.ix_(
                np.arange(crz_perf.shape[0]),  # axis 0: flight levels
                np.arange(crz_perf.shape[1]),  # axis 1: all TAS's
                np.arange(crz_perf.shape[2]),  # axis 2: all zero ROC
                mass_inds,  # axis 3: mass value
            )
        ]

        non_zero_ff_inds = np.nonzero(crz_perf_reduced)
        ffs = crz_perf_reduced[non_zero_ff_inds]
        fls = self.crz_FLs[non_zero_ff_inds[0]]
        masses = bounding_mass[non_zero_ff_inds[3]]

        # Prepare FF matrix
        ff_mat = np.full((len(self.crz_FLs), len(bounding_mass)), np.nan)

        # Fill FF matrix
        for kk in range(len(ffs)):
            ii = np.where(self.crz_FLs == fls[kk])[0][0]
            jj = np.where(bounding_mass == masses[kk])[0][0]
            ff_mat[ii, jj] = ffs[kk]

        # Get flight level and mass weights for interpolation
        fl_weight = self.traj_data['FL_weight'][i]
        mass_weight = (seg_start_mass - bounding_mass[0]) / (
            bounding_mass[1] - bounding_mass[0]
        )

        # Perform bilinear interpolation
        ff_1 = ff_mat[0, 0] + (ff_mat[1, 0] - ff_mat[0, 0]) * fl_weight
        ff_2 = ff_mat[0, 1] + (ff_mat[1, 1] - ff_mat[0, 1]) * fl_weight

        ff = ff_1 + (ff_2 - ff_1) * mass_weight
        return ff

    def __legacy_climb(self) -> None:
        """Computes state over the climb segment using AEIC v2 methods
        based on BADA-3 formulas.
        """
        # Create a mask for ROC limiting to only positive values (climb)
        pos_roc_mask = np.array(self.ac_performance.performance_table_cols[2]) > 0

        # Convert ROC mask to the indices of positive ROC
        roc_inds = np.where(pos_roc_mask)[0]
        pos_rocs = np.array(self.ac_performance.performance_table_cols[2])[roc_inds]

        # Filter performance data to positive ROC
        pos_roc_perf = self.ac_performance.performance_table[
            np.ix_(
                np.arange(self.ac_performance.performance_table.shape[0]),
                # axis 0: flight levels
                np.arange(self.ac_performance.performance_table.shape[1]),
                # axis 1: all TAS's
                np.where(pos_roc_mask)[0],
                # axis 2: all positive ROC
                np.arange(self.ac_performance.performance_table.shape[3]),
                # axis 3: mass value
            )
        ]

        # We first compute the instantaneous data at each flight level
        # to avoid repeat calculations.
        # In AEIC v2 fuel flow and TAS are only dependent on flight level in climb.
        for i in range(0, self.NClm):
            alt = self.traj_data['altitude'][i]
            tas_interp, ff_interp, _ = self.__calc_FL_interp_vals(i, alt, pos_roc_perf)
            self.traj_data['fuelFlow'][i] = ff_interp
            self.traj_data['tas'][i] = tas_interp

        # Now we get rate of climb by running the flight
        for i in range(0, self.NClm - 1):
            FL = self.traj_data['FLs'][i]
            tas = self.traj_data['tas'][i]
            ff = self.traj_data['fuelFlow'][i]
            seg_start_mass = self.traj_data['acMass'][i]

            # Calculate rate of climb
            roc = self.__calc_roc_climb(i, FL, seg_start_mass, pos_roc_perf, pos_rocs)
            self.traj_data['rocs'][i] = roc

            # Calculate the forward true airspeed (will be used for ground speed)
            fwd_tas = np.sqrt(tas**2 - roc**2)

            # Get time to complete alititude change segment and total fuel burned
            segment_time = (
                self.traj_data['altitude'][i + 1] - self.traj_data['altitude'][i]
            ) / roc
            segment_fuel = ff * segment_time
            self.traj_data['groundSpeed'][i], self.traj_data['heading'][i], u, v = (
                compute_ground_speed(
                    lon=self.traj_data['latitude'][i],
                    lat=self.traj_data['latitude'][i],
                    az=self.traj_data['azimuth'][i],
                    alt_ft=FL * 100,
                    tas_ms=fwd_tas,
                    weather_data=None,
                )
            )

            # Calculate distance along route travelled
            dist = self.traj_data['groundSpeed'][i] * segment_time

            self.traj_data['longitude'][i + 1], self.traj_data['latitude'][i + 1], _ = (
                self.geod.fwd(
                    self.traj_data['longitude'][i],
                    self.traj_data['latitude'][i],
                    self.traj_data['azimuth'][i],
                    dist,
                )
            )

            lon_arr, lat_arr, _ = self.arr_lon_lat_alt
            self.traj_data['azimuth'][i + 1], _, _ = self.geod.inv(
                self.traj_data['longitude'][i],
                self.traj_data['latitude'][i],
                lon_arr,
                lat_arr,
            )
            # Account for acceleration/deceleration over
            # the segment using end-of-segment tas
            tas_end = self.traj_data['tas'][i + 1]
            kinetic_energy_chg = 1 / 2 * seg_start_mass * (tas_end**2 - tas**2)

            # Calculate fuel required for acceleration
            # NOTE: I have no idea where AEIC v2 got the efficiency of 0.15 from
            accel_fuel = kinetic_energy_chg / (self.fuel_LHV) / 0.15

            segment_fuel += accel_fuel

            # Update the state vector
            self.traj_data['fuelMass'][i + 1] = (
                self.traj_data['fuelMass'][i] - segment_fuel
            )
            self.traj_data['acMass'][i + 1] = self.traj_data['acMass'][i] - segment_fuel
            self.traj_data['groundDist'][i + 1] = self.traj_data['groundDist'][i] + dist
            self.traj_data['flightTime'][i + 1] = (
                self.traj_data['flightTime'][i] + segment_time
            )

    def __legacy_cruise(self, dGD: float) -> None:
        """Computes state over cruise segment using AEIC v2 methods
        based on BADA-3 formulas"""
        subset_performance = self.ac_performance.performance_table[
            np.ix_(
                self.crz_FL_inds,
                # axis 0: flight levels
                np.arange(self.ac_performance.performance_table.shape[1]),
                # axis 1: all TAS's
                np.where(self.zero_roc_mask)[0],
                # axis 2: ROC ≈ 0
                np.arange(self.ac_performance.performance_table.shape[3]),
                # axis 3: all mass values
            )
        ]

        # TAS in cruise is only dependent on flight level
        tas_interp, fl_weight = self.__calc_tas_crz(
            self.NClm, self.crz_start_altitude, subset_performance
        )
        self.traj_data['tas'][self.NClm : self.NClm + self.NCrz] = tas_interp
        self.traj_data['rocs'][self.NClm : self.NClm + self.NCrz] = 0
        self.traj_data['FL_weight'][self.NClm : self.NClm + self.NCrz] = fl_weight

        # Get fuel flow, ground speed, etc. for cruise segments
        for i in range(self.NClm, self.NClm + self.NCrz - 1):
            self.traj_data['groundSpeed'][i], self.traj_data['heading'][i], _, _ = (
                compute_ground_speed(
                    lon=self.traj_data['latitude'][i],
                    lat=self.traj_data['latitude'][i],
                    az=self.traj_data['azimuth'][i],
                    alt_ft=self.crz_FL * 100,
                    tas_ms=self.traj_data['tas'][i],
                    weather_data=None,
                )
            )

            # Calculate time required to fly the segment
            segment_time = dGD / self.traj_data['groundSpeed'][i]

            self.traj_data['longitude'][i + 1], self.traj_data['latitude'][i + 1], _ = (
                self.geod.fwd(
                    self.traj_data['longitude'][i],
                    self.traj_data['latitude'][i],
                    self.traj_data['azimuth'][i],
                    dGD,
                )
            )
            lon_arr, lat_arr, _ = self.arr_lon_lat_alt
            self.traj_data['azimuth'][i + 1], _, _ = self.geod.inv(
                self.traj_data['longitude'][i],
                self.traj_data['latitude'][i],
                lon_arr,
                lat_arr,
            )

            # Get fuel flow rate based on FL and mass interpolation
            ff = self.__calc_ff_cruise(
                i, self.traj_data['acMass'][i], subset_performance
            )

            # Calculate fuel burn in [kg] over the segment
            segment_fuel = ff * segment_time

            # Set aircraft state values
            self.traj_data['fuelFlow'][i + 1] = ff
            self.traj_data['fuelMass'][i + 1] = (
                self.traj_data['fuelMass'][i] - segment_fuel
            )
            self.traj_data['acMass'][i + 1] = self.traj_data['acMass'][i] - segment_fuel

            self.traj_data['flightTime'][i + 1] = (
                self.traj_data['flightTime'][i] + segment_time
            )

    def __legacy_descent(self) -> None:
        """Computes state over the descent segment using AEIC v2
        methods based on BADA-3 formulas"""
        startN = self.NClm + self.NCrz
        endN = startN + self.NDes

        # Create a mask for ROC limiting to only positive values (climb)
        neg_roc_mask = np.array(self.ac_performance.performance_table_cols[2]) < 0

        # Convert ROC mask to the indices of positive ROC
        # roc_inds = np.where(neg_roc_mask)[0]
        # neg_rocs = np.array(self.ac_performance.performance_table_cols[2])[roc_inds]

        # Filter performance data to positive ROC
        neg_roc_perf = self.ac_performance.performance_table[
            np.ix_(
                np.arange(self.ac_performance.performance_table.shape[0]),
                # axis 0: flight levels
                np.arange(self.ac_performance.performance_table.shape[1]),
                # axis 1: all TAS's
                np.where(neg_roc_mask)[0],
                # axis 2: all positive ROC
                np.arange(self.ac_performance.performance_table.shape[3]),
                # axis 3: mass value
            )
        ]

        # We first compute the instantaneous data at each flight level
        # to avoid repeat calculations.
        # In AEIC v2 fuel flow and TAS are only dependent on flight level.
        for i in range(startN, endN):
            alt = self.traj_data['altitude'][i]
            tas_interp, ff_interp, roc_interp = self.__calc_FL_interp_vals(
                i, alt, neg_roc_perf
            )
            self.traj_data['fuelFlow'][i] = ff_interp
            self.traj_data['tas'][i] = tas_interp
            self.traj_data['rocs'][i] = roc_interp

        # Now we calculate segment level info by running the flight
        for i in range(startN, endN - 1):
            tas = self.traj_data['tas'][i]
            ff = self.traj_data['fuelFlow'][i]
            roc = self.traj_data['rocs'][i]
            seg_start_mass = self.traj_data['acMass'][i]

            # Calculate the forward true airspeed (will be used for ground speed)
            fwd_tas = np.sqrt(tas**2 - roc**2)

            # Get time to complete alititude change segment and total fuel burned
            segment_time = (
                self.traj_data['altitude'][i + 1] - self.traj_data['altitude'][i]
            ) / roc
            segment_fuel = ff * segment_time

            self.traj_data['groundSpeed'][i], self.traj_data['heading'][i], _, _ = (
                compute_ground_speed(
                    lon=self.traj_data['latitude'][i],
                    lat=self.traj_data['latitude'][i],
                    az=self.traj_data['azimuth'][i],
                    alt_ft=self.traj_data['altitude'][i] * METERS_TO_FEET,
                    tas_ms=fwd_tas,
                    weather_data=None,
                )
            )

            # Calculate distance along route travelled
            dist = self.traj_data['groundSpeed'][i] * segment_time

            self.traj_data['longitude'][i + 1], self.traj_data['latitude'][i + 1], _ = (
                self.geod.fwd(
                    self.traj_data['longitude'][i],
                    self.traj_data['latitude'][i],
                    self.traj_data['azimuth'][i],
                    dist,
                )
            )
            lon_arr, lat_arr, _ = self.arr_lon_lat_alt
            self.traj_data['azimuth'][i + 1], _, _ = self.geod.inv(
                self.traj_data['longitude'][i],
                self.traj_data['latitude'][i],
                lon_arr,
                lat_arr,
            )

            # Account for acceleration/deceleration over the segment
            # using end-of-segment tas
            tas_end = self.traj_data['tas'][i + 1]
            kinetic_energy_chg = 1 / 2 * seg_start_mass * (tas_end**2 - tas**2)

            # Calculate fuel required for acceleration
            # NOTE: I have no idea where AEIC v2 got the efficiency of 0.15 from
            accel_fuel = kinetic_energy_chg / (self.fuel_LHV) / 0.15

            segment_fuel += accel_fuel

            # We cannot gain fuel by decelerating in a conventional fuel A/C
            if segment_fuel < 0:
                segment_fuel = 0

            # Update the state vector
            self.traj_data['fuelMass'][i + 1] = (
                self.traj_data['fuelMass'][i] - segment_fuel
            )
            self.traj_data['acMass'][i + 1] = self.traj_data['acMass'][i] - segment_fuel
            self.traj_data['groundDist'][i + 1] = self.traj_data['groundDist'][i] + dist
            self.traj_data['flightTime'][i + 1] = (
                self.traj_data['flightTime'][i] + segment_time
            )

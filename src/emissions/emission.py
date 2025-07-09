# Emissions class
import tomllib

import numpy as np

from AEIC.performance_model import PerformanceModel
from AEIC.trajectories.trajectory import Trajectory
from utils import file_location
from utils.consts import R_air, kappa
from utils.standard_atmosphere import (
    pressure_at_altitude_isa_bada4,
    temperature_at_altitude_isa_bada4,
)
from utils.standard_fuel import get_SLS_equivalent_fuel_flow, get_thrust_cat

from .APU_emissions import get_APU_emissions
from .EI_CO2 import EI_CO2
from .EI_H2O import EI_H2O
from .EI_HCCO import EI_HCCO
from .EI_NOx import BFFM2_EINOx, NOx_speciation
from .EI_PMnvol import PMnvol_MEEM, calculate_PMnvolEI_scope11
from .EI_PMvol import EI_PMvol_NEW
from .EI_SOx import EI_SOx


class Emission:
    """
    Model for determining and aggregating flight emissions across all mission segments,
    including cruise trajectory, LTO (Landing and Take-Off), APU, and GSE emissions,
    as well as lifecycle CO2 adjustments.
    """

    def __init__(
        self, ac_performance: PerformanceModel, trajectory: Trajectory, EDB_data: bool
    ):
        """
        Initialize emissions model:

        Parameters
        ----------
        ac_performance : PerformanceModel
            Aircraft performance object containing climb/cruise/descent
            performance and LTO data.
        trajectory : Trajectory
            Flight trajectory for mission object with altitude, speed,
            and fuel mass time series.
        EDB_data : bool
            Flag indicating whether to use EDB tabulated data (True)
            or user specified LTO data (False).
        fuel_file : str
            Path to TOML file containing fuel properties
        """

        # Load fuel properties from TOML
        fuel_file_loc = file_location(ac_performance.config['fuel_file'])
        with open(fuel_file_loc, 'rb') as f:
            self.fuel = tomllib.load(f)

        # Unpack trajectory lengths: total, climb, cruise, descent points
        self.Ntot = trajectory.Ntot
        self.NClm = trajectory.NClm
        self.NCrz = trajectory.NCrz
        self.NDes = trajectory.NDes

        # Flag to use performance model for all segments or just cruise
        self.traj_emissions_all = ac_performance.config['climb_descent_usage']
        # Mode for PMnvol emissions in LTO
        self.pmnvol_mode = ac_performance.config['nvpm_method']

        # Pre-allocate structured arrays for emission indices per point
        self.emission_indices = np.empty((), dtype=self.__emission_dtype(self.Ntot))

        # Pre-allocate LTO emissions arrays
        self.LTO_emission_indices = np.empty((), dtype=self.__emission_dtype(4))
        self.LTO_emissions_g = np.empty((), dtype=self.__emission_dtype(4))

        # Pre-allocate APU and GSE emissions arrays (single-mode shapes)
        self.APU_emission_indices = np.empty((), dtype=self.__emission_dtype(1))
        self.APU_emissions_g = np.empty((), dtype=self.__emission_dtype(1))
        self.GSE_emissions_g = np.empty((), dtype=self.__emission_dtype(1))

        # Storage for pointwise (segment) emissions and summed totals
        self.pointwise_emissions_g = np.empty(
            (), dtype=self.__emission_dtype(self.Ntot)
        )
        self.summed_emission_g = np.empty((), dtype=self.__emission_dtype(1))

        # Compute fuel burn per segment from fuelMass time series
        fuel_mass = trajectory.traj_data['fuelMass']
        fuel_burn = np.zeros_like(fuel_mass)
        # Difference between sequential mass values for ascent segments
        fuel_burn[1:] = fuel_mass[:-1] - fuel_mass[1:]
        self.fuel_burn_per_segment = fuel_burn

        # Calculate cruise trajectory emissions (CO2, H2O, SOx, NOx, HC, CO, PM)
        self.get_trajectory_emissions(trajectory, ac_performance, EDB_data=EDB_data)

        # Calculate LTO emissions for ground and approach/climb modes
        self.get_LTO_emissions(ac_performance, nvpm_method=self.pmnvol_mode)

        # Compute APU emissions based on LTO results and EDB parameters
        (self.APU_emission_indices, self.APU_emissions_g, apu_fuel_burn) = (
            get_APU_emissions(
                self.APU_emission_indices,
                self.APU_emissions_g,
                self.LTO_emission_indices,
                ac_performance.APU_data,
                self.LTO_noProp,
                self.LTO_no2Prop,
                self.LTO_honoProp,
            )
        )
        self.total_fuel_burn += apu_fuel_burn

        # Compute Ground Service Equipment (GSE) emissions based on WNSF type
        self.get_GSE_emissions(
            ac_performance.model_info['General_Information']['aircraft_class']
        )

        # Sum all emission contributions: trajectory + LTO + APU + GSE
        self.sum_total_emissions()

        # Add lifecycle CO2 emissions to total
        self.get_lifecycle_emissions(self.fuel, trajectory)

    def sum_total_emissions(self):
        """
        Aggregate emissions (g) across all sources into summed_emission_g.
        Sums pointwise trajectory, LTO, APU, and GSE emissions for each species.
        """
        for field in self.summed_emission_g.dtype.names:
            self.summed_emission_g[field] = (
                np.sum(self.pointwise_emissions_g[field])
                + np.sum(self.LTO_emissions_g[field])
                + self.APU_emissions_g[field]
                + self.GSE_emissions_g[field]
            )

    def get_trajectory_emissions(self, trajectory, ac_performance, EDB_data=True):
        """
        Calculate emission indices (g/species per kg fuel) for each flight segment.

        Parameters
        ----------
        trajectory : Trajectory
            Contains altitudes, speeds, and fuel flows for each time step.
        ac_performance : PerformanceModel
            Provides EDB or LTO data matrices for EI lookup.
        EDB_data : bool, optional
            Use tabulated EDB emissions (True) or LTO-mode data (False).
        """
        # Determine start/end indices for climb-cruise-descent or full mission
        if self.traj_emissions_all:
            i_start, i_end = 0, self.Ntot
        else:
            i_start, i_end = 0, -self.NDes

        # --- Compute CO2 and H2O emission indices for cruise ---
        self.emission_indices['CO2'][i_start:i_end], _ = EI_CO2(self.fuel)
        self.emission_indices['H2O'][i_start:i_end] = EI_H2O(self.fuel)

        # --- Compute SOx emission indices ---
        (
            self.emission_indices['SO2'][i_start:i_end],
            self.emission_indices['SO4'][i_start:i_end],
        ) = EI_SOx(self.fuel)

        # Select LTO EI arrays depending on EDB_data flag
        if EDB_data:
            lto_co_ei_array = np.array(ac_performance.EDB_data['CO_EI_matrix'])
            lto_hc_ei_array = np.array(ac_performance.EDB_data['HC_EI_matrix'])
            lto_nox_ei_array = np.array(ac_performance.EDB_data['NOX_EI_matrix'])
            lto_ff_array = np.array(ac_performance.EDB_data['fuelflow_KGperS'])
        else:
            # Extract HC, CO, NOx, and fuel-flow from thrust_settings dict
            settings = ac_performance.LTO_data['thrust_settings'].values()
            lto_co_ei_array = np.array([mode['CO_EI'] for mode in settings])
            lto_hc_ei_array = np.array([mode['HC_EI'] for mode in settings])
            lto_nox_ei_array = np.array([mode['NOX_EI'] for mode in settings])
            lto_ff_array = np.array([mode['FUEL_KGs'] for mode in settings])

        # --- Compute NOx, NO, NO2, HONO indices via BFFM2 model ---
        flight_alts = trajectory.traj_data['altitude'][i_start:i_end]
        flight_temps = temperature_at_altitude_isa_bada4(flight_alts)
        flight_pressures = pressure_at_altitude_isa_bada4(flight_alts)
        mach_number = trajectory.traj_data['tas'][i_start:i_end] / np.sqrt(
            kappa * R_air * flight_temps
        )
        sls_equiv_fuel_flow = get_SLS_equivalent_fuel_flow(
            trajectory.traj_data['fuelFlow'][i_start:i_end],
            flight_pressures,
            flight_temps,
            mach_number,
            ac_performance.model_info['General_Information']['n_eng'],
        )

        (
            self.emission_indices['NOx'][i_start:i_end],
            self.emission_indices['NO'][i_start:i_end],
            self.emission_indices['NO2'][i_start:i_end],
            self.emission_indices['HONO'][i_start:i_end],
            *_,
        ) = BFFM2_EINOx(
            sls_equiv_fuel_flow=sls_equiv_fuel_flow,
            NOX_EI_matrix=lto_nox_ei_array,
            fuelflow_performance=lto_ff_array,
            Pamb=flight_pressures,
            Tamb=flight_temps,
        )

        # --- Compute HC and CO indices --
        self.emission_indices['HC'][i_start:i_end] = EI_HCCO(
            sls_equiv_fuel_flow,
            lto_hc_ei_array,
            lto_ff_array,
            Tamb=flight_temps,
            Pamb=flight_pressures,
            cruiseCalc=True,
        )
        self.emission_indices['CO'][i_start:i_end] = EI_HCCO(
            sls_equiv_fuel_flow,
            lto_co_ei_array,
            lto_ff_array,
            Tamb=flight_temps,
            Pamb=flight_pressures,
            cruiseCalc=True,
        )

        # --- Compute volatile organic PM and organic carbon ---
        thrustCat = get_thrust_cat(
            trajectory.traj_data['fuelFlow'][i_start:i_end],
            lto_ff_array,
            cruiseCalc=True,
        )
        (
            self.emission_indices['PMvol'][i_start:i_end],
            self.emission_indices['OCic'][i_start:i_end],
        ) = EI_PMvol_NEW(trajectory.traj_data['fuelFlow'][i_start:i_end], thrustCat)

        # --- Compute black carbon indices via MEEM ---
        (
            self.emission_indices['PMnvolGMD'][i_start:i_end],
            self.emission_indices['PMnvol'][i_start:i_end],
            self.emission_indices['PMnvolN'][i_start:i_end],
        ) = PMnvol_MEEM(
            ac_performance.EDB_data,
            flight_alts,
            flight_temps,
            flight_pressures,
            mach_number,
        )

        self.total_fuel_burn = np.sum(self.fuel_burn_per_segment[i_start:i_end])
        # Multiply each index by fuel burn per segment to get g emissions/time-step
        for field in self.pointwise_emissions_g.dtype.names:
            self.pointwise_emissions_g[field][i_start:i_end] = (
                self.emission_indices[field][i_start:i_end]
                * self.fuel_burn_per_segment[i_start:i_end]
            )

    def get_LTO_emissions(self, ac_performance, EDB_LTO=True, nvpm_method="SCOPE11"):
        """
        Compute Landing-and-Takeoff cycle emission indices and quantities.

        Parameters
        ----------
        ac_performance : PerformanceModel
            Provides EDB or LTO tabular data for EI values.
        EDB_LTO : bool, optional
            Use EDB data for LTO (True) or thrust_settings (False).
        nvpm_method : str, optional
            Method for number-based PM emissions ('SCOPE11', 'foa3', etc.).
        """

        # Standard TIM durations
        # https://www.icao.int/environmental-protection/Documents/EnvironmentalReports/2016/ENVReport2016_pg73-74.pdf
        TIM_TakeOff = 0.7 * 60
        TIM_Climb = 2.2 * 60
        TIM_Approach = 4.0 * 60
        TIM_Taxi = 26.0 * 60

        # Assemble durations array for modes
        if self.traj_emissions_all:
            TIM_Climb = 0.0
            TIM_Approach = 0.0
            TIM_LTO = np.array([TIM_Taxi, TIM_Approach, TIM_Climb, TIM_TakeOff])
        else:
            TIM_LTO = np.array([TIM_Taxi, TIM_Approach, TIM_Climb, TIM_TakeOff])

        # Select fuel flow rates per mode
        if EDB_LTO:
            fuel_flows_LTO = np.array(ac_performance.EDB_data['fuelflow_KGperS'])
        else:
            fuel_flows_LTO = np.array(
                [
                    mode['FUEL_KGs']
                    for mode in ac_performance.LTO_data['thrust_settings'].values()
                ]
            )

        # Compute CO2, H2O, SOx indices using same methods as cruise
        self.LTO_emission_indices['CO2'], _ = EI_CO2(self.fuel)
        self.LTO_emission_indices['H2O'] = EI_H2O(self.fuel)
        (
            self.LTO_emission_indices['SO2'],
            self.LTO_emission_indices['SO4'],
        ) = EI_SOx(self.fuel)

        # For NOx, HC, CO, either use EDB_LTO or thrust_settings dict
        if EDB_LTO:
            self.LTO_emission_indices['NOx'] = np.array(
                ac_performance.EDB_data['NOX_EI_matrix']
            )
            self.LTO_emission_indices['HC'] = np.array(
                ac_performance.EDB_data['HC_EI_matrix']
            )
            self.LTO_emission_indices['CO'] = np.array(
                ac_performance.EDB_data['CO_EI_matrix']
            )
        else:
            settings = ac_performance.LTO_data['thrust_settings'].values()
            self.LTO_emission_indices['NOx'] = np.array(
                [mode['EI_NOx'] for mode in settings]
            )
            self.LTO_emission_indices['HC'] = np.array(
                [mode['EI_HC'] for mode in settings]
            )
            self.LTO_emission_indices['CO'] = np.array(
                [mode['EI_CO'] for mode in settings]
            )

        # Determine NOx speciation fractions based on thrust category
        thrustCat = get_thrust_cat(fuel_flows_LTO, None, cruiseCalc=False)
        (
            self.LTO_noProp,
            self.LTO_no2Prop,
            self.LTO_honoProp,
        ) = NOx_speciation(thrustCat)

        # Compute organic PM for LTO modes
        LTO_PMvol, LTO_OCic = EI_PMvol_NEW(fuel_flows_LTO, thrustCat)
        self.LTO_emission_indices['PMvol'] = LTO_PMvol
        self.LTO_emission_indices['OCic'] = LTO_OCic

        # Select black carbon emission indices based on nvpm_method
        if nvpm_method in ('foa3', 'newsnci'):
            PMnvolEI_ICAOthrust = ac_performance.EDB_data['PMnvolEI_best_ICAOthrust']
        elif nvpm_method in ('fox', 'dop', 'sst'):
            PMnvolEI_ICAOthrust = ac_performance.EDB_data['PMnvolEI_new_ICAOthrust']
        elif nvpm_method == 'SCOPE11':
            # SCOPE11 provides best, lower, and upper bounds
            edb = ac_performance.EDB_data
            PMnvolEI_ICAOthrust = calculate_PMnvolEI_scope11(
                np.array(edb['SN_matrix']),
                np.array(edb['PR']),
                np.array(edb['ENGINE_TYPE']),
                np.array(edb['BP_Ratio']),
            )
            PMnvolEIN_ICAOthrust = ac_performance.EDB_data['PMnvolEIN_best_ICAOthrust']
            PMnvolEI_lo_ICAOthrust = ac_performance.EDB_data[
                'PMnvolEI_lower_ICAOthrust'
            ]
            PMnvolEIN_lo_ICAOthrust = ac_performance.EDB_data[
                'PMnvolEIN_lower_ICAOthrust'
            ]
            PMnvolEI_hi_ICAOthrust = ac_performance.EDB_data[
                'PMnvolEI_upper_ICAOthrust'
            ]
            PMnvolEIN_hi_ICAOthrust = ac_performance.EDB_data[
                'PMnvolEIN_upper_ICAOthrust'
            ]
        else:
            raise ValueError(
                f"Re-define PMnvol estimation method: pmnvolSwitch = {nvpm_method}"
            )

        # Assign BC indices arrays for selected modes
        self.LTO_emission_indices['PMnvol'] = PMnvolEI_ICAOthrust
        if nvpm_method == 'SCOPE11':
            self.LTO_emission_indices['PMnvol_lo'] = PMnvolEI_lo_ICAOthrust
            self.LTO_emission_indices['PMnvol_hi'] = PMnvolEI_hi_ICAOthrust
            self.LTO_emission_indices['PMnvolN'] = PMnvolEIN_ICAOthrust
            self.LTO_emission_indices['PMnvolN_lo'] = PMnvolEIN_lo_ICAOthrust
            self.LTO_emission_indices['PMnvolN_hi'] = PMnvolEIN_hi_ICAOthrust

        # --- Compute LTO emissions in grams per mode
        # by multiplying EI by durations*flows ---
        LTO_fuel_burn = TIM_LTO * fuel_flows_LTO
        self.total_fuel_burn += np.sum(LTO_fuel_burn)
        for field in self.LTO_emission_indices.dtype.names:
            # If using performance model for climb and approach then set EIs to 0
            if self.traj_emissions_all:
                self.LTO_emission_indices[field][1:-1] = 0.0
            self.LTO_emissions_g[field] = (
                self.LTO_emission_indices[field] * LTO_fuel_burn
            )

    def get_GSE_emissions(self, wnsf):
        """
        Calculate Ground Service Equipment emissions based
        on aircraft size/freight type (WNSF).

        Parameters
        ----------
        wnsf : str
            Wide, Narrow, Small, or Freight ('w','n','s','f').
        """
        # Map WNSF letter to index for nominal emission lists
        mapping = {'wide': 0, 'narrow': 1, 'small': 2, 'freight': 3}
        idx = mapping.get(wnsf.lower())
        if idx is None:
            raise ValueError(
                "Invalid WNSF code; must be one of 'wide','narrow','small','freight'"
            )

        # Nominal emissions per engine start cycle [g/cycle]
        CO2_nom = [58e3, 18e3, 10e3, 58e3]  # g/cycle
        NOx_nom = [0.9e3, 0.4e3, 0.3e3, 0.9e3]  # g/cycle
        HC_nom = [0.07e3, 0.04e3, 0.03e3, 0.07e3]  # g/cycle (NMVOC)
        CO_nom = [0.3e3, 0.15e3, 0.1e3, 0.3e3]  # g/cycle
        PM10_nom = [0.055e3, 0.025e3, 0.020e3, 0.055e3]  # g/cycle (≈PM2.5)

        # Pick out the scalar values
        self.GSE_emissions_g['CO2'] = CO2_nom[idx]
        self.GSE_emissions_g['NOx'] = NOx_nom[idx]
        self.GSE_emissions_g['HC'] = HC_nom[idx]
        self.GSE_emissions_g['CO'] = CO_nom[idx]
        pm_core = PM10_nom[idx]

        # Fuel (kg/cycle) from CO2:
        #   EI_CO2 = fuel * 3.16 * 1000  ⇒  fuel = EI_CO2/(3.16*1000)
        CO2_EI, _ = EI_CO2(self.fuel)
        gse_fuel = self.GSE_emissions_g['CO2'] / CO2_EI
        self.total_fuel_burn += gse_fuel

        # NOx split
        self.GSE_emissions_g['NO'] = self.GSE_emissions_g['NOx'] * 0.90
        self.GSE_emissions_g['NO2'] = self.GSE_emissions_g['NOx'] * 0.09
        self.GSE_emissions_g['HONO'] = self.GSE_emissions_g['NOx'] * 0.01

        # Sulfate / SO2 fraction (independent of WNSF)
        GSE_FSC = 5.0  # fuel‐sulfur concentration (ppm)
        GSE_EPS = 0.02  # fraction → sulfate
        # g SO4 per kg fuel:
        self.GSE_emissions_g['SO4'] = (GSE_FSC / 1e6) * 1000.0 * GSE_EPS * (96.0 / 32.0)
        # g SO2 per kg fuel:
        self.GSE_emissions_g['SO2'] = (
            (GSE_FSC / 1e6) * 1000.0 * (1.0 - GSE_EPS) * (64.0 / 32.0)
        )

        # Subtract sulfate from the core PM₁₀ then split 50:50
        pm_minus_so4 = pm_core - self.GSE_emissions_g['SO4']
        self.GSE_emissions_g['PMvol'] = pm_minus_so4 * 0.5
        self.GSE_emissions_g['PMnvol'] = pm_minus_so4 * 0.5

    def get_lifecycle_emissions(self, fuel, traj):
        # add lifecycle CO2 emissions for climate model run
        lc_CO2 = (
            fuel['LC_CO2'] * (traj.fuel_mass * fuel['Energy_MJ_per_kg'])
        ) - self.summed_emission_g['CO2']
        self.summed_emission_g['CO2'] += lc_CO2

    ###################
    # PRIVATE METHODS #
    ###################
    def __emission_dtype(self, shape):
        n = (shape,)
        return (
            [
                ('CO2', np.float64, n),
                ('H2O', np.float64, n),
                ('HC', np.float64, n),
                ('CO', np.float64, n),
                ('NOx', np.float64, n),
                ('NO', np.float64, n),
                ('NO2', np.float64, n),
                ('HONO', np.float64, n),
                ('PMnvol', np.float64, n),
                ('PMnvol_lo', np.float64, n),
                ('PMnvol_hi', np.float64, n),
                ('PMnvolN', np.float64, n),
                ('PMnvolN_lo', np.float64, n),
                ('PMnvolN_hi', np.float64, n),
                ('PMnvolGMD', np.float64, n),
                ('PMvol', np.float64, n),
                ('OCic', np.float64, n),
                ('SO2', np.float64, n),
                ('SO4', np.float64, n),
            ]
            if self.pmnvol_mode == 'SCOPE11'
            else [
                ('CO2', np.float64, n),
                ('H2O', np.float64, n),
                ('HC', np.float64, n),
                ('CO', np.float64, n),
                ('NOx', np.float64, n),
                ('NO', np.float64, n),
                ('NO2', np.float64, n),
                ('HONO', np.float64, n),
                ('PMnvol', np.float64, n),
                ('PMnvolGMD', np.float64, n),
                ('PMvol', np.float64, n),
                ('OCic', np.float64, n),
                ('SO2', np.float64, n),
                ('SO4', np.float64, n),
            ]
        )

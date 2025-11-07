# Emissions class
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from AEIC.performance_model import PerformanceModel
from AEIC.trajectories import Trajectory
from AEIC.utils.consts import R_air, kappa
from AEIC.utils.files import file_location
from AEIC.utils.standard_atmosphere import (
    pressure_at_altitude_isa_bada4,
    temperature_at_altitude_isa_bada4,
)
from AEIC.utils.standard_fuel import get_SLS_equivalent_fuel_flow, get_thrust_cat

from .APU_emissions import get_APU_emissions
from .EI_CO2 import EI_CO2
from .EI_H2O import EI_H2O
from .EI_HCCO import EI_HCCO
from .EI_NOx import BFFM2_EINOx, NOx_speciation
from .EI_PMnvol import PMnvol_MEEM, calculate_PMnvolEI_scope11
from .EI_PMvol import EI_PMvol_FOA3, EI_PMvol_NEW
from .EI_SOx import EI_SOx


@dataclass(frozen=True)
class EmissionSettings:
    fuel_file: str
    traj_all: bool
    apu_enabled: bool
    gse_enabled: bool
    pmvol_method: str
    pmnvol_method: str
    lifecycle_enabled: bool
    metric_flags: Mapping[str, bool]
    method_flags: Mapping[str, str]


@dataclass(frozen=True)
class AtmosphericState:
    temperature: np.ndarray | None
    pressure: np.ndarray | None
    mach: np.ndarray | None


class Emission:
    """
    Model for determining and aggregating flight emissions across all mission segments,
    including cruise trajectory, LTO (Landing and Take-Off), APU, and GSE emissions,
    as well as lifecycle CO2 adjustments.
    """

    def __init__(self, ac_performance: PerformanceModel, trajectory: Trajectory):
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
        """

        self.trajectory = trajectory
        self.performance_model = ac_performance
        settings = self._parse_emission_settings(ac_performance.config)
        self.traj_emissions_all = settings['traj_all']
        self.pmvol_method = settings['pmvol_method']
        self.pmnvol_method = settings['pmnvol_method']
        self.apu_enabled = settings['apu']
        self.gse_enabled = settings['gse']
        self.lifecycle_enabled = settings['lifecycle']
        self.metric_flags = settings['metric_flags']
        self.method_flags = settings['methods']
        self._include_pmnvol_number = self.metric_flags[
            'PMnvol'
        ] and self.pmnvol_method in ('scope11', 'meem')
        self._scope11_cache = None

        with open(file_location(settings['fuel_file']), 'rb') as f:
            self.fuel = tomllib.load(f)
        self.co2_ei, self.nvol_carb_cont = EI_CO2(self.fuel)
        self.h2o_ei = EI_H2O(self.fuel)
        self.so2_ei, self.so4_ei = EI_SOx(self.fuel)

        # Unpack trajectory lengths: total, climb, cruise, descent points
        self.Ntot = trajectory.Ntot
        self.NClm = trajectory.NClm
        self.NCrz = trajectory.NCrz
        self.NDes = trajectory.NDes

        fill_value = -1.0  # Setting all emissions to -1 at the start
        # This helps in testing as any value still -1
        # after computing emissions means something is wrong
        self.emission_indices = np.full(
            (), fill_value, dtype=self.__emission_dtype(self.Ntot)
        )
        self.LTO_emission_indices = np.full(
            (), fill_value, dtype=self.__emission_dtype(4)
        )
        self.LTO_emissions_g = np.full((), fill_value, dtype=self.__emission_dtype(4))
        self.APU_emission_indices = np.full(
            (), fill_value, dtype=self.__emission_dtype(1)
        )
        self.APU_emissions_g = np.full((), fill_value, dtype=self.__emission_dtype(1))
        self.GSE_emissions_g = np.full((), fill_value, dtype=self.__emission_dtype(1))
        self.pointwise_emissions_g = np.full(
            (), fill_value, dtype=self.__emission_dtype(self.Ntot)
        )
        self.summed_emission_g = np.full((), fill_value, dtype=self.__emission_dtype(1))

        self._initialize_field_controls()
        self._apply_metric_mask(
            self.emission_indices,
            self.LTO_emission_indices,
            self.LTO_emissions_g,
            self.APU_emission_indices,
            self.APU_emissions_g,
            self.GSE_emissions_g,
            self.pointwise_emissions_g,
            self.summed_emission_g,
        )

        # Compute fuel burn per segment from fuelMass time series
        fuel_mass = trajectory.traj_data['fuelMass']
        fuel_burn = np.zeros_like(fuel_mass)
        # Difference between sequential mass values for ascent segments
        fuel_burn[1:] = fuel_mass[:-1] - fuel_mass[1:]
        self.fuel_burn_per_segment = fuel_burn

        self.total_fuel_burn = 0.0
        self.LTO_noProp = np.zeros(4)
        self.LTO_no2Prop = np.zeros(4)
        self.LTO_honoProp = np.zeros(4)

    def compute_emissions(self):
        """
        Compute all emissions
        """
        # Calculate cruise trajectory emissions (CO2, H2O, SOx, NOx, HC, CO, PM)
        self.get_trajectory_emissions()

        # Calculate LTO emissions for ground and approach/climb modes
        self.get_LTO_emissions()

        if self.apu_enabled:
            (
                self.APU_emission_indices,
                self.APU_emissions_g,
                apu_fuel_burn,
            ) = get_APU_emissions(
                self.APU_emission_indices,
                self.APU_emissions_g,
                self.LTO_emission_indices,
                self.performance_model.APU_data,
                self.LTO_noProp,
                self.LTO_no2Prop,
                self.LTO_honoProp,
                EI_H2O=self.h2o_ei,
                nvpm_method=self.pmnvol_method,
            )
            self._apply_metric_mask(self.APU_emission_indices)
            self._apply_metric_mask(self.APU_emissions_g)
            self.total_fuel_burn += apu_fuel_burn

        # Compute Ground Service Equipment (GSE) emissions based on WNSF type
        if self.gse_enabled:
            self.get_GSE_emissions(
                self.performance_model.model_info['General_Information'][
                    'aircraft_class'
                ]
            )
            self._apply_metric_mask(self.GSE_emissions_g)

        # Sum all emission contributions: trajectory + LTO + APU + GSE
        self.sum_total_emissions()

        # Add lifecycle CO2 emissions to total
        if self.metric_flags['CO2'] and self.lifecycle_enabled:
            self.get_lifecycle_emissions(self.fuel, self.trajectory)

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

    def get_trajectory_emissions(self):
        """
        Calculate emission indices (g/species per kg fuel) for each flight segment.

        Parameters
        ----------
        trajectory : Trajectory
            Contains altitudes, speeds, and fuel flows for each time step.
        ac_performance : PerformanceModel
            Provides EDB or LTO data matrices for EI lookup.
        """
        trajectory = self.trajectory
        ac_performance = self.performance_model
        idx_slice = (
            slice(0, self.Ntot)
            if self.traj_emissions_all
            else slice(0, self.Ntot - self.NDes)
        )
        traj_data = trajectory.traj_data
        lto_inputs = self._extract_lto_inputs()
        lto_ff_array = lto_inputs['fuel_flow']
        fuel_flow = traj_data['fuelFlow'][idx_slice]
        thrust_categories = get_thrust_cat(fuel_flow, lto_ff_array, cruiseCalc=True)

        needs_hc = self.metric_flags['HC'] or (
            self.metric_flags['PMvol'] and self.pmvol_method == 'foa3'
        )
        needs_co = self.metric_flags['CO']
        needs_nox = self.metric_flags['NOx']
        needs_pmnvol_meem = self.metric_flags['PMnvol'] and (
            self.pmnvol_method == 'meem'
        )
        needs_atmos = needs_hc or needs_co or needs_nox or needs_pmnvol_meem

        if needs_atmos:
            flight_alts = traj_data['altitude'][idx_slice]
            flight_temps = temperature_at_altitude_isa_bada4(flight_alts)
            flight_pressures = pressure_at_altitude_isa_bada4(flight_alts)
            mach_number = traj_data['tas'][idx_slice] / np.sqrt(
                kappa * R_air * flight_temps
            )
        else:
            flight_temps = flight_pressures = mach_number = None

        needs_sls_ff = needs_hc or needs_co or needs_nox
        if needs_sls_ff:
            sls_equiv_fuel_flow = get_SLS_equivalent_fuel_flow(
                fuel_flow,
                flight_pressures,
                flight_temps,
                mach_number,
                ac_performance.model_info['General_Information']['n_eng'],
            )
        else:
            sls_equiv_fuel_flow = None

        if self.metric_flags['CO2']:
            self.emission_indices['CO2'][idx_slice] = self.co2_ei
        if self.metric_flags['H2O']:
            self.emission_indices['H2O'][idx_slice] = self.h2o_ei
        if self.metric_flags['SOx']:
            self.emission_indices['SO2'][idx_slice] = self.so2_ei
            self.emission_indices['SO4'][idx_slice] = self.so4_ei

        if needs_nox:
            nox_method = self.method_flags['nox']
            if nox_method == 'bffm2':
                (
                    self.emission_indices['NOx'][idx_slice],
                    self.emission_indices['NO'][idx_slice],
                    self.emission_indices['NO2'][idx_slice],
                    self.emission_indices['HONO'][idx_slice],
                    *_,
                ) = BFFM2_EINOx(
                    sls_equiv_fuel_flow=sls_equiv_fuel_flow,
                    NOX_EI_matrix=lto_inputs['nox_ei'],
                    fuelflow_performance=lto_ff_array,
                    Pamb=flight_pressures,
                    Tamb=flight_temps,
                )
            elif nox_method == 'p3t3':
                print("P3T3 method not implemented yet..")
                pass
            elif nox_method == 'none':
                pass
            else:
                raise NotImplementedError(
                    f"EI_NOx_method '{self.method_flags['nox']}' is not supported."
                )

        hc_ei = None
        if needs_hc:
            hc_ei = EI_HCCO(
                sls_equiv_fuel_flow,
                lto_inputs['hc_ei'],
                lto_ff_array,
                Tamb=flight_temps,
                Pamb=flight_pressures,
                cruiseCalc=True,
            )
            if self.metric_flags['HC']:
                self.emission_indices['HC'][idx_slice] = hc_ei

        if needs_co:
            co_ei = EI_HCCO(
                sls_equiv_fuel_flow,
                lto_inputs['co_ei'],
                lto_ff_array,
                Tamb=flight_temps,
                Pamb=flight_pressures,
                cruiseCalc=True,
            )
            self.emission_indices['CO'][idx_slice] = co_ei

        if self.metric_flags['PMvol']:
            pmvol_ei = ocic_ei = None
            if self.pmvol_method == 'fuel_flow':
                thrust_labels = self._thrust_band_labels(thrust_categories)
                pmvol_ei, ocic_ei = EI_PMvol_NEW(fuel_flow, thrust_labels)
            elif self.pmvol_method == 'foa3':
                thrust_pct = self._thrust_percentages_from_categories(thrust_categories)
                pmvol_ei, ocic_ei = EI_PMvol_FOA3(thrust_pct, hc_ei)
            elif self.pmvol_method == 'none':
                pass
            else:
                raise NotImplementedError(
                    f"EI_PMvol_method '{self.pmvol_method}' is not supported."
                )

            if pmvol_ei is not None:
                self.emission_indices['PMvol'][idx_slice] = pmvol_ei
                self.emission_indices['OCic'][idx_slice] = ocic_ei

        if self.metric_flags['PMnvol']:
            pmnvol_method = self.pmnvol_method
            if pmnvol_method == 'meem':
                (
                    self.emission_indices['PMnvolGMD'][idx_slice],
                    self.emission_indices['PMnvol'][idx_slice],
                    pmnvol_num,
                ) = PMnvol_MEEM(
                    ac_performance.EDB_data,
                    traj_data['altitude'][idx_slice],
                    flight_temps,
                    flight_pressures,
                    mach_number,
                )
                if (
                    self._include_pmnvol_number
                    and 'PMnvolN' in self.emission_indices.dtype.names
                ):
                    self.emission_indices['PMnvolN'][idx_slice] = pmnvol_num
            elif pmnvol_method == 'scope11':
                profile = self._scope11_profile(ac_performance)
                self.emission_indices['PMnvol'][idx_slice] = (
                    self._map_mode_values_to_categories(
                        profile['mass'], thrust_categories
                    )
                )
                self.emission_indices['PMnvolGMD'][idx_slice] = 0.0
                if (
                    self._include_pmnvol_number
                    and profile['number'] is not None
                    and 'PMnvolN' in self.emission_indices.dtype.names
                ):
                    self.emission_indices['PMnvolN'][idx_slice] = (
                        self._map_mode_values_to_categories(
                            profile['number'], thrust_categories
                        )
                    )
            elif pmnvol_method == 'none':
                pass
            else:
                raise NotImplementedError(
                    f"EI_PMnvol_method '{self.pmnvol_method}' is not supported."
                )

        self.total_fuel_burn = np.sum(self.fuel_burn_per_segment[idx_slice])
        for field in self._active_fields:
            self.pointwise_emissions_g[field][idx_slice] = (
                self.emission_indices[field][idx_slice]
                * self.fuel_burn_per_segment[idx_slice]
            )

    def get_LTO_emissions(self):
        """
        Compute Landing-and-Takeoff cycle emission indices and quantities.
        """
        ac_performance = self.performance_model

        # Standard TIM durations
        # https://www.icao.int/environmental-protection/Documents/EnvironmentalReports/2016/ENVReport2016_pg73-74.pdf
        TIM_TakeOff = 0.7 * 60
        TIM_Climb = 2.2 * 60
        TIM_Approach = 4.0 * 60
        TIM_Taxi = 26.0 * 60

        if self.traj_emissions_all:
            TIM_Climb = 0.0
            TIM_Approach = 0.0

        TIM_LTO = np.array([TIM_Taxi, TIM_Approach, TIM_Climb, TIM_TakeOff])
        lto_inputs = self._extract_lto_inputs()
        fuel_flows_LTO = lto_inputs['fuel_flow']
        thrustCat = get_thrust_cat(fuel_flows_LTO, None, cruiseCalc=False)
        thrust_labels = self._thrust_band_labels(thrustCat)

        if self.metric_flags['CO2']:
            self.LTO_emission_indices['CO2'] = np.full_like(
                fuel_flows_LTO, self.co2_ei, dtype=float
            )
        if self.metric_flags['H2O']:
            self.LTO_emission_indices['H2O'] = np.full_like(
                fuel_flows_LTO, self.h2o_ei, dtype=float
            )
        if self.metric_flags['SOx']:
            self.LTO_emission_indices['SO2'] = np.full_like(
                fuel_flows_LTO, self.so2_ei, dtype=float
            )
            self.LTO_emission_indices['SO4'] = np.full_like(
                fuel_flows_LTO, self.so4_ei, dtype=float
            )

        if self.metric_flags['NOx']:
            if self.method_flags['nox'] == 'none':
                self.LTO_noProp = np.zeros_like(fuel_flows_LTO)
                self.LTO_no2Prop = np.zeros_like(fuel_flows_LTO)
                self.LTO_honoProp = np.zeros_like(fuel_flows_LTO)
            else:
                self.LTO_emission_indices['NOx'] = lto_inputs['nox_ei']
                (
                    self.LTO_noProp,
                    self.LTO_no2Prop,
                    self.LTO_honoProp,
                ) = NOx_speciation(thrustCat)
                self.LTO_emission_indices['NO'] = (
                    self.LTO_emission_indices['NOx'] * self.LTO_noProp
                )
                self.LTO_emission_indices['NO2'] = (
                    self.LTO_emission_indices['NOx'] * self.LTO_no2Prop
                )
                self.LTO_emission_indices['HONO'] = (
                    self.LTO_emission_indices['NOx'] * self.LTO_honoProp
                )
        else:
            self.LTO_noProp = np.zeros_like(fuel_flows_LTO)
            self.LTO_no2Prop = np.zeros_like(fuel_flows_LTO)
            self.LTO_honoProp = np.zeros_like(fuel_flows_LTO)

        if self.metric_flags['HC']:
            self.LTO_emission_indices['HC'] = lto_inputs['hc_ei']
        if self.metric_flags['CO']:
            self.LTO_emission_indices['CO'] = lto_inputs['co_ei']

        if self.metric_flags['PMvol']:
            if self.pmvol_method == 'fuel_flow':
                LTO_PMvol, LTO_OCic = EI_PMvol_NEW(fuel_flows_LTO, thrust_labels)
            elif self.pmvol_method == 'foa3':
                LTO_PMvol, LTO_OCic = EI_PMvol_FOA3(
                    lto_inputs['thrust_pct'], lto_inputs['hc_ei']
                )
            elif self.pmvol_method == 'none':
                LTO_PMvol = LTO_OCic = np.zeros_like(fuel_flows_LTO)
            else:
                raise NotImplementedError(
                    f"EI_PMvol_method '{self.pmvol_method}' is not supported."
                )
            self.LTO_emission_indices['PMvol'] = LTO_PMvol
            self.LTO_emission_indices['OCic'] = LTO_OCic

        if self.metric_flags['PMnvol']:
            pmnvol_method = self.pmnvol_method
            if pmnvol_method in ('foa3', 'newsnci', 'meem'):
                PMnvolEI_ICAOthrust = np.asarray(
                    ac_performance.EDB_data['PMnvolEI_best_ICAOthrust'], dtype=float
                )
                PMnvolEIN_ICAOthrust = None
            elif pmnvol_method in ('fox', 'dop', 'sst'):
                PMnvolEI_ICAOthrust = np.asarray(
                    ac_performance.EDB_data['PMnvolEI_new_ICAOthrust'], dtype=float
                )
                PMnvolEIN_ICAOthrust = None
            elif pmnvol_method == 'scope11':
                profile = self._scope11_profile(ac_performance)
                PMnvolEI_ICAOthrust = profile['mass']
                PMnvolEIN_ICAOthrust = profile['number']
            elif pmnvol_method == 'none':
                PMnvolEI_ICAOthrust = np.zeros_like(fuel_flows_LTO)
                PMnvolEIN_ICAOthrust = None
            else:
                raise ValueError(
                    f'''Re-define PMnvol estimation method:
                    pmnvolSwitch = {self.pmnvol_method}'''
                )

            self.LTO_emission_indices['PMnvol'] = PMnvolEI_ICAOthrust
            if (
                self._include_pmnvol_number
                and PMnvolEIN_ICAOthrust is not None
                and 'PMnvolN' in self.LTO_emission_indices.dtype.names
            ):
                self.LTO_emission_indices['PMnvolN'] = PMnvolEIN_ICAOthrust
        self.LTO_emission_indices['PMnvolGMD'] = np.zeros_like(fuel_flows_LTO)

        LTO_fuel_burn = TIM_LTO * fuel_flows_LTO
        self.total_fuel_burn += np.sum(LTO_fuel_burn)

        for field in self.LTO_emission_indices.dtype.names:
            if (
                self.traj_emissions_all
                and field in self.LTO_emission_indices.dtype.names
            ):
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
        idx = self._wnsf_index(wnsf)
        nominal = self._gse_nominal_profile(idx)
        self._assign_constant_indices(
            self.GSE_emissions_g,
            {key: nominal[key] for key in ('CO2', 'NOx', 'HC', 'CO')},
        )
        pm_core = nominal['PM10']

        CO2_EI = getattr(self, 'co2_ei', EI_CO2(self.fuel)[0])
        gse_fuel = self.GSE_emissions_g['CO2'] / CO2_EI
        self.total_fuel_burn += gse_fuel

        H2O_EI = getattr(self, 'h2o_ei', EI_H2O(self.fuel))
        self.GSE_emissions_g['H2O'] = H2O_EI * gse_fuel

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
        # No PMnvolN or PMnvolGMD or OCic
        if 'PMnvolN' in self.GSE_emissions_g.dtype.names:
            self.GSE_emissions_g['PMnvolN'] = 0.0
        self.GSE_emissions_g['PMnvolGMD'] = 0.0
        self.GSE_emissions_g['OCic'] = 0.0

    def get_lifecycle_emissions(self, fuel, traj):
        """Apply lifecycle CO2 adjustments when requested by the config."""
        if hasattr(self, "metric_flags") and not self.metric_flags.get('CO2', True):
            return
        # add lifecycle CO2 emissions for climate model run
        lc_CO2 = (
            fuel['LC_CO2'] * (traj.total_fuel_mass * fuel['Energy_MJ_per_kg'])
        ) - self.summed_emission_g['CO2']
        self.summed_emission_g['CO2'] += lc_CO2
=======
        lifecycle_total = fuel['LC_CO2'] * (traj.fuel_mass * fuel['Energy_MJ_per_kg'])
        self.summed_emission_g['CO2'] = lifecycle_total

    def compute_EI_NOx(
        self,
        idx_slice: slice,
        lto_inputs,
        atmos_state: AtmosphericState,
        sls_equiv_fuel_flow,
    ):
        """Fill NOx-related EI arrays according to the selected method."""
        method = self.method_flags.get('nox', 'none')
        if method == 'none':
            return
        if method == 'bffm2':
            if (
                sls_equiv_fuel_flow is None
                or atmos_state.temperature is None
                or atmos_state.pressure is None
            ):
                raise RuntimeError(
                    "BFFM2 NOx requires atmosphere and SLS equivalent fuel flow."
                )
            (
                self.emission_indices['NOx'][idx_slice],
                self.emission_indices['NO'][idx_slice],
                self.emission_indices['NO2'][idx_slice],
                self.emission_indices['HONO'][idx_slice],
                *_,
            ) = BFFM2_EINOx(
                sls_equiv_fuel_flow=sls_equiv_fuel_flow,
                NOX_EI_matrix=lto_inputs['nox_ei'],
                fuelflow_performance=lto_inputs['fuel_flow'],
                Pamb=atmos_state.pressure,
                Tamb=atmos_state.temperature,
            )
        elif method == 'p3t3':
            print("P3T3 method not implemented yet..")
        else:
            raise NotImplementedError(
                f"EI_NOx_method '{self.method_flags['nox']}' is not supported."
            )
>>>>>>> 3cd11cb (cleaned up emissions class, solution slightly more elegant now, emissions not in config set to -1):src/emissions/emission.py

    ###################
    # PRIVATE METHODS #
    ###################
    def _parse_emission_settings(self, config):
        """Flatten relevant config keys and derive boolean/method flags."""

        def bool_opt(key, default=True):
            value = config.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in ('1', 'true', 'yes', 'y')
            return bool(value)

        def method_opt(key, default):
            raw = config.get(key, default)
            if raw is None:
                raw = default
            if isinstance(raw, str):
                raw_clean = raw.strip() or default
            else:
                raw_clean = str(raw)
            return raw_clean.lower()

        fuel_name = config.get('Fuel') or config.get('fuel_file') or 'conventional_jetA'
        nox_method = method_opt('EI_NOx_method', 'BFFM2')
        hc_method = method_opt('EI_HC_method', 'BFFM2')
        co_method = method_opt('EI_CO_method', 'BFFM2')
        pmvol_method = method_opt('EI_PMvol_method', 'fuel_flow')
        pmnvol_method = method_opt('EI_PMnvol_method', 'scope11')

        metric_flags = {
            'CO2': bool_opt('CO2_calculation', True),
            'H2O': bool_opt('H2O_calculation', True),
            'SOx': bool_opt('SOx_calculation', True),
            'NOx': nox_method != 'none',
            'HC': hc_method != 'none',
            'CO': co_method != 'none',
            'PMvol': pmvol_method != 'none',
            'PMnvol': pmnvol_method != 'none',
        }

        return {
            'fuel_file': f"fuels/{fuel_name}.toml",
            'traj_all': bool_opt('climb_descent_usage', True),
            'apu': bool_opt('APU_calculation', True),
            'gse': bool_opt('GSE_calculation', True),
            'pmvol_method': pmvol_method,
            'pmnvol_method': pmnvol_method,
            'lifecycle': bool_opt('LC_calculation', True),
            'metric_flags': metric_flags,
            'methods': {
                'nox': nox_method,
                'hc': hc_method,
                'co': co_method,
                'pmvol': pmvol_method,
                'pmnvol': pmnvol_method,
            },
        }

    def _initialize_field_controls(self):
        """Map dtype fields to metric flags so we can zero disabled outputs."""
        # TODO: This might be a little overboard, and there's
        # got to be another way to solve this problem
        field_groups = {
            'CO2': 'CO2',
            'H2O': 'H2O',
            'HC': 'HC',
            'CO': 'CO',
            'NOx': 'NOx',
            'NO': 'NOx',
            'NO2': 'NOx',
            'HONO': 'NOx',
            'PMvol': 'PMvol',
            'OCic': 'PMvol',
            'PMnvol': 'PMnvol',
            'PMnvolN': 'PMnvol',
            'PMnvolGMD': 'PMnvol',
            'SO2': 'SOx',
            'SO4': 'SOx',
        }
        controls = {}
        include_number = getattr(
            self,
            "_include_pmnvol_number",
            getattr(self, 'pmnvol_mode', '').lower() in ('scope11', 'meem'),
        )

        for field in self.emission_indices.dtype.names:
            group = field_groups.get(field)
            if group is None:
                controls[field] = True
                continue
            enabled = self.metric_flags.get(group, False)
            if field == 'PMnvolN':
                enabled = enabled and include_number
            controls[field] = enabled

        self._field_controls = controls
        self._active_fields = tuple(
            field for field in self.emission_indices.dtype.names if controls[field]
        )

    def _extract_lto_inputs(self):
        """Return ordered fuel-flow and EI arrays for either EDB or user LTO data."""
        if self.performance_model.config['LTO_input_mode'] == "EDB":
            edb = self.performance_model.EDB_data
            fuel_flow = np.asarray(edb['fuelflow_KGperS'], dtype=float)
            nox_ei = np.asarray(edb['NOX_EI_matrix'], dtype=float)
            hc_ei = np.asarray(edb['HC_EI_matrix'], dtype=float)
            co_ei = np.asarray(edb['CO_EI_matrix'], dtype=float)
            thrust_pct = np.array([7.0, 30.0, 85.0, 100.0], dtype=float)
        else:
            settings = self.performance_model.LTO_data['thrust_settings']
            ordered_modes = self._ordered_thrust_settings(settings)
            fuel_flow = np.array(
                [mode['FUEL_KGs'] for mode in ordered_modes], dtype=float
            )
            nox_ei = np.array(
                [mode.get('NOX_EI', mode.get('EI_NOx', 0.0)) for mode in ordered_modes],
                dtype=float,
            )
            hc_ei = np.array(
                [mode.get('HC_EI', mode.get('EI_HC', 0.0)) for mode in ordered_modes],
                dtype=float,
            )
            co_ei = np.array(
                [mode.get('CO_EI', mode.get('EI_CO', 0.0)) for mode in ordered_modes],
                dtype=float,
            )
            thrust_pct = np.array(
                [mode.get('THRUST_FRAC', 0.0) * 100.0 for mode in ordered_modes],
                dtype=float,
            )

        return {
            'fuel_flow': fuel_flow,
            'nox_ei': nox_ei,
            'hc_ei': hc_ei,
            'co_ei': co_ei,
            'thrust_pct': thrust_pct,
        }

    def _ordered_thrust_settings(self, settings):
        """Preserve canonical idle -> takeoff order for thrust setting dictionaries."""
        if not isinstance(settings, dict):
            return list(settings)
        lower_lookup = {key.lower(): value for key, value in settings.items()}
        ordered = []
        for key in ('idle', 'approach', 'climb', 'takeoff'):
            if key in lower_lookup:
                ordered.append(lower_lookup[key])
        for value in settings.values():
            if value not in ordered:
                ordered.append(value)
        return ordered

    def _thrust_band_labels(self, thrust_categories: np.ndarray) -> np.ndarray:
        """Translate numeric thrust codes into the L/H labels used by EI_PMvol_NEW."""
        labels = np.full(thrust_categories.shape, 'H', dtype='<U1')
        labels[thrust_categories == 2] = 'L'
        return labels

    def _thrust_percentages_from_categories(self, thrust_categories: np.ndarray):
        """Convert thrust codes into representative ICAO mode percentages."""
        thrust_pct = np.full(thrust_categories.shape, 85.0, dtype=float)
        thrust_pct[thrust_categories == 2] = 7.0
        thrust_pct[thrust_categories == 3] = 30.0
        return thrust_pct

    def _apply_metric_mask(self, *arrays):
        """Zero out disabled pollutant fields so opt-outs show as 0 instead of -1."""
        for array in arrays:
            if array is None:
                continue
            for field, enabled in self._field_controls.items():
                if (not enabled) and (field in array.dtype.names):
                    array[field] = 0.0

    def _scope11_profile(self, ac_performance):
        """Cache SCOPE11 lookups so we do the work only once."""
        if self._scope11_cache is None:
            edb = ac_performance.EDB_data
            mass = calculate_PMnvolEI_scope11(
                np.array(edb['SN_matrix']),
                np.array(edb['PR']),
                np.array(edb['ENGINE_TYPE']),
                np.array(edb['BP_Ratio']),
            )
            number = edb.get('PMnvolEIN_best_ICAOthrust')
            self._scope11_cache = {
                'mass': np.asarray(mass, dtype=float),
                'number': None if number is None else np.asarray(number, dtype=float),
            }
        return self._scope11_cache

    def _map_mode_values_to_categories(
        self, mode_values: np.ndarray, thrust_categories: np.ndarray
    ):
        """Broadcast mode-level EIs to arbitrary thrust-category time series."""
        values = np.asarray(mode_values, dtype=float).ravel()
        if values.size == 0:
            return np.zeros_like(thrust_categories, dtype=float)
        mapped = np.full(thrust_categories.shape, values[-1], dtype=float)
        mapped[thrust_categories == 2] = values[0]
        if values.size > 1:
            mapped[thrust_categories == 3] = values[1]
        if values.size > 2:
            mapped[thrust_categories == 1] = values[2]
        return mapped

    def __emission_dtype(self, shape):
        """Build the structured dtype used for every emission array."""
        n = (shape,)
        fields = [
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
        include_number = getattr(
            self,
            "_include_pmnvol_number",
            getattr(self, 'pmnvol_mode', '').lower() in ('scope11', 'meem'),
        )
        if include_number:
            fields.append(('PMnvolN', np.float64, n))
        return fields

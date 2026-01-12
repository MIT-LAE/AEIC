# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from AEIC.config import config
from AEIC.performance import LTOInputs
from AEIC.performance.models import PerformanceModel
from AEIC.trajectories.trajectory import Trajectory
from AEIC.utils.consts import R_air, kappa
from AEIC.utils.standard_atmosphere import (
    pressure_at_altitude_isa_bada4,
    temperature_at_altitude_isa_bada4,
)
from AEIC.utils.standard_fuel import (
    get_SLS_equivalent_fuel_flow,
    get_thrust_cat_cruise,
    get_thrust_cat_lto,
)

from .APU_emissions import get_APU_emissions
from .config import (
    EINOxMethod,
    PMnvolMethod,
    PMvolMethod,
)
from .EI_CO2 import EI_CO2
from .EI_H2O import EI_H2O
from .EI_HCCO import EI_HCCO
from .EI_NOx import BFFM2_EINOx, NOx_speciation
from .EI_PMnvol import PMnvol_MEEM, calculate_PMnvolEI_scope11
from .EI_PMvol import EI_PMvol_FOA3, EI_PMvol_FuelFlow
from .EI_SOx import EI_SOx


@dataclass(frozen=True)
class AtmosphericState:
    """Per-segment Atmospheric state used by EI models.

    Temperature (K) and pressure (Pa) follow the trajectory indexing, while
    `mach` is derived from TAS. Fields may be None when a method does not
    require atmosphere data (e.g., when NOx/PM indices are disabled)."""

    temperature: np.ndarray | None
    pressure: np.ndarray | None
    mach: np.ndarray | None


@dataclass(frozen=True)
class EmissionSlice:
    """Emission data for a single source such as trajectory, LTO, APU, or GSE.

    `indices` identifies the contributing mission segments (or None for
    aggregate-only sources) and `emissions_g` is the structured per-species
    array of integrated masses in grams aligned with those indices."""

    indices: np.ndarray | None
    emissions_g: np.ndarray


@dataclass(frozen=True)
class TrajectoryEmissionSlice(EmissionSlice):
    """EmissionSlice specialized for total trajectory contribution.

    Adds per-segment fuel burn (kg) and the total fuel burned so downstream
    analysis can compute intensity metrics or lifecycle adjustments alongside
    the species masses."""

    fuel_burn_per_segment: np.ndarray
    total_fuel_burn: float


@dataclass(frozen=True)
class EmissionsOutput:
    """Data container that stores aggregated emissions for a mission
    broken down by flight phase/components."""

    trajectory: TrajectoryEmissionSlice
    lto: EmissionSlice
    apu: EmissionSlice | None
    gse: EmissionSlice | None
    total: np.ndarray
    lifecycle_co2_g: float | None = None


class Emission:
    """
    Stateless emissions calculator that can be reused across trajectories.
    Configure it with a ``PerformanceModel`` once, then call ``emit`` with each
    trajectory to obtain an :class:`EmissionsOutput` data container.
    """

    def __init__(self, ac_performance: PerformanceModel):
        """
        Parameters
        ----------
        ac_performance : PerformanceModel
            Aircraft performance object containing climb/cruise/descent
            performance and LTO data.
        """

        self.performance_model = ac_performance

        self._include_pmnvol_number = config.emissions.pmnvol_enabled and (
            config.emissions.pmnvol_method in (PMnvolMethod.SCOPE11, PMnvolMethod.MEEM)
        )
        self._scope11_cache = None

        with open(config.emissions.fuel_file, 'rb') as f:
            self.fuel = tomllib.load(f)

        self._reset_run_state()

    def emit(self, trajectory: Trajectory) -> EmissionsOutput:
        """Compute emissions for a single trajectory and return structured results."""
        self._prepare_run_state(trajectory)
        self.compute_emissions()
        lifecycle_adjustment = None
        if config.emissions.co2_enabled and config.emissions.lifecycle_enabled:
            lifecycle_adjustment = self.get_lifecycle_emissions(self.fuel, trajectory)
        return self._collect_emissions(lifecycle_adjustment)

    def emit_trajectory(self, trajectory: Trajectory) -> TrajectoryEmissionSlice:
        """Convenience helper returning only the trajectory portion of emissions."""
        return self.emit(trajectory).trajectory

    def emit_lto(self, trajectory: Trajectory) -> EmissionSlice:
        """Convenience helper returning only the LTO emissions."""
        return self.emit(trajectory).lto

    def emit_apu(self, trajectory: Trajectory) -> EmissionSlice | None:
        """Convenience helper returning only the APU emissions (if enabled)."""
        return self.emit(trajectory).apu

    def emit_gse(self, trajectory: Trajectory) -> EmissionSlice | None:
        """Convenience helper returning only the GSE emissions (if enabled)."""
        return self.emit(trajectory).gse

    def compute_emissions(self):
        """
        Compute all emissions
        """
        # Calculate cruise trajectory emissions (CO2, H2O, SOx, NOx, HC, CO, PM)
        self.get_trajectory_emissions()

        # Calculate LTO emissions for ground and approach/climb modes
        self.get_LTO_emissions()

        if config.emissions.apu_enabled:
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
                EI_H2O=EI_H2O(self.fuel),
                nvpm_method=config.emissions.pmnvol_method,
            )
            self.total_fuel_burn += apu_fuel_burn

        # Compute Ground Service Equipment (GSE) emissions based on WNSF type
        if config.emissions.gse_enabled:
            self.get_GSE_emissions(self.performance_model.aircraft_class)

        # Sum all emission contributions: trajectory + LTO + APU + GSE
        self.sum_total_emissions()

    def sum_total_emissions(self):
        """
        Aggregate emissions (g) across all sources into summed_emission_g.
        Sums pointwise trajectory, LTO, APU, and GSE emissions for each species.
        """
        for field in self.summed_emission_g.dtype.names:
            total = np.sum(self.pointwise_emissions_g[field])
            total += np.sum(self.LTO_emissions_g[field])
            if config.emissions.apu_enabled:
                total += np.sum(self.APU_emissions_g[field])
            if config.emissions.gse_enabled:
                total += np.sum(self.GSE_emissions_g[field])
            self.summed_emission_g[field] = total

    def get_trajectory_emissions(self):
        """
        Calculate emissions for each flight trajectory point.
        """
        trajectory = self.trajectory
        ac_performance = self.performance_model
        idx_slice = self._trajectory_slice()
        lto_inputs = self._extract_lto_inputs()
        lto_ff_array = lto_inputs.fuel_flow
        fuel_flow = trajectory.fuel_flow[idx_slice]
        thrust_categories = get_thrust_cat_cruise(fuel_flow, lto_ff_array)

        needs_hc = config.emissions.hc_enabled or (
            config.emissions.pmvol_enabled
            and config.emissions.pmvol_method is PMvolMethod.FOA3
        )
        needs_co = config.emissions.co_enabled
        needs_nox = config.emissions.nox_enabled
        needs_pmnvol_meem = config.emissions.pmnvol_enabled and (
            config.emissions.pmnvol_method == 'meem'
        )
        needs_atmos = needs_hc or needs_co or needs_nox or needs_pmnvol_meem

        altitudes = trajectory.altitude[idx_slice]
        tas = trajectory.true_airspeed[idx_slice]
        atmos_state = self._atmospheric_state(altitudes, tas, needs_atmos)

        needs_sls_ff = needs_hc or needs_co or needs_nox
        sls_equiv_fuel_flow = self._sls_equivalent_fuel_flow(
            needs_sls_ff, fuel_flow, atmos_state
        )

        constants = self._constant_species_values()
        if constants:
            self._assign_constant_indices(self.emission_indices, constants, idx_slice)

        if needs_nox:
            self.compute_EI_NOx(idx_slice, lto_inputs, atmos_state, sls_equiv_fuel_flow)

        hc_ei = None
        if needs_hc:
            hc_ei = self._compute_EI_HCCO(
                sls_equiv_fuel_flow,
                lto_inputs.EI_HC,
                lto_ff_array,
                atmos_state,
            )
            if config.emissions.hc_enabled:
                self.emission_indices['HC'][idx_slice] = hc_ei

        if needs_co:
            co_ei = self._compute_EI_HCCO(
                sls_equiv_fuel_flow,
                lto_inputs.EI_CO,
                lto_ff_array,
                atmos_state,
            )
            self.emission_indices['CO'][idx_slice] = co_ei

        self._calculate_EI_PMvol(idx_slice, thrust_categories, fuel_flow, hc_ei)

        if config.emissions.pmnvol_enabled:
            self._calculate_EI_PMnvol(
                idx_slice,
                thrust_categories,
                altitudes,
                atmos_state,
                ac_performance,
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
        TIM_LTO = self._get_LTO_TIMs()
        lto_inputs = self._extract_lto_inputs()
        fuel_flows_LTO = lto_inputs.fuel_flow
        thrustCat = get_thrust_cat_lto(fuel_flows_LTO)
        thrust_labels = self._thrust_band_labels(thrustCat)

        constants = self._constant_species_values()
        if constants:
            self._assign_constant_indices(self.LTO_emission_indices, constants)

        self._get_LTO_nox(thrustCat, lto_inputs)

        if config.emissions.hc_enabled:
            self.LTO_emission_indices['HC'] = lto_inputs.EI_HC
        if config.emissions.co_enabled:
            self.LTO_emission_indices['CO'] = lto_inputs.EI_CO

        if config.emissions.pmvol_enabled:
            self._get_LTO_PMvol(fuel_flows_LTO, thrust_labels, lto_inputs)

        if config.emissions.pmnvol_enabled:
            self._get_LTO_PMnvol(ac_performance, fuel_flows_LTO)
        self.LTO_emission_indices['PMnvolGMD'] = np.zeros_like(fuel_flows_LTO)

        LTO_fuel_burn = TIM_LTO * fuel_flows_LTO
        self.total_fuel_burn += np.sum(LTO_fuel_burn)

        for field in self.LTO_emission_indices.dtype.names:
            if (
                config.emissions.climb_descent_usage
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

        co2_result = getattr(self, 'co2_ei', None)
        if co2_result is None:
            co2_result = EI_CO2(self.fuel)
        CO2_EI = co2_result.EI_CO2
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

    def get_lifecycle_emissions(self, fuel, traj) -> float | None:
        """Apply lifecycle CO2 adjustments when requested by the config."""
        if not config.emissions.co2_enabled:
            return None
        fuel_mass = getattr(traj, 'fuel_mass', None)
        fuel_used = fuel_mass[-1] - fuel_mass[0]
        if fuel_mass is None:
            raise ValueError(
                "Trajectory is missing total fuel_mass required for lifecycle CO2."
            )
        lifecycle_total = float(fuel['LC_CO2'] * (fuel_used * fuel['Energy_MJ_per_kg']))
        self.summed_emission_g['CO2'] += lifecycle_total
        return lifecycle_total

    def compute_EI_NOx(
        self,
        idx_slice: slice,
        lto_inputs,
        atmos_state: AtmosphericState,
        sls_equiv_fuel_flow,
    ):
        """Fill NOx-related EI arrays according to the selected method."""
        method = config.emissions.nox_method
        if method is EINOxMethod.NONE:
            return
        if method is EINOxMethod.BFFM2:
            if (
                sls_equiv_fuel_flow is None
                or atmos_state.temperature is None
                or atmos_state.pressure is None
            ):
                raise RuntimeError(
                    "BFFM2 NOx requires atmosphere and SLS equivalent fuel flow."
                )
            bffm2_result = BFFM2_EINOx(
                sls_equiv_fuel_flow=sls_equiv_fuel_flow,
                EI_NOx_matrix=lto_inputs.EI_NOx,
                fuelflow_performance=lto_inputs.fuel_flow,
                Pamb=atmos_state.pressure,
                Tamb=atmos_state.temperature,
            )
            self.emission_indices['NOx'][idx_slice] = bffm2_result.NOxEI
            self.emission_indices['NO'][idx_slice] = bffm2_result.NOEI
            self.emission_indices['NO2'][idx_slice] = bffm2_result.NO2EI
            self.emission_indices['HONO'][idx_slice] = bffm2_result.HONOEI
        elif method is EINOxMethod.P3T3:
            print("P3T3 method not implemented yet..")
        else:
            raise NotImplementedError(
                f"EI_NOx_method '{config.emissions.nox_method.value}' is not supported."
            )

    ###################
    # PRIVATE METHODS #
    ###################
    def _reset_run_state(self):
        """Initialize per-run placeholders so attributes always exist."""
        self.trajectory = None
        self.Ntot = 0
        self.NClm = 0
        self.NCrz = 0
        self.NDes = 0
        self.emission_indices = None
        self.LTO_emission_indices = None
        self.LTO_emissions_g = None
        self.APU_emission_indices = None
        self.APU_emissions_g = None
        self.GSE_emissions_g = None
        self.pointwise_emissions_g = None
        self.summed_emission_g = None
        self.fuel_burn_per_segment = None
        self.total_fuel_burn = 0.0
        self.LTO_noProp = np.zeros(4)
        self.LTO_no2Prop = np.zeros(4)
        self.LTO_honoProp = np.zeros(4)

    def _prepare_run_state(self, trajectory: Trajectory):
        """Allocate arrays and derived data for a single emission run."""
        self.trajectory = trajectory
        self.Ntot = trajectory.X_npoints
        self.NClm = trajectory.n_climb
        self.NCrz = trajectory.n_cruise
        self.NDes = trajectory.n_descent

        self.emission_indices = self._new_emission_array(self.Ntot)
        self.LTO_emission_indices = self._new_emission_array(4)
        self.LTO_emissions_g = self._new_emission_array(4)
        self.APU_emission_indices = self._new_emission_array(1)
        self.APU_emissions_g = self._new_emission_array(1)
        self.GSE_emissions_g = self._new_emission_array(1)
        self.pointwise_emissions_g = self._new_emission_array(self.Ntot)
        self.summed_emission_g = self._new_emission_array(1)

        self._initialize_field_controls()

        fuel_mass = trajectory.fuel_mass
        fuel_burn = np.zeros_like(fuel_mass)
        fuel_burn[1:] = fuel_mass[:-1] - fuel_mass[1:]
        self.fuel_burn_per_segment = fuel_burn
        self.total_fuel_burn = 0.0
        self.LTO_noProp = np.zeros(4)
        self.LTO_no2Prop = np.zeros(4)
        self.LTO_honoProp = np.zeros(4)

    def _collect_emissions(self, lifecycle_adjustment: float | None) -> EmissionsOutput:
        """Bundle intermediate arrays into immutable emission slices."""
        total_fuel_burn = float(np.asarray(self.total_fuel_burn).reshape(-1)[0])
        trajectory_slice = TrajectoryEmissionSlice(
            indices=self.emission_indices,
            emissions_g=self.pointwise_emissions_g,
            fuel_burn_per_segment=self.fuel_burn_per_segment,
            total_fuel_burn=total_fuel_burn,
        )
        lto_slice = EmissionSlice(
            indices=self.LTO_emission_indices,
            emissions_g=self.LTO_emissions_g,
        )
        apu_slice = (
            EmissionSlice(
                indices=self.APU_emission_indices,
                emissions_g=self.APU_emissions_g,
            )
            if config.emissions.apu_enabled
            else None
        )
        gse_slice = (
            EmissionSlice(indices=None, emissions_g=self.GSE_emissions_g)
            if config.emissions.gse_enabled
            else None
        )
        return EmissionsOutput(
            trajectory=trajectory_slice,
            lto=lto_slice,
            apu=apu_slice,
            gse=gse_slice,
            total=self.summed_emission_g,
            lifecycle_co2_g=lifecycle_adjustment,
        )

    def _initialize_field_controls(self):
        """Map dtype fields to metric flags so we can zero disabled outputs."""
        pmnvol_enabled = config.emissions.pmnvol_enabled and self._include_pmnvol_number
        self._field_controls = {
            'CO2': config.emissions.co2_enabled,
            'H2O': config.emissions.h2o_enabled,
            'HC': config.emissions.hc_enabled,
            'CO': config.emissions.co_enabled,
            'NOx': config.emissions.nox_enabled,
            'NO': config.emissions.nox_enabled,
            'NO2': config.emissions.nox_enabled,
            'HONO': config.emissions.nox_enabled,
            'PMvol': config.emissions.pmvol_enabled,
            'OCic': config.emissions.pmvol_enabled,
            'PMnvol': pmnvol_enabled,
            'PMnvolN': pmnvol_enabled,
            'PMnvolGMD': pmnvol_enabled,
            'SO2': config.emissions.sox_enabled,
            'SO4': config.emissions.sox_enabled,
        }

        self._active_fields = tuple(
            field
            for field in self.emission_indices.dtype.names
            if self._field_controls[field]
        )

    def _extract_lto_inputs(self) -> LTOInputs:
        """Return ordered fuel-flow and EI arrays for LTO data."""
        assert self.performance_model.lto_performance is not None
        return LTOInputs.from_performance(self.performance_model.lto_performance)

    def _thrust_band_labels(self, thrust_categories: np.ndarray) -> np.ndarray:
        """Translate numeric thrust codes into the L/H labels used by
        EI_PMvol_FuelFlow."""
        labels = np.full(thrust_categories.shape, 'H', dtype='<U1')
        labels[thrust_categories == 2] = 'L'
        return labels

    def _thrust_percentages_from_categories(self, thrust_categories: np.ndarray):
        """Convert thrust codes into representative ICAO mode percentages."""
        thrust_pct = np.full(thrust_categories.shape, 85.0, dtype=float)
        thrust_pct[thrust_categories == 2] = 7.0
        thrust_pct[thrust_categories == 3] = 30.0
        return thrust_pct

    def _new_emission_array(self, length: int):
        """Create a structured emission array with a consistent dtype footprint."""
        return np.full((), -1.0, dtype=self.__emission_dtype(length))

    def _assign_constant_indices(
        self, target_array, values: Mapping[str, float], idx_slice=slice(None)
    ):
        """Populate constant EI values for all enabled fields in a structured array."""
        controls = getattr(self, '_field_controls', {})
        for field, value in values.items():
            if field in target_array.dtype.names and controls.get(field, True):
                target_array[field][idx_slice] = value

    def _trajectory_slice(self):
        """Return the portion of the mission covered by trajectory emissions."""
        end = (
            self.Ntot if config.emissions.climb_descent_usage else self.Ntot - self.NDes
        )
        return slice(0, end)

    def _constant_species_values(self):
        """Return constant EI values that do not depend on thrust or atmosphere."""
        constants = {}
        if config.emissions.co2_enabled:
            co2_result = EI_CO2(self.fuel)
            constants['CO2'] = co2_result.EI_CO2
        if config.emissions.h2o_enabled:
            constants['H2O'] = EI_H2O(self.fuel)
        if config.emissions.sox_enabled:
            sox_result = EI_SOx(self.fuel)
            constants['SO2'] = sox_result.EI_SO2
            constants['SO4'] = sox_result.EI_SO4
        return constants

    def _atmospheric_state(
        self,
        altitude: np.ndarray,
        tas: np.ndarray,
        enabled: bool,
    ) -> AtmosphericState:
        """Compute temperature, pressure, and Mach profiles when needed."""
        if not enabled:
            return AtmosphericState(None, None, None)
        temps = temperature_at_altitude_isa_bada4(altitude)
        pressures = pressure_at_altitude_isa_bada4(altitude)
        mach = tas / np.sqrt(kappa * R_air * temps)
        return AtmosphericState(temps, pressures, mach)

    def _sls_equivalent_fuel_flow(
        self,
        enabled: bool,
        fuel_flow: np.ndarray,
        atmos_state: AtmosphericState,
    ):
        """Return sea-level static equivalent fuel flow"""
        if not enabled:
            return None
        return get_SLS_equivalent_fuel_flow(
            fuel_flow,
            atmos_state.pressure,
            atmos_state.temperature,
            atmos_state.mach,
            self.performance_model.number_of_engines,
        )

    def _compute_EI_HCCO(
        self,
        sls_equiv_fuel_flow,
        emission_matrix,
        performance_fuel_flow,
        atmos_state: AtmosphericState,
    ):
        """Shared helper for HC/CO EI calculations."""
        if (
            sls_equiv_fuel_flow is None
            or atmos_state.temperature is None
            or atmos_state.pressure is None
        ):
            raise RuntimeError(
                "HC/CO EI calculation requires atmosphere and SLS equivalent fuel flow."
            )
        return EI_HCCO(
            sls_equiv_fuel_flow,
            emission_matrix,
            performance_fuel_flow,
            Tamb=atmos_state.temperature,
            Pamb=atmos_state.pressure,
            cruiseCalc=True,
        )

    def _calculate_EI_PMvol(
        self,
        idx_slice: slice,
        thrust_categories: np.ndarray,
        fuel_flow: np.ndarray,
        hc_ei: np.ndarray | None,
    ):
        """Populate PMvol/OCic trajectory indices according to the configured method."""
        if (
            not config.emissions.pmvol_enabled
            or config.emissions.pmvol_method is PMvolMethod.NONE
        ):
            return
        if config.emissions.pmvol_method is PMvolMethod.FUEL_FLOW:
            thrust_labels = self._thrust_band_labels(thrust_categories)
            pmvol_ei, ocic_ei = EI_PMvol_FuelFlow(fuel_flow, thrust_labels)
        elif config.emissions.pmvol_method is PMvolMethod.FOA3:
            if hc_ei is None:
                raise RuntimeError("FOA3 PMvol calculation requires HC EIs.")
            thrust_pct = self._thrust_percentages_from_categories(thrust_categories)
            pmvol_ei, ocic_ei = EI_PMvol_FOA3(thrust_pct, hc_ei)
        else:
            raise NotImplementedError(
                f"EI_PMvol_method '{config.emissions.pmvol_method.value}' "
                "is not supported."
            )
        self.emission_indices['PMvol'][idx_slice] = pmvol_ei
        self.emission_indices['OCic'][idx_slice] = ocic_ei

    def _calculate_EI_PMnvol(
        self,
        idx_slice: slice,
        thrust_categories: np.ndarray,
        altitudes: np.ndarray,
        atmos_state: AtmosphericState,
        ac_performance: PerformanceModel,
    ):
        """Populate PMnvol indices for trajectory points."""
        method = config.emissions.pmnvol_method
        if method is PMnvolMethod.NONE:
            return
        if method is PMnvolMethod.MEEM:
            if (
                atmos_state.temperature is None
                or atmos_state.pressure is None
                or atmos_state.mach is None
            ):
                raise RuntimeError("MEEM PMnvol requires atmospheric state.")
            (
                self.emission_indices['PMnvolGMD'][idx_slice],
                self.emission_indices['PMnvol'][idx_slice],
                pmnvol_num,
            ) = PMnvol_MEEM(
                ac_performance.EDB_data,
                altitudes,
                atmos_state.temperature,
                atmos_state.pressure,
                atmos_state.mach,
            )
            if (
                self._include_pmnvol_number
                and 'PMnvolN' in self.emission_indices.dtype.names
            ):
                self.emission_indices['PMnvolN'][idx_slice] = pmnvol_num
            return

        if method is PMnvolMethod.SCOPE11:
            profile = self._scope11_profile(ac_performance)
            self.emission_indices['PMnvol'][idx_slice] = (
                self._map_mode_values_to_categories(profile['mass'], thrust_categories)
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
            return

        raise NotImplementedError(
            f"EI_PMnvol_method '{config.emissions.pmnvol_method.value}' "
            "is not supported."
        )

    def _get_LTO_TIMs(self):
        """Return the ICAO standard time-in-mode vector for Taxi → TO segments."""
        # TODO: Units?
        TIM_TakeOff = 0.7 * 60
        TIM_Climb = 2.2 * 60
        TIM_Approach = 4.0 * 60
        TIM_Taxi = 26.0 * 60
        durations = np.array([TIM_Taxi, TIM_Approach, TIM_Climb, TIM_TakeOff])
        if config.emissions.climb_descent_usage:
            durations[1:3] = 0.0
        return durations

    def _get_LTO_nox(self, thrust_categories, lto_inputs):
        """Fill LTO NOx and species splits while keeping auxiliary arrays in sync."""
        fuel_flows_LTO = lto_inputs.fuel_flow
        zeros = np.zeros_like(fuel_flows_LTO)
        if not config.emissions.nox_enabled:
            self.LTO_noProp = zeros
            self.LTO_no2Prop = zeros
            self.LTO_honoProp = zeros
            return

        if config.emissions.nox_method is EINOxMethod.NONE:
            self.LTO_noProp = zeros
            self.LTO_no2Prop = zeros
            self.LTO_honoProp = zeros
            return

        self.LTO_emission_indices['NOx'] = lto_inputs.EI_NOx
        (
            self.LTO_noProp,
            self.LTO_no2Prop,
            self.LTO_honoProp,
        ) = NOx_speciation(thrust_categories)
        self.LTO_emission_indices['NO'] = (
            self.LTO_emission_indices['NOx'] * self.LTO_noProp
        )
        self.LTO_emission_indices['NO2'] = (
            self.LTO_emission_indices['NOx'] * self.LTO_no2Prop
        )
        self.LTO_emission_indices['HONO'] = (
            self.LTO_emission_indices['NOx'] * self.LTO_honoProp
        )

    def _get_LTO_PMvol(self, fuel_flows_LTO, thrust_labels, lto_inputs):
        """Set PMvol/OCic EIs for the LTO cycle."""
        if config.emissions.pmvol_method is PMvolMethod.FUEL_FLOW:
            LTO_PMvol, LTO_OCic = EI_PMvol_FuelFlow(fuel_flows_LTO, thrust_labels)
        elif config.emissions.pmvol_method is PMvolMethod.FOA3:
            LTO_PMvol, LTO_OCic = EI_PMvol_FOA3(lto_inputs.thrust_pct, lto_inputs.EI_HC)
        elif config.emissions.pmvol_method is PMvolMethod.NONE:
            LTO_PMvol = LTO_OCic = np.zeros_like(fuel_flows_LTO)
        else:
            raise NotImplementedError(
                f"EI_PMvol_method '{config.emissions.pmvol_method.value}' "
                "is not supported."
            )
        self.LTO_emission_indices['PMvol'] = LTO_PMvol
        self.LTO_emission_indices['OCic'] = LTO_OCic

    def _get_LTO_PMnvol(self, ac_performance: PerformanceModel, fuel_flows_LTO):
        """Set PMnvol EIs for the four LTO thrust points."""
        method = config.emissions.pmnvol_method
        if method in (PMnvolMethod.FOA3, PMnvolMethod.MEEM):
            # TODO: FIX
            PMnvolEI = np.asarray(
                ac_performance.EDB_data['PMnvolEI_best_ICAOthrust'], dtype=float
            )
            PMnvolEIN = None
        elif method is PMnvolMethod.SCOPE11:
            profile = self._scope11_profile(ac_performance)
            PMnvolEI = profile['mass']
            PMnvolEIN = profile['number']
        elif method is PMnvolMethod.NONE:
            PMnvolEI = np.zeros_like(fuel_flows_LTO)
            PMnvolEIN = None
        else:
            raise ValueError(
                f"""Re-define PMnvol estimation method:
                pmnvolSwitch = {config.emissions.pmnvol_method.value}"""
            )

        self.LTO_emission_indices['PMnvol'] = PMnvolEI
        if (
            self._include_pmnvol_number
            and PMnvolEIN is not None
            and 'PMnvolN' in self.LTO_emission_indices.dtype.names
        ):
            self.LTO_emission_indices['PMnvolN'] = PMnvolEIN

    def _wnsf_index(self, wnsf: str) -> int:
        """Map user-friendly WNSF labels to the internal lookup order."""
        mapping = {'wide': 0, 'narrow': 1, 'small': 2, 'freight': 3}
        idx = mapping.get(wnsf.lower())
        if idx is None:
            raise ValueError(
                "Invalid WNSF code; must be one of 'wide','narrow','small','freight'"
            )
        return idx

    def _gse_nominal_profile(self, idx: int):
        """Return nominal per-cycle GSE emissions for the requested aircraft class."""
        return {
            'CO2': [58e3, 18e3, 10e3, 58e3][idx],
            'NOx': [0.9e3, 0.4e3, 0.3e3, 0.9e3][idx],
            'HC': [0.07e3, 0.04e3, 0.03e3, 0.07e3][idx],
            'CO': [0.3e3, 0.15e3, 0.1e3, 0.3e3][idx],
            'PM10': [0.055e3, 0.025e3, 0.020e3, 0.055e3][idx],
        }

    def _scope11_profile(self, ac_performance):
        """Cache SCOPE11 lookups so we do the work only once."""
        # TODO: FIX
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
        if self._include_pmnvol_number:
            fields.append(('PMnvolN', np.float64, n))
        return fields

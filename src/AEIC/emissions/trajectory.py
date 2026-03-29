# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from AEIC.config import config
from AEIC.config.emissions import (
    ClimbDescentMode,
    EINOxMethod,
    nvpmMethod,
)
from AEIC.performance.models import BasePerformanceModel
from AEIC.trajectories.trajectory import Trajectory
from AEIC.types import Species, SpeciesValues

from .ei.hcco import EI_HCCO
from .ei.nox import BFFM2_EINOx
from .ei.nvpm import nvPM_MEEM
from .types import AtmosphericState, EmissionsSubset
from .utils import (
    constant_species_values,
    get_SLS_equivalent_fuel_flow,
)

if TYPE_CHECKING:
    from AEIC.performance.types import LTOPerformance
    from AEIC.types import Fuel


def get_trajectory_emissions(
    pm: BasePerformanceModel,
    traj: Trajectory,
    fuel_burn_per_segment: np.ndarray,
    fuel: Fuel,
) -> EmissionsSubset[np.ndarray]:
    """
    Calculate emissions for each flight trajectory point.
    """

    # Output emissions indices and emissions.
    indices = SpeciesValues[np.ndarray]()
    emissions = SpeciesValues[np.ndarray]()

    # Set up constant emission index values.
    for species, value in constant_species_values(fuel).items():
        if species in config.emissions.enabled_species:
            indices[species] = np.full(len(traj), value, dtype=float)

    atmos_state = AtmosphericState(traj.altitude, traj.true_airspeed)
    sls_equiv_fuel_flow = get_SLS_equivalent_fuel_flow(
        fuel_flow=traj.fuel_flow,
        Pamb=atmos_state.pressure,
        Tamb=atmos_state.temperature,
        mach_number=atmos_state.mach,
        n_eng=pm.number_of_engines,
    )

    if Species.NOx in config.emissions.enabled_species:
        indices.update(compute_EI_NOx(pm.lto, atmos_state, sls_equiv_fuel_flow))

    if Species.HC in config.emissions.enabled_species:
        indices[Species.HC] = EI_HCCO(
            sls_equiv_fuel_flow,
            pm.lto.EI_HC,
            pm.lto.fuel_flow,
            Tamb=atmos_state.temperature,
            Pamb=atmos_state.pressure,
        )

    if Species.CO in config.emissions.enabled_species:
        indices[Species.CO] = EI_HCCO(
            sls_equiv_fuel_flow,
            pm.lto.EI_CO,
            pm.lto.fuel_flow,
            Tamb=atmos_state.temperature,
            Pamb=atmos_state.pressure,
        )

    if config.emissions.nvpm_enabled:
        indices.update(_calculate_EI_nvPM(pm, traj.altitude, atmos_state))

    for species in indices.keys():
        emissions[species] = indices[species] * fuel_burn_per_segment

    idx_slice = _trajectory_slice(traj)
    for species in indices.keys():
        indices[species][: idx_slice.start] = 0.0
        indices[species][idx_slice.stop :] = 0.0
        emissions[species][: idx_slice.start] = 0.0
        emissions[species][idx_slice.stop :] = 0.0
    total_fuel_burn = np.sum(fuel_burn_per_segment[idx_slice])

    return EmissionsSubset(
        indices=indices,
        emissions=emissions,
        fuel_burn=total_fuel_burn,
    )


def compute_EI_NOx(
    lto: LTOPerformance,
    atmos_state: AtmosphericState,
    sls_equiv_fuel_flow: np.ndarray,
) -> SpeciesValues[np.ndarray]:
    """Fill NOₓ-related EI arrays according to the selected method."""
    indices = SpeciesValues[np.ndarray]()
    match config.emissions.nox_method:
        case EINOxMethod.NONE:
            pass
        case EINOxMethod.BFFM2:
            bffm2_result = BFFM2_EINOx(
                sls_equiv_fuel_flow=sls_equiv_fuel_flow,
                EI_NOx_matrix=lto.EI_NOx,
                fuelflow_performance=lto.fuel_flow,
                Pamb=atmos_state.pressure,
                Tamb=atmos_state.temperature,
            )
            indices[Species.NOx] = bffm2_result.NOxEI
            indices[Species.NO] = bffm2_result.NOEI
            indices[Species.NO2] = bffm2_result.NO2EI
            indices[Species.HONO] = bffm2_result.HONOEI
        case EINOxMethod.P3T3:
            print("P3T3 method not implemented yet..")
        case _:
            raise NotImplementedError(
                f"EI_NOx_method '{config.emissions.nox_method.value}' is not supported."
            )
    return indices


def _calculate_EI_nvPM(
    pm: BasePerformanceModel,
    altitudes: np.ndarray,
    atmos_state: AtmosphericState | None = None,
) -> SpeciesValues[np.ndarray]:
    """Populate nvPM indices for trajectory points."""
    indices = SpeciesValues[np.ndarray]()

    match config.emissions.nvpm_method:
        case nvpmMethod.NONE:
            pass

        case nvpmMethod.MEEM:
            assert atmos_state is not None, (
                'Atmospheric state is required for MEEM nvPM.'
            )
            (
                indices[Species.nvPM],
                nvpm_num,
            ) = nvPM_MEEM(
                pm.edb,
                altitudes,
                atmos_state.temperature,
                atmos_state.pressure,
                atmos_state.mach,
            )
            if Species.nvPM_N in config.emissions.enabled_species:
                indices[Species.nvPM_N] = nvpm_num

        case _:
            raise NotImplementedError(
                f"EI_nvPM_method '{config.emissions.nvpm_method.value}' "
                "is not supported."
            )

    return indices


def _trajectory_slice(traj: Trajectory) -> slice:
    if config.emissions.climb_descent_mode != ClimbDescentMode.TRAJECTORY:
        # Climb and descent segments are handled in the LTO emissions
        # calculations.
        return slice(traj.n_climb, len(traj) - traj.n_descent)
    else:
        return slice(0, len(traj))

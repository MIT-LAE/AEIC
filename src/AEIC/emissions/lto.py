# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from typing import TYPE_CHECKING

from AEIC.config import config
from AEIC.config.emissions import (
    ClimbDescentMode,
    EINOxMethod,
    EInvPMMethod,
)
from AEIC.emissions.types import EmissionsSubset
from AEIC.performance.edb import EDBEntry
from AEIC.performance.models.base import BasePerformanceModel
from AEIC.performance.types import ThrustMode, ThrustModeValues
from AEIC.types import Species, SpeciesValues
from AEIC.units import MINUTES_TO_SECONDS

from .ei.nox import NOx_speciation
from .utils import constant_species_values, scope11_profile

if TYPE_CHECKING:
    from AEIC.performance.types import LTOPerformance
    from AEIC.types import Fuel


def get_LTO_emissions(
    performance_model: BasePerformanceModel, fuel: Fuel
) -> EmissionsSubset[ThrustModeValues]:
    """
    Compute Landing-and-Takeoff cycle emission indices and quantities.
    """
    lto_indices = SpeciesValues[ThrustModeValues]()
    lto_emissions = SpeciesValues[ThrustModeValues]()

    lto_data = performance_model.lto

    for species, value in constant_species_values(fuel).items():
        if species in config.emissions.enabled_species:
            lto_indices[species] = ThrustModeValues({m: value for m in ThrustMode})

    lto_indices.update(_lto_nox(lto_data))

    if Species.HC in config.emissions.enabled_species:
        lto_indices[Species.HC] = lto_data.EI_HC
    if Species.CO in config.emissions.enabled_species:
        lto_indices[Species.CO] = lto_data.EI_CO

    if Species.nvPM in config.emissions.enabled_species:
        lto_indices.update(_lto_nvpm(performance_model.edb))

    lto_fuel_burn = _LTO_TIMS * lto_data.fuel_flow
    if config.emissions.climb_descent_mode != ClimbDescentMode.LTO:
        # These are handled in the trajectory emissions calculation.
        for mode in [ThrustMode.APPROACH, ThrustMode.CLIMB]:
            lto_fuel_burn[mode] = 0.0
        for species in lto_indices:
            # Need to be careful about copying these and making them mutable,
            # since a lot of them may come from immutable configuration data.
            lto_indices[species] = lto_indices[species].copy(mutable=True)
            for mode in [ThrustMode.APPROACH, ThrustMode.CLIMB]:
                lto_indices[species][mode] = 0.0

    for species in lto_indices.keys():
        lto_emissions[species] = lto_indices[species] * lto_fuel_burn

    return EmissionsSubset(lto_indices, lto_emissions, lto_fuel_burn.sum())


_LTO_TIMS = ThrustModeValues(
    {
        ThrustMode.IDLE: 26.0 * MINUTES_TO_SECONDS,
        ThrustMode.APPROACH: 4.0 * MINUTES_TO_SECONDS,
        ThrustMode.CLIMB: 2.2 * MINUTES_TO_SECONDS,
        ThrustMode.TAKEOFF: 0.7 * MINUTES_TO_SECONDS,
    }
)
"""ICAO standard time-in-mode vector for Taxi → TO segments."""


def _lto_nox(lto_data: LTOPerformance) -> SpeciesValues[ThrustModeValues]:
    """Calculate LTO total NOₓ and NOₓ speciation."""
    indices = SpeciesValues[ThrustModeValues]()

    lto_nox = lto_data.EI_NOx

    if (
        not config.emissions.nox_enabled
        or config.emissions.nox_method is EINOxMethod.NONE
    ):
        return indices

    indices[Species.NOx] = lto_nox
    speciation = NOx_speciation()
    indices[Species.NO] = lto_nox * speciation.no
    indices[Species.NO2] = lto_nox * speciation.no2
    indices[Species.HONO] = lto_nox * speciation.hono
    return indices


def _lto_nvpm(edb: EDBEntry) -> SpeciesValues[ThrustModeValues]:
    """Calculate LTO nvPM emission indices."""
    indices = SpeciesValues[ThrustModeValues]()
    nvpm_num: ThrustModeValues | None = None

    match config.emissions.nvpm_method:
        case EInvPMMethod.MEEM:
            # Use nvPM EI/EInum from EDB if they exist, otherwise use SCOPE11
            use_edb_nvpm = all(
                edb.nvPM_mass_matrix[mode] > 0.0 and edb.nvPM_num_matrix[mode] > 0.0
                for mode in ThrustMode
            )
            if use_edb_nvpm:
                nvpm_mass = edb.nvPM_mass_matrix.copy() * 1e-3  # mg/kg to g/kg
                nvpm_num = edb.nvPM_num_matrix.copy()
            else:
                profile = scope11_profile(edb)
                nvpm_mass = profile.mass.copy()
                if profile.number is not None:
                    nvpm_num = profile.number.copy()
        case EInvPMMethod.NONE:
            nvpm_mass = ThrustModeValues(0.0)
        case _:
            raise ValueError(
                f"Unsupported nvPM estimation method:\
                {config.emissions.nvpm_method.value}"
            )

    indices[Species.nvPM] = nvpm_mass
    if nvpm_num is not None and Species.nvPM_N in config.emissions.enabled_species:
        indices[Species.nvPM_N] = nvpm_num

    return indices

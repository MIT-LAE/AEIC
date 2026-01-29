# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from AEIC.config import config
from AEIC.config.emissions import (
    ClimbDescentMode,
    EINOxMethod,
    PMnvolMethod,
    PMvolMethod,
)
from AEIC.performance.models.base import BasePerformanceModel
from AEIC.performance.utils.edb import EDBEntry
from AEIC.types import (
    EmissionsDict,
    EmissionsSubset,
    ModeValues,
    Species,
    ThrustLabelArray,
    ThrustMode,
)
from AEIC.units import MINUTES_TO_SECONDS

from .ei.nox import NOx_speciation
from .ei.pmvol import EI_PMvol_FOA3, EI_PMvol_FuelFlow
from .utils import constant_species_values, scope11_profile

if TYPE_CHECKING:
    from AEIC.types import Fuel, LTOPerformance


def get_LTO_emissions(
    performance_model: BasePerformanceModel, fuel: Fuel
) -> EmissionsSubset[ModeValues]:
    """
    Compute Landing-and-Takeoff cycle emission indices and quantities.
    """
    lto_indices = EmissionsDict[ModeValues]()
    lto_emissions = EmissionsDict[ModeValues]()

    lto_data = performance_model.lto

    for species, value in constant_species_values(fuel).items():
        if species in config.emissions.enabled_species:
            lto_indices[species] = ModeValues({m: value for m in ThrustMode})

    lto_indices.update(_lto_nox(lto_data))

    if Species.HC in config.emissions.enabled_species:
        lto_indices[Species.HC] = lto_data.EI_HC
    if Species.CO in config.emissions.enabled_species:
        lto_indices[Species.CO] = lto_data.EI_CO

    if Species.PMvol in config.emissions.enabled_species:
        lto_indices.update(_lto_pmvol(lto_data))

    if Species.PMnvol in config.emissions.enabled_species:
        lto_indices.update(_lto_pmnvol(performance_model.edb))
    lto_indices[Species.PMnvolGMD] = ModeValues(0.0)

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


_LTO_TIMS = ModeValues(
    {
        ThrustMode.IDLE: 26.0 * MINUTES_TO_SECONDS,
        ThrustMode.APPROACH: 4.0 * MINUTES_TO_SECONDS,
        ThrustMode.CLIMB: 2.2 * MINUTES_TO_SECONDS,
        ThrustMode.TAKEOFF: 0.7 * MINUTES_TO_SECONDS,
    }
)
"""ICAO standard time-in-mode vector for Taxi → TO segments."""


def _lto_nox(lto_data: LTOPerformance) -> EmissionsDict[ModeValues]:
    """Fill LTO NOx and species splits while keeping auxiliary arrays in sync."""
    indices = EmissionsDict[ModeValues]()

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


def _lto_pmvol(lto_data: LTOPerformance) -> EmissionsDict[ModeValues]:
    """Set PMvol/OCic EIs for the LTO cycle."""

    indices = EmissionsDict[ModeValues]()
    if config.emissions.pmvol_method is PMvolMethod.NONE:
        indices[Species.PMvol] = ModeValues()
        indices[Species.OCic] = ModeValues()
        return indices

    match config.emissions.pmvol_method:
        case PMvolMethod.FUEL_FLOW:
            thrust_labels = ThrustLabelArray(
                np.array([m.label for m in lto_data.fuel_flow.keys()])
            )
            tmp_PMvol, tmp_OCic = EI_PMvol_FuelFlow(
                lto_data.fuel_flow.as_array(), thrust_labels
            )
            LTO_PMvol = ModeValues(tmp_PMvol)
            LTO_OCic = ModeValues(tmp_OCic)
        case PMvolMethod.FOA3:
            tmp_PMvol, tmp_OCic = EI_PMvol_FOA3(
                lto_data.thrust_pct.as_array(), lto_data.EI_HC.as_array()
            )
            LTO_PMvol = ModeValues(tmp_PMvol)
            LTO_OCic = ModeValues(tmp_OCic)
        case _:
            raise NotImplementedError(
                f"EI_PMvol_method '{config.emissions.pmvol_method.value}' "
                "is not supported."
            )
    indices[Species.PMvol] = LTO_PMvol
    indices[Species.OCic] = LTO_OCic

    return indices


def _lto_pmnvol(edb: EDBEntry) -> EmissionsDict[ModeValues]:
    """Set PMnvol EIs for the four LTO thrust points."""
    indices = EmissionsDict[ModeValues]()
    PMnvolEIN = None
    match config.emissions.pmnvol_method:
        case PMnvolMethod.FOA3 | PMnvolMethod.MEEM:
            # TODO: This doesn't exist. Where should it come from?
            # PMnvolEI = edb.PMnvolEI_best_ICAOthrust.asarray()
            # TODO: This is just a temporary placeholder until this is
            # implemented properly.
            PMnvolEI = ModeValues(0.0)
        case PMnvolMethod.SCOPE11:
            profile = scope11_profile(edb)
            PMnvolEI = profile.mass.copy()
            if profile.number is not None:
                PMnvolEIN = profile.number.copy()
        case PMnvolMethod.NONE:
            PMnvolEI = ModeValues(0.0)
        case _:
            raise ValueError(
                f"""Re-define PMnvol estimation method:
                pmnvolSwitch = {config.emissions.pmnvol_method.value}"""
            )

    indices[Species.PMnvol] = PMnvolEI
    if PMnvolEIN is not None and Species.PMnvolN in config.emissions.enabled_species:
        indices[Species.PMnvolN] = PMnvolEIN

    return indices

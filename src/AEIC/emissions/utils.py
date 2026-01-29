# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from AEIC.config import config
from AEIC.performance.utils.edb import EDBEntry
from AEIC.types import EmissionsDict, ModeValues, Species

from .ei.co2 import EI_CO2
from .ei.h2o import EI_H2O
from .ei.pmnvol import calculate_PMnvolEI_scope11
from .ei.sox import EI_SOx

if TYPE_CHECKING:
    from AEIC.types import Fuel


@dataclass
class Scope11Profile:
    mass: ModeValues
    number: ModeValues | None


@functools.cache
def scope11_profile(edb: EDBEntry) -> Scope11Profile:
    mass = calculate_PMnvolEI_scope11(edb.SN_matrix, edb.engine_type, edb.BP_Ratio)
    # TODO: Fix.
    # number = edb.PMnvolEIN_best_ICAOthrust
    number = None
    return Scope11Profile(
        mass,
        None if number is None else ModeValues(number),
    )


def constant_species_values(fuel: Fuel) -> EmissionsDict[float]:
    """Return constant EI values that do not depend on thrust or atmosphere."""
    constants = EmissionsDict[float]()

    if Species.CO2 in config.emissions.enabled_species:
        constants[Species.CO2] = EI_CO2(fuel)
    if Species.H2O in config.emissions.enabled_species:
        constants[Species.H2O] = EI_H2O(fuel)
    if (
        Species.SO2 in config.emissions.enabled_species
        or Species.SO4 in config.emissions.enabled_species
    ):
        sox_result = EI_SOx(fuel)
        if Species.SO2 in config.emissions.enabled_species:
            constants[Species.SO2] = sox_result.EI_SO2
        if Species.SO4 in config.emissions.enabled_species:
            constants[Species.SO4] = sox_result.EI_SO4

    return constants

from AEIC.config import config
from AEIC.config.emissions import PMnvolMethod
from AEIC.emissions.types import EmissionsSubset
from AEIC.performance.apu import APU
from AEIC.performance.types import ThrustMode, ThrustModeValues
from AEIC.types import Fuel, Species, SpeciesValues

from .ei.nox import NOx_speciation


def get_APU_emissions(
    lto_indices: SpeciesValues[ThrustModeValues],
    apu: APU,
    fuel: Fuel,
    apu_time: float = 900,
) -> EmissionsSubset[float]:
    """
    Calculate APU emissions using time in modes and given APU data.

    Parameters
    ----------
    lto_indices : ndarray
        previously determined LTO emission indices
    apu: APU
        APU data
    fuel: Fuel
        Fuel data
    apu_time: float
        Time in mode for APU; default value = 900 seconds (Stettler et al. 2011)
    """

    indices = SpeciesValues[float]()
    emissions = SpeciesValues[float]()

    apu_running = apu.fuel_kg_per_s != 0.0
    apu_fuel_burn = apu.fuel_kg_per_s * apu_time

    # SOx
    indices[Species.SO2] = (
        lto_indices[Species.SO2][ThrustMode.IDLE] if apu_running else 0.0
    )
    indices[Species.SO4] = (
        lto_indices[Species.SO4][ThrustMode.IDLE] if apu_running else 0.0
    )
    indices[Species.SOx] = indices[Species.SO2] + indices[Species.SO4]

    # Particulate‐matter breakdown (deterministic BC fraction of 0.95)
    APU_PM10 = max(apu.PM10_g_per_kg - indices[Species.SO4], 0.0)
    bc_prop = 0.95
    indices[Species.PMnvol] = APU_PM10 * bc_prop
    indices[Species.PMvol] = APU_PM10 - indices[Species.PMnvol]

    if config.emissions.pmnvol_method in (PMnvolMethod.SCOPE11, PMnvolMethod.MEEM):
        indices[Species.PMnvolN] = 0.0
    indices[Species.PMnvolGMD] = 0.0
    indices[Species.OCic] = 0.0

    # NO/NO₂/HONO speciation.
    # TODO: Is using the idle values here right?
    nox_speciation = NOx_speciation()
    indices[Species.NO] = apu.PM10_g_per_kg * nox_speciation.no[ThrustMode.IDLE]
    indices[Species.NO2] = apu.PM10_g_per_kg * nox_speciation.no2[ThrustMode.IDLE]
    indices[Species.HONO] = apu.PM10_g_per_kg * nox_speciation.hono[ThrustMode.IDLE]

    indices[Species.NOx] = apu.NOx_g_per_kg
    indices[Species.HC] = apu.HC_g_per_kg
    indices[Species.CO] = apu.CO_g_per_kg

    # H₂O.
    indices[Species.H2O] = fuel.EI_H2O

    # CO₂ via mass balance.
    if apu_running:
        co2_ei_nom = 3160
        nvol_carb_cont = 0.95

        co2 = co2_ei_nom
        co2 -= (44 / 28) * indices[Species.CO]
        co2 -= (44 / (82 / 5)) * indices[Species.HC]
        co2 -= (44 / (55 / 4)) * indices[Species.PMvol]
        co2 -= (44 / 12) * nvol_carb_cont * indices[Species.PMnvol]
        indices[Species.CO2] = co2
    else:
        indices[Species.CO2] = 0.0

    for species in indices:
        emissions[species] = indices[species] * apu_fuel_burn

    return EmissionsSubset(indices, emissions, apu_fuel_burn)

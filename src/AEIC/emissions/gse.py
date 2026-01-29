# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from typing import TYPE_CHECKING

from AEIC.types import AircraftClass, EmissionsDict, EmissionsSubset, Species
from AEIC.units import KG_TO_GRAMS, PPM

from .ei.co2 import EI_CO2
from .ei.h2o import EI_H2O

if TYPE_CHECKING:
    from AEIC.types import Fuel


def get_GSE_emissions(
    aircraft_class: AircraftClass, fuel: Fuel
) -> EmissionsSubset[float]:
    """Calculate Ground Service Equipment emissions based on aircraft class.
    Returns emissions dictionary and fuel burn."""
    gse = EmissionsDict[float]()

    nominal, pm_core = _gse_nominal_profile(aircraft_class)
    for species in [Species.CO2, Species.NOx, Species.HC, Species.CO]:
        gse[species] = nominal[species]

    gse_fuel = gse[Species.CO2] / EI_CO2(fuel)

    gse[Species.H2O] = EI_H2O(fuel) * gse_fuel

    # NOx split.
    gse[Species.NO] = gse[Species.NOx] * 0.90
    gse[Species.NO2] = gse[Species.NOx] * 0.09
    gse[Species.HONO] = gse[Species.NOx] * 0.01

    # Sulfate / SO2 fraction (independent of aircraft class).
    GSE_FSC = 5.0 * PPM  # fuel‐sulfur concentration (ppm)
    GSE_EPS = 0.02  # fraction → sulfate
    # Molecular weights.
    MWT_O2 = 16.0 * 2
    MWT_SO2 = 32.0 + 16.0 * 2
    MWT_SO4 = 32.0 + 16.0 * 4
    # g SO4 per kg fuel:
    gse[Species.SO4] = GSE_FSC * KG_TO_GRAMS * GSE_EPS * (MWT_SO4 / MWT_O2)
    # g SO2 per kg fuel:
    gse[Species.SO2] = GSE_FSC * KG_TO_GRAMS * (1.0 - GSE_EPS) * (MWT_SO2 / MWT_O2)

    # Subtract sulfate from the core PM₁₀ then split 50:50.
    pm_minus_so4 = pm_core - gse[Species.SO4]
    gse[Species.PMvol] = pm_minus_so4 * 0.5
    gse[Species.PMnvol] = pm_minus_so4 * 0.5

    # No PMnvolN or PMnvolGMD or OCic.
    gse[Species.PMnvolN] = 0.0
    gse[Species.PMnvolGMD] = 0.0
    gse[Species.OCic] = 0.0

    return EmissionsSubset(emissions=gse, fuel_burn=gse_fuel)


def _gse_nominal_profile(
    aircraft_class: AircraftClass,
) -> tuple[EmissionsDict[float], float]:
    """Return nominal per-cycle GSE emissions for the requested aircraft class.

    Return value is an emissions dictionary for CO2, NOx, HC, CO, and a
    single value for PM10."""
    match aircraft_class:
        case AircraftClass.WIDE:
            return (
                EmissionsDict[float](
                    {
                        Species.CO2: 58e3,
                        Species.NOx: 0.9e3,
                        Species.HC: 0.07e3,
                        Species.CO: 0.3e3,
                    }
                ),
                0.055e3,
            )
        case AircraftClass.NARROW:
            return (
                EmissionsDict[float](
                    {
                        Species.CO2: 18e3,
                        Species.NOx: 0.4e3,
                        Species.HC: 0.04e3,
                        Species.CO: 0.15e3,
                    }
                ),
                0.025e3,
            )
        case AircraftClass.SMALL:
            return (
                EmissionsDict[float](
                    {
                        Species.CO2: 10e3,
                        Species.NOx: 0.3e3,
                        Species.HC: 0.03e3,
                        Species.CO: 0.1e3,
                    }
                ),
                0.020e3,
            )
        case AircraftClass.FREIGHT:
            return (
                EmissionsDict[float](
                    {
                        Species.CO2: 58e3,
                        Species.NOx: 0.9e3,
                        Species.HC: 0.07e3,
                        Species.CO: 0.3e3,
                    }
                ),
                0.055e3,
            )

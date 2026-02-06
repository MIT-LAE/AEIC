from dataclasses import dataclass

from AEIC.types import Fuel


@dataclass(frozen=True)
class SOxEmissionResult:
    """Structured SOₓ emission indices."""

    EI_SO2: float
    EI_SO4: float


# Molecular weights
MW_SO2 = 64.0
MW_SO4 = 96.0
MW_S = 32.0


def EI_SOx(fuel: Fuel) -> SOxEmissionResult:
    """
    Calculate universal SOₓ emissions indices (SO2EI and SO4EI).

    Parameters
    ----------

    fuel : Mapping[str, Any]
        Fuel information (input from toml file)

    Returns
    -------
    SOxEmissionResult
        Structured SO₂/SO₄ emissions indices [g/kg fuel]
    """

    # Convert fuel sulfur content (ppm) to fraction.
    sulfur_frac = fuel.fuel_sulfur_content_nom / 1.0e6

    # Compute emissions indices (g/kg fuel).
    return SOxEmissionResult(
        EI_SO2=sulfur_frac * (1 - fuel.sulfate_yield_nom) * MW_SO2 / MW_S * 1.0e3,
        EI_SO4=sulfur_frac * fuel.sulfate_yield_nom * MW_SO4 / MW_S * 1.0e3,
    )

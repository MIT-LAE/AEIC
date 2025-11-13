from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CO2EmissionResult:
    """Named return values for CO₂ EI calculations."""

    EI_CO2: float
    nvolCarbCont: float


def EI_CO2(fuel: Mapping[str, Any]) -> CO2EmissionResult:
    """
    Calculate carbon-balanced CO2 emissions index (EI).

    Parameters
    ----------
    fuel : Mapping[str, Any]
        Fuel information (input from toml file)

    Returns
    -------
    CO2EmissionResult
        Structured CO2 emissions metadata.
    """

    # if mcs == 1:
    #     CO2EInom = trirnd(3148, 3173, 3160, rv)
    #     nvolCarbCont = 0.9 + (0.98 - 0.9) * np.random.rand()
    # else:
    CO2EInom = fuel['EI_CO2']
    nvolCarbCont = fuel['nvolCarbCont']

    return CO2EmissionResult(EI_CO2=CO2EInom, nvolCarbCont=nvolCarbCont)

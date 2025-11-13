from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SOxEmissionResult:
    """Structured SOx emission indices."""

    EI_SO2: float
    EI_SO4: float


def EI_SOx(fuel: Mapping[str, Any]) -> SOxEmissionResult:
    """
    Calculate universal SOx emissions indices (SO2EI and SO4EI).

    Parameters
    ----------

    fuel : Mapping[str, Any]
        Fuel information (input from toml file)

    Returns
    -------
    SOxEmissionResult
        Structured SO2/SO4 emissions indices [g/kg fuel]
    """
    # Nominal values
    FSCnom = fuel['FSCnom']
    Epsnom = fuel['Epsnom']

    # Apply MC for FSC
    # if mcsFSC == 1:
    #     FSC = trirnd(500, 700, FSCnom, rvFSC)
    # else:
    FSC = FSCnom

    # Apply MC for Eps
    # if mcsEps == 1:
    #     Eps = trirnd(0.005, 0.05, Epsnom, rvEps)
    # else:
    Eps = Epsnom

    # Molecular weights
    MW_SO2 = 64.0
    MW_SO4 = 96.0
    MW_S = 32.0

    # Compute emissions indices (g/kg fuel)
    SO4EI = 1e3 * ((FSC / 1e6) * Eps * MW_SO4) / MW_S
    SO2EI = 1e3 * ((FSC / 1e6) * (1 - Eps) * MW_SO2) / MW_S

    return SOxEmissionResult(EI_SO2=SO2EI, EI_SO4=SO4EI)

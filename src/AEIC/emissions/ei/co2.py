# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AEIC.types import Fuel


def EI_CO2(fuel: Fuel) -> float:
    """
    Calculate carbon-balanced CO2 emissions index (EI).

    Parameters
    ----------
    fuel : Fuel
        Fuel information (input from toml file)

    Returns
    -------
    CO2_EI : float
        CO2 emissions index [g/kg fuel]
    """

    return fuel.EI_CO2

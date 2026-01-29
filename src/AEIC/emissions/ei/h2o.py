# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AEIC.types import Fuel


def EI_H2O(fuel: Fuel) -> float:
    """
    Calculate H2O emissions index (EI).

    Parameters
    ----------
    fuel : dictionary
        Fuel information (input from toml file)

    Returns
    -------
    H2O_EI : float
        H2O emissions index [g/kg fuel]
    """

    return fuel.EI_H2O

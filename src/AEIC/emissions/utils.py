# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from AEIC.config import config
from AEIC.performance.edb import EDBEntry
from AEIC.performance.types import ThrustMode, ThrustModeArray, ThrustModeValues
from AEIC.types import Species, SpeciesValues

from .ei.pmnvol import calculate_PMnvolEI_scope11
from .ei.sox import EI_SOx

if TYPE_CHECKING:
    from AEIC.types import Fuel


@dataclass
class Scope11Profile:
    mass: ThrustModeValues
    number: ThrustModeValues | None


@functools.cache
def scope11_profile(edb: EDBEntry) -> Scope11Profile:
    mass = calculate_PMnvolEI_scope11(edb.SN_matrix, edb.engine_type, edb.BP_Ratio)
    # TODO: Fix.
    # number = edb.PMnvolEIN_best_ICAOthrust
    number = None
    return Scope11Profile(
        mass,
        None if number is None else ThrustModeValues(number),
    )


def constant_species_values(fuel: Fuel) -> SpeciesValues[float]:
    """Return constant EI values that do not depend on thrust or atmosphere."""
    constants = SpeciesValues[float]()

    if Species.CO2 in config.emissions.enabled_species:
        constants[Species.CO2] = fuel.EI_CO2
    if Species.H2O in config.emissions.enabled_species:
        constants[Species.H2O] = fuel.EI_H2O
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


def get_thrust_cat_cruise(
    ff_eval: np.ndarray, ff_cal: ThrustModeValues
) -> ThrustModeArray:
    """Classify each cruise fuel-flow value into a discrete thrust-setting category.

    Parameters
    ----------
    ff_eval : numpy.ndarray
        Fuel-flow values to be evaluated
        Shape ``(n_times,)``.
    ff_cal : numpy.ndarray
        Calibration fuel-flow points: at least the first three cruise
        calibration points are required (typically Idle, Approach,
        Climb/Take-off).

    Returns
    -------
    thrustCat : numpy.ndarray, dtype=int (ICAOThrustMode)
        Integer category codes, same length as ``ff_eval``:

    Raises
    ------
    ValueError
        If the size/length constraints on ``ff_eval`` or ``ff_cal`` are violated.

    Notes
    -----
    Two mid-points define the category boundaries
    ``lowLimit  = (ff_cal[0] + ff_cal[1]) / 2``
    ``approachLimit = (ff_cal[1] + ff_cal[2]) / 2``

    * Idle          : ``ff_eval ≤ lowLimit``       →  ``IDLE``
    * Take-off/Climb: ``ff_eval >  approachLimit`` →  ``TAKEOFF_CLIMB``
    * Approach      : remainder                    →  ``APPROACH``

    """

    # Define thresholds from the first three calibration points
    lowLimit = (ff_cal[ThrustMode.IDLE] + ff_cal[ThrustMode.APPROACH]) / 2.0
    approachLimit = (ff_cal[ThrustMode.APPROACH] + ff_cal[ThrustMode.CLIMB]) / 2.0

    return ThrustModeArray(
        np.select(
            [
                ff_eval <= lowLimit,
                ff_eval > approachLimit,
            ],
            [
                ThrustMode.IDLE,
                ThrustMode.CLIMB,
            ],
            default=ThrustMode.APPROACH,
        )
    )


def get_SLS_equivalent_fuel_flow(
    fuel_flow: np.ndarray,
    Pamb: np.ndarray,
    Tamb: np.ndarray,
    mach_number: np.ndarray,
    z: float = 3.8,
    P_SL: float = 101325.0,
    T_SL: float = 288.15,
    n_eng: int = 2,
):
    """
    Convert in-flight fuel flow to its sea-level-static (SLS) equivalent
    using the **Fuel-Flow Method 2** correction (Eq. 40 in
    DuBois & Paynter, 2006).

    Parameters
    ----------
    fuel_flow : numpy.ndarray
        In-flight (altitude) fuel flow, :math:`\\dot m_{f,\\text{alt}}`  .
    Pamb : numpy.ndarray
        Ambient static pressure, :math:`P_{amb}`, in **Pa**.
    mach_number : numpy.ndarray
        Flight Mach number, :math:`M`.
    z : float, default 3.8
        Exponent applied to the temperature ratio :math:`\\theta`; default
        value 3.8.
    P_SL : float, default 101 325 Pa
        Sea-level standard static pressure, :math:`P_{SL}` (ISA).

    Returns
    -------
    Wf_SL : float or numpy.ndarray
        Sea-level-static equivalent fuel flow, :math:`\\dot m_{f,SL}`
        in **kg s⁻¹**.

    References
    ----------
    DuBois, D. & Paynter, G. (2006). *Fuel Flow Method 2 for Estimating
    Aircraft Emissions*. SAE Technical Paper 2006-01-1987.
    """
    # -----------------------------
    # Convert Wf_alt → Wf_SL (Eq. (40) in DuBois and Paynter 2006):
    # -----------------------------
    # δ_amb
    delta_amb = Pamb / P_SL
    # temperature ratio (isentropic estimate)
    theta_amb = Tamb / T_SL
    # apply Fuel-Flow-Method 2 correction
    Wf_SL = (
        (fuel_flow / n_eng)
        * (theta_amb**z)
        / delta_amb
        * np.exp(0.2 * (mach_number**2))
    )
    return Wf_SL

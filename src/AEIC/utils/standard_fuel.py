# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from AEIC.types import ThrustMode, ThrustModeArray

if TYPE_CHECKING:
    from AEIC.types import ModeValues


def get_thrust_cat_cruise(ff_eval: np.ndarray, ff_cal: ModeValues) -> ThrustModeArray:
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
    if ff_eval.size < 3:
        raise ValueError("For cruise, ff_eval must have at least 3 entries.")

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

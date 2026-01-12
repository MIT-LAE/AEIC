from enum import IntEnum

import numpy as np


class ICAOThrustMode(IntEnum):
    """ICAO thrust setting categories."""

    TAKEOFF_CLIMB = 1
    IDLE = 2
    APPROACH = 3


def get_thrust_cat_lto(ff_eval: np.ndarray) -> np.ndarray:
    """
    Classify each LTO fuel-flow value into a discrete thrust-setting category.

    Parameters
    ----------
    ff_eval : numpy.ndarray
        Fuel-flow values to be evaluated
        Shape ``(n_times,)``.
    ff_cal : numpy.ndarray
        Calibration fuel-flow points: exactly four LTO-mode points are expected.

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
    The four thrust categories are fixed to
    ``[2, 2, 3, 1]`` (Idle, Approach, Climb, Take-off) and simply returned.
    """
    # LTO case: assume exactly 4 calibration points (LTO modes)
    # Categories fixed: [IDLE, APPROACH, CLIMB, TAKEOFF]
    if ff_eval.size != 4:
        raise ValueError("For LTO, ff_eval must have length 4.")
    return np.array(
        [
            ICAOThrustMode.IDLE,
            ICAOThrustMode.APPROACH,
            ICAOThrustMode.TAKEOFF_CLIMB,
            ICAOThrustMode.TAKEOFF_CLIMB,
        ]
    )


def get_thrust_cat_cruise(ff_eval: np.ndarray, ff_cal: np.ndarray) -> np.ndarray:
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
    lowLimit = (ff_cal[0] + ff_cal[1]) / 2.0
    approachLimit = (ff_cal[1] + ff_cal[2]) / 2.0

    return np.select(
        [
            ff_eval <= lowLimit,
            ff_eval > approachLimit,
        ],
        [
            ICAOThrustMode.IDLE,
            ICAOThrustMode.TAKEOFF_CLIMB,
        ],
        default=ICAOThrustMode.APPROACH,
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

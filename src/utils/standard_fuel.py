import numpy as np
from utils.consts import kappa

def get_thrust_cat(ff_eval: np.ndarray,
                   ff_cal:  np.ndarray,
                   cruiseCalc: bool) -> np.ndarray:
    """
    Classify each fuel-flow value into a discrete thrust-setting category.

    Parameters
    ----------
    ff_eval : numpy.ndarray
        Fuel-flow values to be evaluated  
        Shape ``(n_times,)``.
    ff_cal : numpy.ndarray
        Calibration fuel-flow points  
        *If* ``cruiseCalc`` is ``True``: at least the first three cruise
        calibration points are required  
        (typically Idle, Approach, Climb/Take-off).  
        *If* ``cruiseCalc`` is ``False``: exactly four LTO-mode points
        are expected.
    cruiseCalc : bool
        • ``True``  →  Cruise calculation.  
        • ``False`` →  Only LTO calculation.

    Returns
    -------
    thrustCat : numpy.ndarray, dtype=int
        Integer category codes, same length as ``ff_eval``:  

        | Code | Meaning (ICAO convention) |
        |------|---------------------------|
        | 1    | Take-off / Climb          |
        | 2    | Idle                      |
        | 3    | Approach                  |

    Raises
    ------
    ValueError
        If the size/length constraints on ``ff_eval`` or ``ff_cal`` are violated.

    Notes
    -----
    **Cruise mode (`cruiseCalc=True`)**

    Two mid-points define the category boundaries  
    ``lowLimit  = (ff_cal[0] + ff_cal[1]) / 2``  
    ``approachLimit = (ff_cal[1] + ff_cal[2]) / 2``  

    * Idle          : ``ff_eval ≤ lowLimit``       →  code 2  
    * Take-off/Climb: ``ff_eval >  approachLimit`` →  code 1  
    * Approach      : remainder                    →  code 3  

    **LTO mode (`cruiseCalc=False`)**

    The four thrust categories are fixed to
    ``[2, 2, 3, 1]`` (Idle, Idle-2, Approach, Take-off) and simply returned.
    """
    n_times = ff_eval.shape[0]
    thrustCat = np.zeros(n_times, dtype=int)

    if cruiseCalc:
        if ff_eval.size < 3:
            raise ValueError("fuelflow_KGperS must have at least 3 entries when cruiseCalc=True.")
        # Define thresholds from the first three calibration points
        lowLimit = (ff_cal[0] + ff_cal[1]) / 2.0
        approachLimit = (ff_cal[1] + ff_cal[2]) / 2.0

        # Assign categories elementwise
        thrustCat[ff_eval <= lowLimit] = 2
        thrustCat[ff_eval > approachLimit] = 1
        # The remainder (where thrustCat == 0) are Approach
        thrustCat[thrustCat == 0] = 3

    else:
        # LTO case: assume exactly 4 calibration points (LTO modes)
        # Categories fixed: [2, 2, 3, 1]
        if ff_eval.size != 4:
            raise ValueError("When cruiseCalc=False, fuelflow_KGperS must have length 11.")
        base = np.array([2, 2, 3, 1], dtype=int)
        # We linearly interpolate each fuelfactor against the 11-point calibration?
        # But MATLAB simply tiles these 11 categories across each column. Since here we have
        # 1D fuelfactor, we assume it also has length 11 in the pure LTO scenario.
        if n_times != 4:
            raise ValueError("When cruiseCalc=False, fuelfactor must have length 4.")
        thrustCat = base.copy()
    return thrustCat


def get_SLS_equivalent_fuel_flow(fuel_flow:   np.ndarray,
                                 Pamb:        np.ndarray,
                                 mach_number: np.ndarray,
                                 z:     float = 3.8,
                                 P_SL:  float = 101_325.0):
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
    theta_amb = delta_amb ** ((kappa - 1.0) / kappa)
    # apply Fuel-Flow-Method 2 correction
    Wf_SL = fuel_flow * (theta_amb ** z) / delta_amb * np.exp(0.2 * mach_number ** 2)
    return Wf_SL
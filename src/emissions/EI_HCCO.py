import numpy as np


def EI_HCCO(
    fuelflow_evaluate: np.ndarray,
    x_EI_matrix: np.ndarray,
    fuelflow_calibrate: np.ndarray,
    Tamb: np.ndarray = np.empty(()),
    Pamb: np.ndarray = np.empty(()),
    cruiseCalc: bool = False,
) -> np.ndarray:
    """
    BFFM2 bilinear HC/CO fit to SLS data

    Parameters
    ----------
    fuelflow_evaluate : ndarray, shape (n_points,)
        Fuel flows [kg/s] at which to compute xEI. Must be 1D.
    x_EI_matrix : ndarray, shape (4,)
        Baseline emission indices [g x / kg fuel] at four calibration fuel‐flow points.
    fuelflow_calibrate : ndarray, shape (4,)
        Calibration fuel flows [kg/s] corresponding to x_EI_matrix
    cruiseCalc : bool
        If True, apply cruise correction (ambient T and P) to the final xEI.
    Tamb : ndarray, shape (n_points,)
        Ambient temperature [K] for cruise correction (if cruiseCalc is True).
    Pamb : ndarray, shape (n_points,)
        Ambient pressure [Pa] for cruise correction (if cruiseCalc is True).

    Returns
    -------
    xEI : ndarray, shape (n_points,)
        The HC+CO emission index [g x / kg fuel] at each fuelflow_evaluate.
    """

    # Validate inputs
    if x_EI_matrix.ndim != 1 or x_EI_matrix.size != 4:
        raise ValueError("x_EI_matrix must be a 1D array of length 4.")
    if fuelflow_calibrate.ndim != 1 or fuelflow_calibrate.size != 4:
        raise ValueError("fuelflow_calibrate must be a 1D array of length 4.")
    if fuelflow_evaluate.ndim != 1:
        raise ValueError("fuelflow_evaluate must be a 1D array (n_points,).")

    # Make copies to avoid modifying originals
    x_EI = x_EI_matrix.astype(float).copy()
    ff_cal = fuelflow_calibrate.astype(float).copy()
    ff_eval = fuelflow_evaluate.astype(float).copy()

    # ----------------------------------------------------------------------------
    # 1. Compute slanted‐line parameters in log10 space
    #    slope = [log10(xEI[1]) - log10(xEI[0])] / [log10(ff_cal[1]) - log10(ff_cal[0])]
    #    base_log_fuel = log10(ff_cal[0])
    #    base_log_EI   = log10(xEI[0])
    # ----------------------------------------------------------------------------
    # Prevent log10(0) by assuming calibration flows/EIs are strictly positive
    slope_num = np.log10(x_EI[1]) - np.log10(x_EI[0])
    slope_den = np.log10(ff_cal[1]) - np.log10(ff_cal[0])
    if np.isclose(slope_den, 0.0):
        slope = 0.0
    else:
        slope = slope_num / slope_den

    base_log_fuel = np.log10(ff_cal[0])
    base_log_EI = np.log10(x_EI[0])

    # ----------------------------------------------------------------------------
    # 2. Compute horizontal‐line level: midpoint of logs at calibration points 2 and 3
    #    x_horzline = 0.5 * [ log10(xEI[2]) + log10(xEI[3]) ]
    # ----------------------------------------------------------------------------
    x_horzline = 0.5 * (np.log10(x_EI[2]) + np.log10(x_EI[3]))

    # ----------------------------------------------------------------------------
    # 3. Compute intersection (in log10 fuel) between slanted and horizontal segments:
    #    x_intercept =
    #      [ 2*log10(ff_cal[0])*slope + log10(xEI[2]) + log10(xEI[3]) - 2*log10(xEI[0])]
    #                  / (2 * slope) , if slope != 0
    #    If slope == 0, force intercept := log10(ff_cal[1]) to use horizontal segment
    # ----------------------------------------------------------------------------
    if np.isclose(slope, 0.0):
        x_intercept = np.log10(ff_cal[1])
    else:
        numerator = (
            2.0 * np.log10(ff_cal[0]) * slope
            + np.log10(x_EI[2])
            + np.log10(x_EI[3])
            - 2.0 * np.log10(x_EI[0])
        )
        x_intercept = numerator / (2.0 * slope)

    # ----------------------------------------------------------------------------
    # 4. Enforce SAGE v1.5 rules row‐wise (here only one "row" since 1D):
    #    (a) If x_intercept > log10(ff_cal[2]), clamp it to log10(ff_cal[2]).
    #    (b) Else if x_intercept < log10(ff_cal[1]) and slope < 0:
    #        set x_horzline := log10(xEI[1]) and clamp x_intercept := log10(ff_cal[1]).
    #    (c) Else if slope >= 0: force slope=0, base_log_fuel=0, base_log_EI=x_horzline,
    #            and clamp x_intercept := log10(ff_cal[1]).
    # ----------------------------------------------------------------------------
    log_ff_cal1 = np.log10(ff_cal[1])
    log_ff_cal2 = np.log10(ff_cal[2])

    if x_intercept > log_ff_cal2:
        x_intercept = log_ff_cal2

    elif (x_intercept < log_ff_cal1) and (slope < 0.0):
        x_horzline = np.log10(x_EI[1])
        x_intercept = log_ff_cal1

    elif slope >= 0.0:
        slope = 0.0
        base_log_fuel = 0.0
        base_log_EI = x_horzline
        x_intercept = log_ff_cal1

    # ----------------------------------------------------------------------------
    # 5. Allocate output array and compute xEI for each evaluation point
    # ----------------------------------------------------------------------------
    n_points = ff_eval.size
    xEI_out = np.zeros(n_points, dtype=float)

    # Compute log10 of evaluation fuel flows, masking out non‐positive flows
    log_ff = np.zeros(n_points, dtype=float)
    positive_mask = ff_eval > 0.0
    log_ff[positive_mask] = np.log10(ff_eval[positive_mask])

    # Lower segment: log_ff < x_intercept
    lower_mask = positive_mask & (log_ff < x_intercept)
    # Upper segment: log_ff >= x_intercept
    upper_mask = log_ff >= x_intercept

    # Slanted‐line formula for "lower" points
    if np.any(lower_mask):
        xEI_out[lower_mask] = 10.0 ** (
            slope * (log_ff[lower_mask] - base_log_fuel) + base_log_EI
        )

    # Horizontal‐line (constant) for "upper" points
    if np.any(upper_mask):
        xEI_out[upper_mask] = 10.0**x_horzline

    # Replace any NaNs (e.g., from log10(0) → -inf) with zero
    xEI_out[np.isnan(xEI_out)] = 0.0

    # ----------------------------------------------------------------------------
    # 6. ACRP low‐thrust correction:
    #    For any fuelflow_evaluate < ff_cal[0], use:
    #       xEI_acrp = xEI * [1 + (–52) * (fuelflow_evaluate – ff_cal[0])]
    #    Then overwrite those points with xEI_acrp.
    # ----------------------------------------------------------------------------
    ACRP_slope = -52.0
    low_thrust_mask = ff_eval < ff_cal[0]
    if np.any(low_thrust_mask):
        delta_ff = ff_eval[low_thrust_mask] - ff_cal[0]
        xEI_acrp = xEI_out[low_thrust_mask] * (1.0 + ACRP_slope * delta_ff)
        xEI_out[low_thrust_mask] = xEI_acrp

    # ----------------------------------------------------------------------------
    # 7. Cruise correction (if requested):
    #    Multiply entire xEI_out by (θ^3.3)/(δ^1.02),
    #    where θ = Tamb / 288.15, δ = Pamb / 101325.
    # ----------------------------------------------------------------------------
    if cruiseCalc:
        theta_amb = Tamb / 288.15
        delta_amb = Pamb / 101325.0
        factor = (theta_amb**3.3) / (delta_amb**1.02)
        xEI_out *= factor

    return xEI_out

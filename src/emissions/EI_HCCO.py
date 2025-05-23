import numpy as np 

def EI_HCCO(
    fuelfactor: np.ndarray,
    EI_HCCO_matrix: np.ndarray,
    fuelflow_KGperS: np.ndarray,
    cruiseCalc: bool = False,
    Tamb: float = 288.15,
    Pamb: float = 101325.0
    )-> np.ndarray:
    """
    Calculate EI(x) for each EQP using BFFM2 and given fuel flows.

    Parameters
    ----------
    fuelfactor : ndarray, shape (n_types, n_times)
        Fuel flow factor per type and time.
    EI_HCCO_matrix : ndarray, shape (n_types, 4)
        Baseline emissions index values at reference fuel flows.
    fuelflow_KGperS : ndarray, shape (n_types, 4)
        Reference fuel flow rates corresponding to EI_HCCO_matrix columns.
    cruiseCalc : bool
        Whether to apply cruise-phase corrections.
    Tamb : float
        Ambient temperature [K].
    Pamb : float
        Ambient pressure [Pa].

    Returns
    -------
    xEI : ndarray, shape (n_types, n_times)
        Emissions index per type and time.
    """
    n_types, _ = EI_HCCO_matrix.shape

    # Compute slanted line parameters
    log_xEI = np.log10(EI_HCCO_matrix)
    log_ff = np.log10(fuelflow_KGperS)

    slopes = (log_xEI[:, 1] - log_xEI[:, 0]) / (log_ff[:, 1] - log_ff[:, 0])
    x_slantline = np.column_stack((slopes, log_ff[:, 0], log_xEI[:, 0]))

    # Horizontal line at midpoint of higher-power EIs
    x_horzline = 0.5 * (log_xEI[:, 2] + log_xEI[:, 3])

    # Compute intercepts of slanted and horizontal lines
    x_intercept = (
        2 * log_ff[:, 0] * slopes
        + log_xEI[:, 2] + log_xEI[:, 3]
        - 2 * log_xEI[:, 0]
    ) / (2 * slopes)

    # Adjust intercepts and slanted parameters per type
    for i in range(n_types):
        if x_intercept[i] > log_ff[i, 2]:
            x_intercept[i] = log_ff[i, 2]
        elif x_intercept[i] < log_ff[i, 1] and slopes[i] < 0:
            x_horzline[i] = log_xEI[i, 1]
            x_intercept[i] = log_ff[i, 1]
        elif slopes[i] >= 0:
            slopes[i] = 0
            x_slantline[i, 0] = 0
            x_slantline[i, 1] = 0
            x_slantline[i, 2] = x_horzline[i]
            x_intercept[i] = log_ff[i, 1]

    # Allocate output
    n_times = fuelfactor.shape[1]
    xEI = np.zeros_like(fuelfactor)

    # Compute EI by fuel factor thresholds
    for i in range(n_types):
        ff_i = fuelfactor[i, :]
        log_ff_i = np.log10(np.where(ff_i > 0, ff_i, 1e-12))
        mask_low = (ff_i > 0) & (log_ff_i < x_intercept[i])
        mask_high = log_ff_i >= x_intercept[i]

        # Slanted line region
        xEI[i, mask_low] = 10 ** (
            slopes[i] * (log_ff_i[mask_low] - x_slantline[i, 1])
            + x_slantline[i, 2]
        )
        # Horizontal region
        xEI[i, mask_high] = 10 ** (x_horzline[i])

    # Replace NaNs with zero
    xEI = np.nan_to_num(xEI)

    # ACRP low-thrust correction
    ACRP_slope = -52
    threshold = fuelflow_KGperS[:, 0][:, None]
    mask_acrp = fuelfactor < threshold
    ACRP_EI = xEI * (1 + ACRP_slope * (fuelfactor - threshold))
    xEI = np.where(mask_acrp, ACRP_EI, xEI)

    # Cruise-phase ambient correction
    if cruiseCalc:
        theta_amb = Tamb / 288.15
        delta_amb = Pamb / 101325.0
        factor = (theta_amb ** 3.3) / (delta_amb ** 1.02)
        xEI = xEI * factor

    return xEI
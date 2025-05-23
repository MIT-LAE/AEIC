import numpy as np

def EI_PMnvol(
    thrusts: np.ndarray,
    PMnvolEI_ICAOthrust: np.ndarray
) -> np.ndarray:
    """
    Calculate EI(PMnvol) (black carbon) [g/kg fuel] via linear interpolation
    from reference thrust-specific values.

    Parameters
    ----------
    thrusts : ndarray, shape (n_types, n_times)
        ICAO thrust settings (%) for each mode and time.
    PMnvolEI_ICAOthrust : ndarray, shape (n_types, 5)
        Emissions indices at thrusts [0, 7, 30, 85, 100] (%).
    mcs7, mcs30, mcs85 : ints
        Monte Carlo switches (unused, commented out in original).
    rv7, rv30, rv85 : floats
        Random variates (unused, commented out in original).

    Returns
    -------
    PMnvolEI : ndarray, shape (n_types, n_times)
        Interpolated black carbon emissions index [g/kg fuel].
    """
    # Define reference thrust levels
    ICAO_thrust = np.array([0, 7, 30, 85, 100], dtype=float)

    # Allocate output
    n_types, n_times = thrusts.shape
    PMnvolEI = np.zeros((n_types, n_times), dtype=float)

    # Perform interpolation row-by-row
    for i in range(n_types):
        PMnvolEI[i, :] = np.interp(thrusts[i, :], ICAO_thrust, PMnvolEI_ICAOthrust[i, :])

    return PMnvolEI

def EI_PMnvolN(thrusts: np.ndarray,
                      PMnvolEIN_ICAOthrust: np.ndarray) -> np.ndarray:
    """
    Interpolate non-volatile particulate matter number-based EI (PMnvolEI_N)
    across ICAO thrust settings.

    Parameters
    ----------
    thrusts : ndarray, shape (n_types, n_times)
        ICAO thrust settings (%) for each mode and time.
    PMnvolEIN_ICAOthrust : ndarray, shape (n_types, 5)
        Reference EI values at thrusts [0, 7, 30, 85, 100] (%).

    Returns
    -------
    PMnvolEI_N : ndarray, shape (n_types, n_times)
        Interpolated non-volatile PM EI [g/kg fuel].
    """
    # Define reference thrust levels
    ICAO_thrust = np.array([0, 7, 30, 85, 100], dtype=float)

    # Allocate output array
    n_types, n_times = thrusts.shape
    PMnvolEI_N = np.zeros((n_types, n_times), dtype=float)

    # Perform interpolation for each type (row)
    for i in range(n_types):
        PMnvolEI_N[i, :] = np.interp(thrusts[i, :],
                                     ICAO_thrust,
                                     PMnvolEIN_ICAOthrust[i, :])

    return PMnvolEI_N
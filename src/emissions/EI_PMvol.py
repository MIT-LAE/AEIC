import numpy as np

def EI_PMvol_NEW(fuelflow: np.ndarray, thrustCat: np.ndarray):
    """
    Calculate EI(PMvolo) and OCicEI based on fuel flow 

    Parameters
    ----------
    fuelflow : ndarray, shape (n_types, 4)
        Fuel flow factor per type and 4 thrust modes.

    Returns
    -------
    PMvoloEI : ndarray, shape (n_types, 4)
        Emissions index for volatile organic PM [g/kg fuel].
    OCicEI : ndarray, shape (n_types, 4)
        Emissions index for organic carbon internal [g/kg fuel].
    """

    # 1) Fixed OC_ic = 20 mg/kg → 0.02 g/kg
    OCic_val = 20.0e-3  # g/kg

    # 2) Deterministic lube-oil contributions (midpoints of low/high ranges)
    lubeContrL = 0.15   # midpoint of 10–20%
    lubeContrH = 0.50   # midpoint of 40–60%
    lubeContr = np.where(thrustCat == 'L', lubeContrL, lubeContrH)

    # 3) Compute PMvoloEI = OCicEI / (1 − lubeContr)
    PMvolo_vec = OCic_val / (1.0 - lubeContr)

    # 4) Tile to match fuelflow shape
    OCicEI   = np.full_like(fuelflow, OCic_val, dtype=float)
    PMvoloEI = PMvolo_vec.copy()

    return PMvoloEI, OCicEI


def EI_PMvol_FOA3(thrusts: np.ndarray, HCEI: np.ndarray):
    """
    Calculate volatile organic PM emissions index (PMvoloEI) and OC internal EI (OCicEI)
    using the FOA3.0 method (Wayson et al., 2009).

    Parameters
    ----------
    thrusts : ndarray, shape (n_types, n_times)
        ICAO thrust settings (%) for each mode and time.
    HCEI : ndarray, shape (n_types, n_times)
        Hydrocarbon emissions index [g/kg fuel] for each mode and time.

    Returns
    -------
    PMvoloEI : ndarray, shape (n_types, n_times)
        Emissions index for volatile organic PM [g/kg fuel].
    OCicEI : ndarray, shape (n_types, n_times)
        Same as PMvoloEI (internal organic carbon component).
    """
    # FOA3 delta values (mg organic carbon per g fuel)
    ICAO_thrust = np.array([7, 30, 85, 100], dtype=float)
    delta = np.array([6.17, 56.25, 76.0, 115.0], dtype=float)

    # Interpolate delta for each thrust value
    delta_matrix = np.interp(thrusts, ICAO_thrust, delta)

    # PMvoloEI: delta [mg/g] * HCEI [g/kg] / 1000 -> g/kg
    PMvoloEI = delta_matrix * HCEI / 1000.0

    # OC internal EI equals PMvoloEI
    OCicEI = PMvoloEI.copy()

    return PMvoloEI, OCicEI
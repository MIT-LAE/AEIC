import numpy as np

def EI_PMvol_NEW(fuelflow: np.ndarray):
    """
    Calculate EI(PMvolo) and OCicEI based on fuel flow and Monte Carlo switches.

    Parameters
    ----------
    fuelflow : ndarray, shape (n_types, 11)
        Fuel flow factor per type and 11 thrust modes.

    Returns
    -------
    PMvoloEI : ndarray, shape (n_types, 11)
        Emissions index for volatile organic PM [g/kg fuel].
    OCicEI : ndarray, shape (n_types, 11)
        Emissions index for organic carbon internal [g/kg fuel].
    """
    # Determine nominal OC internal EI (g/kg)
    # if mcsEI == 1:
    #     OCic_scalar = trirnd(1, 40, 20, rvEI) / 1000.0
    # else:
    OCic_scalar = 20.0 / 1000.0

    # Lube oil contribution fractions
    # if mcsLube == 1:
    #     lubeContrL = 0.1 + rvLube * (0.2 - 0.1)
    #     lubeContrH = 0.4 + rvLube * (0.6 - 0.4)
    # else:
    lubeContrL = 0.15
    lubeContrH = 0.5

    # Thrust category per of 11 modes
    thrustCat = np.array(['L','L','L','H','H','H','L','L','L','L','L'])

    # Compute lube contribution per mode
    lubeContr = np.where(thrustCat == 'L', lubeContrL, lubeContrH)

    # Compute PMvolo EI per mode (g/kg)
    PMvolo_per_mode = OCic_scalar / (1.0 - lubeContr)

    # Replicate across types
    n_types = fuelflow.shape[0]
    PMvoloEI = np.tile(PMvolo_per_mode, (n_types, 1))

    # Replicate OCicEI across fuelflow shape
    OCicEI = np.full_like(fuelflow, OCic_scalar)

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
    ICAO_thrust = np.array([0, 7, 30, 85, 100], dtype=float)
    delta = np.array([6.17, 6.17, 56.25, 76.0, 115.0], dtype=float)

    # Interpolate delta for each thrust value
    delta_matrix = np.interp(thrusts, ICAO_thrust, delta)

    # PMvoloEI: delta [mg/g] * HCEI [g/kg] / 1000 -> g/kg
    PMvoloEI = delta_matrix * HCEI / 1000.0

    # OC internal EI equals PMvoloEI
    OCicEI = PMvoloEI.copy()

    return PMvoloEI, OCicEI
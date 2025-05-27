import numpy as np

def EI_SOx(fuelflow: np.ndarray, fuel: dict):
    """
    Calculate universal SOx emissions indices (SO2EI and SO4EI).

    Parameters
    ----------
    fuelflow : ndarray
        Fuel flow array (any shape), units kg of fuel.

    fuel : dictionary
        Fuel information (input from toml file)

    Returns
    -------
    SO2EI : ndarray
        SO2 emissions index [g SO2 per kg fuel], same shape as fuelflow.
    SO4EI : ndarray
        SO4 emissions index [g SO4 per kg fuel], same shape as fuelflow.
    """
    # Nominal values
    FSCnom = fuel['FSCnom']
    Epsnom = fuel['Epsnom']

    # Apply MC for FSC
    # if mcsFSC == 1:
    #     FSC = trirnd(500, 700, FSCnom, rvFSC)
    # else:
    FSC = FSCnom

    # Apply MC for Eps
    # if mcsEps == 1:
    #     Eps = trirnd(0.005, 0.05, Epsnom, rvEps)
    # else:
    Eps = Epsnom

    # Molecular weights
    MW_SO2 = 64.0
    MW_SO4 = 96.0
    MW_S = 32.0

    # Compute emissions indices (g/kg fuel)
    so4_val = 1e3 * ((FSC / 1e6) * Eps * MW_SO4) / MW_S
    so2_val = 1e3 * ((FSC / 1e6) * (1 - Eps) * MW_SO2) / MW_S

    # Broadcast to fuelflow shape
    SO4EI = np.full_like(fuelflow, so4_val, dtype=float)
    SO2EI = np.full_like(fuelflow, so2_val, dtype=float)

    return SO2EI, SO4EI
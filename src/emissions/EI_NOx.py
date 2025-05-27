import numpy as np 

def EI_NOx(
        fuel_flow: np.ndarray,
        NOX_EI_matrix: np.ndarray,
        fuelfactor: np.ndarray,
        P3_kPa:np.ndarray,
        T3_K:np.ndarray,
        cruiseCalc: bool = False,
        Tamb: float = 288.15,
        Pamb: float = 101325.0,
        mode:str = "P3T3",
        sp_humidity: float = 0.00634
    ):
    """
    Calculates NOx emissions indices and speciation.

    Parameters
    ----------
    fuelfactor : ndarray, shape (n_types, n_times)
    NOX_EI_matrix : ndarray, same shape as fuel_flow
    fuel_flow : ndarray, shape (n_types, n_times)
    cruiseCalc : bool, whether to apply cruise corrections
    Tamb : float, ambient temperature [K]
    Pamb : float, ambient pressure [Pa]

    Returns
    -------
    NOxEI, NOEI, NO2EI, HONOEI : ndarrays same shape as fuel_flow
    noProp, no2Prop, honoProp : ndarrays same shape as fuel_flow
    """

    if mode == "P3T3":
        a,b,c,d,e,f,g,h,i,j = [ 8.46329738,  0.00980137, -8.55054025,  0.00981223,  0.02928154,
            0.01037376,  0.03666156,  0.01037419,  0.03664096,  0.01037464]

        H = -19.0*(sp_humidity - 0.00634)
        
        NOxEI = np.exp(H)*(P3_kPa**0.4) * (a * np.exp(b * T3_K) + c * np.exp(d * T3_K) + e * np.exp(f * T3_K) + g * np.exp(h * T3_K) + i * np.exp(j * T3_K))
    elif mode == "BFFM2":
        # Fit log-log linear models per row
        slopes = np.zeros(fuel_flow.shape[0])
        intercepts = np.zeros_like(slopes)
        for i in range(fuel_flow.shape[0]):
            x = np.log10(fuel_flow[i, :])
            y = np.log10(NOX_EI_matrix[i, :])
            slopes[i], intercepts[i] = np.polyfit(x, y, 1)

        # Interpolate to compute NOxEI
        logf = np.log10(fuelfactor)
        NOxEI = 10 ** (logf * slopes[:, None] + intercepts[:, None])

        # Cruise-phase ambient corrections
        if cruiseCalc:
            theta_amb = Tamb / 288.15
            delta_amb = Pamb / 101325.0
            Pamb_psia = delta_amb * 14.696
            beta = (
                7.90298 * (1 - 373.16 / (Tamb + 0.01))
                + 3.00571
                + 5.02808 * np.log10(373.16 / (Tamb + 0.01))
                + 1.3816e-7 * (1 - 10 ** (11.344 * (1 - (Tamb + 0.01) / 373.16)))
                + 8.1328e-3 * (10 ** (3.49149 * (1 - 373.16 / (Tamb + 0.01))) - 1)
            )
            Pv = 0.014504 * 10 ** beta
            phi = 0.6
            omega = (0.62198 * phi * Pv) / (Pamb_psia - phi * Pv)
            H = -19.0 * (omega - 0.0063)
            NOxEI = NOxEI * np.exp(H) * (delta_amb ** 1.02 / theta_amb ** 3.3) ** 0.5
    else:
        raise Exception("Invalid mode input in EI NOx function (BFFM2, P3T3)")

    # Thrust category assignment
    if cruiseCalc:
        lowLimit = (fuel_flow[0, :] + fuel_flow[1, :]) / 2
        approachLimit = (fuel_flow[1, :] + fuel_flow[2, :]) / 2
        thrustCat = np.where(
            fuelfactor <= lowLimit, 2,
            np.where(fuelfactor > approachLimit, 1, 3)
        )
    else:
        thrustCat = np.array([2,2,2,1,1,1,3,2,3,2,2])  # for 11 modes

    # Speciation bounds
    hono_bounds = {
        'H': (0.5, 1.0, 0.75),
        'L': (2.0, 7.0, 4.5),
        'A': (2.0, 7.0, 4.5)
    }
    no2_bounds = {
        'H': (5.0, 10.0 * (100-1)/100, 7.5 * (100-0.75)/100),
        'L': (75.0, 98.0 * (100-4.5)/100, 86.5 * (100-4.5)/100),
        'A': (12.0, 20.0 * (100-4.5)/100, 16.0 * (100-4.5)/100)
    }
    # TODO: check if Monte Carlo for speciation needed
    # if mcsHONO == 1:
    #     honoH = trirnd(*hono_bounds['H'], rvHONO)
    #     honoL = trirnd(*hono_bounds['L'], rvHONO)
    #     honoA = trirnd(*hono_bounds['A'], rvHONO)
    # else:
    honoH_nom = hono_bounds['H'][2]
    honoL_nom = hono_bounds['L'][2]
    honoA_nom = hono_bounds['A'][2]
    honoH, honoL, honoA = honoH_nom, honoL_nom, honoA_nom

    # if mcsNO2 == 1:
    #     no2H = trirnd(*no2_bounds['H'], rvNO2)
    #     no2L = trirnd(*no2_bounds['L'], rvNO2)
    #     no2A = trirnd(*no2_bounds['A'], rvNO2)
    # else:
    no2H, no2L, no2A = no2_bounds['H'][2], no2_bounds['L'][2], no2_bounds['A'][2]

    noH = 100 - honoH - no2H
    noL = 100 - honoL - no2L
    noA = 100 - honoA - no2A

    # Proportion arrays
    honoProp = np.where(thrustCat == 1, honoH,
                 np.where(thrustCat == 2, honoL, honoA)) / 100.0
    no2Prop  = np.where(thrustCat == 1, no2H,
                 np.where(thrustCat == 2, no2L, no2A)) / 100.0
    noProp   = np.where(thrustCat == 1, noH,
                 np.where(thrustCat == 2, noL, noA)) / 100.0

    # Compute speciation EIs
    NOEI   = NOxEI * noProp[:, None]
    NO2EI  = NOxEI * no2Prop[:, None]
    HONOEI = NOxEI * honoProp[:, None]

    return NOxEI, NOEI, NO2EI, HONOEI, noProp, no2Prop, honoProp
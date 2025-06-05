import numpy as np 
import warnings

def EI_NOx(
        fuel_flow_input: np.ndarray,
        NOX_EI_input: np.ndarray,
        fuel_flow_output: np.ndarray,
        P3_kPa = None,
        T3_K = None,
        cruiseCalc: bool = True,
        Tamb: float = 288.15,
        Pamb: float = 101325.0,
        mode:str = "BFFM2",
        sp_humidity: float = 0.00634
    ):
    """
    Calculates NOx emissions indices and speciation.

    Parameters
    ----------
    fuelfactor : ndarray, shape (n_types, n_times)
    NOX_EI : ndarray, same shape as fuel_flow
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
        return BFFM2_EINOx(fuel_flow_input,NOX_EI_input,fuel_flow_output,cruiseCalc,Tamb,Pamb)
    else:
        raise Exception("Invalid mode input in EI NOx function (LOG, P3T3)")

    # Thrust category assignment
    if cruiseCalc:
        lowLimit = (fuel_flow_input[0, :] + fuel_flow_input[1, :]) / 2
        approachLimit = (fuel_flow_input[1, :] + fuel_flow_input[2, :]) / 2
        thrustCat = np.where(
            fuel_flow_output <= lowLimit, 2,
            np.where(fuel_flow_output > approachLimit, 1, 3)
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



import numpy as np


def BFFM2_EINOx(
    fuelfactor: np.ndarray,
    NOX_EI_matrix: np.ndarray,
    fuelflow_KGperS: np.ndarray,
    cruiseCalc: bool = True,
    Tamb: float = 288.15,
    Pamb: float = 101325.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate NOx, NO, NO2, and HONO emission indices 
    All inputs are 1-dimensional arrays of equal length for calibration (fuelflow_KGperS vs. NOX_EI_matrix)
    and 1-dimensional for fuelfactor (multiple evaluation points).

    Parameters
    ----------
    fuelfactor : ndarray, shape (n_times,)
        Fuel flow factors (including delta, theta, and Mach corrections) at which to compute EI.
    NOX_EI_matrix : ndarray, shape (n_cal,)
        Baseline NOx EI values [g NOx / kg fuel] corresponding to calibration fuel flows.
    fuelflow_KGperS : ndarray, shape (n_cal,)
        Calibration fuel flow values [kg/s] for which NOX_EI_matrix is defined.
        Must have at least three entries if cruiseCalc=True.
    cruiseCalc : bool
        If True, apply cruise ambient corrections (temperature and pressure) to NOx EI.
    Tamb : float
        Ambient temperature [K].
    Pamb : float
        Ambient pressure [Pa].

    Returns
    -------
    NOxEI   : ndarray, shape (n_times,)
        Interpolated NOx emission index [g NOx / kg fuel] for each fuelfactor.
    NOEI    : ndarray, shape (n_times,)
        NO emission index [g NO / kg fuel].
    NO2EI   : ndarray, shape (n_times,)
        NO2 emission index [g NO2 / kg fuel].
    HONOEI  : ndarray, shape (n_times,)
        HONO emission index [g HONO / kg fuel].
    noProp  : ndarray, shape (n_times,)
        Fraction of NO within total NOy (unitless).
    no2Prop : ndarray, shape (n_times,)
        Fraction of NO2 within total NOy (unitless).
    honoProp: ndarray, shape (n_times,)
        Fraction of HONO within total NOy (unitless).
    """

    # ----------------------------------------------------------------------------
    # 1. Fit a linear model in log10-space: log10(NOx_EI) vs. log10(fuelflow)
    # ----------------------------------------------------------------------------
    # Ensure no non-positive calibration fuel flows
    ff_cal = fuelflow_KGperS.copy()
    ff_cal[ff_cal <= 0] = 0.01

    # Take logs
    x_log = np.log10(ff_cal)
    y_log = np.log10(NOX_EI_matrix)

    # Perform linear regression (slope, intercept)
    slope, intercept = np.polyfit(x_log, y_log, 1)

    # ----------------------------------------------------------------------------
    # 2. Interpolate NOxEI for each fuelfactor point
    # ----------------------------------------------------------------------------
    # If any fuelfactor <= 0, set to small positive to avoid log10 issues
    ff_eval = fuelfactor.copy()
    ff_eval[ff_eval <= 0] = 0.01

    NOxEI = 10.0 ** (np.log10(ff_eval) * slope + intercept)

    # ----------------------------------------------------------------------------
    # 3. Apply cruise ambient corrections if requested
    # ----------------------------------------------------------------------------
    if cruiseCalc:
        # θ_amb = Tamb / 288.15
        theta_amb = Tamb / 288.15
        # δ_amb = Pamb / 101325
        delta_amb = Pamb / 101325.0
        # Pamb in psia
        Pamb_psia = delta_amb * 14.696

        # Compute saturation vapor pressure term β (per BFFM2)
        beta = (
            7.90298 * (1.0 - (373.16) / (Tamb + 0.01))
            + 3.00571
            + 5.02808 * np.log10((373.16) / (Tamb + 0.01))
            + 1.3816e-7 * (1.0 - (10.0 ** (11.344 * (1.0 - ((Tamb + 0.01) / 373.16)))))
            + 8.1328e-3 * ((10.0 ** (3.49149 * (1.0 - (373.16) / (Tamb + 0.01)))) - 1.0)
        )
        Pv = 0.014504 * (10.0 ** beta)  # [psia]
        phi = 0.6  # 60% relative humidity
        omega = (0.62198 * phi * Pv) / (Pamb_psia - (phi * Pv))
        H = -19.0 * (omega - 0.0063)

        correction = np.exp(H) * ((delta_amb ** 1.02) / (theta_amb ** 3.3)) ** 0.5
        NOxEI = NOxEI * correction

    # ----------------------------------------------------------------------------
    # 4. Determine thrust category for each fuelfactor point
    #    Categories: 1=High (H), 2=Low (L), 3=Approach (A)
    # ----------------------------------------------------------------------------
    n_times = ff_eval.shape[0]
    thrustCat = np.zeros(n_times, dtype=int)

    if cruiseCalc:
        if ff_cal.size < 3:
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
        # LTO case: assume exactly 11 calibration points (LTO modes)
        # Categories fixed: [2,2,2,1,1,1,3,2,3,2,2]
        if ff_cal.size != 11:
            raise ValueError("When cruiseCalc=False, fuelflow_KGperS must have length 11.")
        base = np.array([2, 2, 2, 1, 1, 1, 3, 2, 3, 2, 2], dtype=int)
        # We linearly interpolate each fuelfactor against the 11-point calibration?
        # But MATLAB simply tiles these 11 categories across each column. Since here we have
        # 1D fuelfactor, we assume it also has length 11 in the pure LTO scenario.
        if n_times != 11:
            raise ValueError("When cruiseCalc=False, fuelfactor must have length 11.")
        thrustCat = base.copy()

    # ----------------------------------------------------------------------------
    # 5. Speciation (no Monte Carlo; use nominal percentages)
    # ----------------------------------------------------------------------------
    # HONO nominal (% of NOy)
    honoHnom, honoLnom, honoAnom = 0.75, 4.5, 4.5
    # NO2 nominal computed from (NO2/(NOy - HONO)) * (100 - HONO)
    no2Hnom = 7.5 * (100.0 - honoHnom) / 100.0
    no2Lnom = 86.5 * (100.0 - honoLnom) / 100.0
    no2Anom = 16.0 * (100.0 - honoAnom) / 100.0

    # NO nominal so that NO + NO2 + HONO = 100%
    noHnom = 100.0 - honoHnom - no2Hnom
    noLnom = 100.0 - honoLnom - no2Lnom
    noAnom = 100.0 - honoAnom - no2Anom

    # Allocate arrays for proportions
    honoProp = np.zeros(n_times)
    no2Prop = np.zeros(n_times)
    noProp = np.zeros(n_times)

    # Fill each based on thrust category
    # Category 1 => High, 2 => Low, 3 => Approach
    honoProp[thrustCat == 1] = honoHnom / 100.0
    honoProp[thrustCat == 2] = honoLnom / 100.0
    honoProp[thrustCat == 3] = honoAnom / 100.0

    no2Prop[thrustCat == 1] = no2Hnom / 100.0
    no2Prop[thrustCat == 2] = no2Lnom / 100.0
    no2Prop[thrustCat == 3] = no2Anom / 100.0

    noProp[thrustCat == 1] = noHnom / 100.0
    noProp[thrustCat == 2] = noLnom / 100.0
    noProp[thrustCat == 3] = noAnom / 100.0

    # ----------------------------------------------------------------------------
    # 6. Compute component EIs
    # ----------------------------------------------------------------------------
    if np.isnan(NOxEI).any():
        warnings.warn("NaN encountered in NOxEI calculation.", RuntimeWarning)

    NOEI = NOxEI * noProp     # g NO / kg fuel
    NO2EI = NOxEI * no2Prop   # g NO2 / kg fuel
    HONOEI = NOxEI * honoProp # g HONO / kg fuel

    return NOxEI, NOEI, NO2EI, HONOEI, noProp, no2Prop, honoProp

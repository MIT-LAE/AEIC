import numpy as np 
from utils.standard_fuel import get_fuel_factor, get_thrust_cat
import warnings

# TODO: add P3T3 method support
# def EI_NOx(
#         fuel_flow_trajectory: np.ndarray,
#         NOX_EI_input: np.ndarray,
#         fuel_flow_performance: np.ndarray,
#         Tamb: np.ndarray,
#         Pamb: np.ndarray,
#         mach_number: np.ndarray,
#         P3_kPa = None,
#         T3_K = None,
#         cruiseCalc: bool = True,
#         mode:str = "BFFM2",
#         sp_humidity: float = 0.00634
#     ):
#     """
#     Calculates NOx emissions indices and speciation.

#     Parameters
#     ----------
#     fuelfactor : ndarray, shape (n_types, n_times)
#     NOX_EI : ndarray, same shape as fuel_flow
#     fuel_flow : ndarray, shape (n_types, n_times)
#     cruiseCalc : bool, whether to apply cruise corrections
#     Tamb : float, ambient temperature [K]
#     Pamb : float, ambient pressure [Pa]

#     Returns
#     -------
#     NOxEI, NOEI, NO2EI, HONOEI : ndarrays same shape as fuel_flow
#     noProp, no2Prop, honoProp : ndarrays same shape as fuel_flow
#     """

#     if mode == "P3T3":
#         a,b,c,d,e,f,g,h,i,j = [ 8.46329738,  0.00980137, -8.55054025,  0.00981223,  0.02928154,
#             0.01037376,  0.03666156,  0.01037419,  0.03664096,  0.01037464]

#         H = -19.0*(sp_humidity - 0.00634)
        
#         NOxEI = np.exp(H)*(P3_kPa**0.4) * (a * np.exp(b * T3_K) + c * np.exp(d * T3_K) + e * np.exp(f * T3_K) + g * np.exp(h * T3_K) + i * np.exp(j * T3_K))
#     elif mode == "BFFM2":
#         return BFFM2_EINOx(fuel_flow_trajectory,NOX_EI_input,fuel_flow_performance,Tamb,Pamb,mach_number)
#     else:
#         raise Exception("Invalid mode input in EI NOx function (BFFM2, P3T3)")

#     # Thrust category assignment
#     thrustCat = get_thrust_cat(fuel_flow_trajectory, fuel_flow_performance, cruiseCalc)

#     # Speciation bounds
#     hono_bounds = {
#         'H': (0.5, 1.0, 0.75),
#         'L': (2.0, 7.0, 4.5),
#         'A': (2.0, 7.0, 4.5)
#     }
#     no2_bounds = {
#         'H': (5.0, 10.0 * (100-1)/100, 7.5 * (100-0.75)/100),
#         'L': (75.0, 98.0 * (100-4.5)/100, 86.5 * (100-4.5)/100),
#         'A': (12.0, 20.0 * (100-4.5)/100, 16.0 * (100-4.5)/100)
#     }
#     # TODO: check if Monte Carlo for speciation needed
#     # if mcsHONO == 1:
#     #     honoH = trirnd(*hono_bounds['H'], rvHONO)
#     #     honoL = trirnd(*hono_bounds['L'], rvHONO)
#     #     honoA = trirnd(*hono_bounds['A'], rvHONO)
#     # else:
#     honoH_nom = hono_bounds['H'][2]
#     honoL_nom = hono_bounds['L'][2]
#     honoA_nom = hono_bounds['A'][2]
#     honoH, honoL, honoA = honoH_nom, honoL_nom, honoA_nom

#     # if mcsNO2 == 1:
#     #     no2H = trirnd(*no2_bounds['H'], rvNO2)
#     #     no2L = trirnd(*no2_bounds['L'], rvNO2)
#     #     no2A = trirnd(*no2_bounds['A'], rvNO2)
#     # else:
#     no2H, no2L, no2A = no2_bounds['H'][2], no2_bounds['L'][2], no2_bounds['A'][2]

#     noH = 100 - honoH - no2H
#     noL = 100 - honoL - no2L
#     noA = 100 - honoA - no2A

#     # Proportion arrays
#     honoProp = np.where(thrustCat == 1, honoH,
#                  np.where(thrustCat == 2, honoL, honoA)) / 100.0
#     no2Prop  = np.where(thrustCat == 1, no2H,
#                  np.where(thrustCat == 2, no2L, no2A)) / 100.0
#     noProp   = np.where(thrustCat == 1, noH,
#                  np.where(thrustCat == 2, noL, noA)) / 100.0

#     # Compute speciation EIs
#     NOEI   = NOxEI * noProp[:, None]
#     NO2EI  = NOxEI * no2Prop[:, None]
#     HONOEI = NOxEI * honoProp[:, None]

#     return NOxEI, NOEI, NO2EI, HONOEI, noProp, no2Prop, honoProp

def BFFM2_EINOx(
    fuelflow_trajectory: np.ndarray,
    NOX_EI_matrix: np.ndarray,
    fuelflow_performance: np.ndarray,
    Tamb: np.ndarray,
    Pamb: np.ndarray,
    cruiseCalc: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate NOx, NO, NO2, and HONO emission indices 
    All inputs are 1-dimensional arrays of equal length for calibration (fuelflow_KGperS vs. NOX_EI_matrix)
    and 1-dimensional for fuelfactor (multiple evaluation points).

    Parameters
    ----------
    fuelflow_trajectory : ndarray, shape (n_times,)
        Fuel flow at which to compute EI.
    NOX_EI_matrix : ndarray, shape (n_cal,)
        Baseline NOx EI values [g NOx / kg fuel] corresponding to calibration fuel flows.
    fuelflow_performance : ndarray, shape (n_cal,)
        Calibration fuel flow values [kg/s] for which NOX_EI_matrix is defined.
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
    fuelfactor = fuelflow_trajectory#get_fuel_factor(fuelflow_trajectory, Pamb, mach_number)

    # (Proceed with steps 1, 2, 3 exactly as in the paper)
    # 1) Fit log10(NOX_EI_matrix) vs. log10(fuelflow_KGperS)
    ff_cal = fuelflow_performance.copy()
    ff_cal[ff_cal <= 0.0] = 1e-2
    x_log = np.log10(ff_cal)
    y_log = np.log10(NOX_EI_matrix)
    slope, intercept = np.polyfit(x_log, y_log, 1)

    # 2) Interpolate NOxEI at log10(fuelfactor)
    ff_eval = fuelfactor.copy()
    ff_eval[ff_eval <= 0.0] = 1e-2
    NOxEI_sl = 10.0 ** (slope * np.log10(ff_eval) + intercept)

    # 3) If cruiseCalc=True, apply the humidity/θ/δ correction (Eqs. 44–45)
    if cruiseCalc:
        theta_amb = Tamb / 288.15
        delta_amb = Pamb / 101325.0
        Pamb_psia = delta_amb * 14.696

        # Compute β (saturation vapor – Eq. 44)
        beta = (
            7.90298 * (1.0 - 373.16 / (Tamb + 0.01))
            + 3.00571
            + 5.02808 * np.log10(373.16 / (Tamb + 0.01))
            + 1.3816e-7 * (1.0 - (10.0 ** (11.344 * (1.0 - ((Tamb + 0.01) / 373.16)))))
            + 8.1328e-3 * ((10.0 ** (3.49149 * (1.0 - (373.16 / (Tamb + 0.01))))) - 1.0)
        )
        Pv = 0.014504 * (10.0 ** beta)    # [psia]
        phi = 0.6                        # 60% relative humidity
        omega = (0.62198 * phi * Pv) / (Pamb_psia - (phi * Pv))
        H = -19.0 * (omega - 0.0063)

        # Eq. (45) ambient correction:
        correction = np.exp(H) * ((delta_amb ** 1.02) / (theta_amb ** 3.3)) ** 0.5
        NOxEI = NOxEI_sl * correction
    else:
        NOxEI = NOxEI_sl

    # ----------------------------------------------------------------------------
    # 4. Determine thrust category for each fuelfactor point
    #    Categories: 1=High (H), 2=Low (L), 3=Approach (A)
    # ----------------------------------------------------------------------------
    thrustCat = get_thrust_cat(ff_eval,ff_cal, cruiseCalc)

    # ----------------------------------------------------------------------------
    # 5. Speciation 
    # ----------------------------------------------------------------------------
    noProp, no2Prop, honoProp = NOx_speciation(thrustCat)

    # ----------------------------------------------------------------------------
    # 6. Compute component EIs
    # ----------------------------------------------------------------------------
    if np.isnan(NOxEI).any():
        warnings.warn("NaN encountered in NOxEI calculation.", RuntimeWarning)

    NOEI = NOxEI * noProp     # g NO / kg fuel
    NO2EI = NOxEI * no2Prop   # g NO2 / kg fuel
    HONOEI = NOxEI * honoProp # g HONO / kg fuel

    return NOxEI, NOEI, NO2EI, HONOEI, noProp, no2Prop, honoProp


def NOx_speciation(thrustCat):
    n_times = thrustCat.shape[0]
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
    return noProp, no2Prop, honoProp
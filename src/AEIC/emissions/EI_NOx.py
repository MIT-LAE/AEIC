import warnings
from dataclasses import dataclass

import numpy as np

from AEIC.utils.standard_fuel import get_thrust_cat


@dataclass(frozen=True)
class BFFM2EINOxResult:
    """Bundled NOx emissions indices and speciation data."""

    NOxEI: np.ndarray
    NOEI: np.ndarray
    NO2EI: np.ndarray
    HONOEI: np.ndarray
    noProp: np.ndarray
    no2Prop: np.ndarray
    honoProp: np.ndarray


def BFFM2_EINOx(
    sls_equiv_fuel_flow: np.ndarray,
    NOX_EI_matrix: np.ndarray,
    fuelflow_performance: np.ndarray,
    Tamb: np.ndarray,
    Pamb: np.ndarray,
    cruiseCalc: bool = True,
) -> BFFM2EINOxResult:
    """
    Calculate NOx, NO, NO2, and HONO emission indices
    All inputs are 1-dimensional arrays of equal length for calibration
    (fuelflow_KGperS vs. NOX_EI_matrix)
    and 1-dimensional for SLS_equivalent_fuel_flow (multiple evaluation points).

    Parameters
    ----------
    fuelflow_trajectory : ndarray, shape (n_times,)
        Fuel flow at which to compute EI.
    NOX_EI_matrix : ndarray, shape (n_cal,)
        Baseline NOx EI values [g NOx / kg fuel]
        corresponding to calibration fuel flows.
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
    BFFM2EINOxResult
        Structured NOx EI arrays and speciation fractions.
    """

    # ---------------------------------------------------------------------
    # 1) Piece-wise log–log interpolation (Idle→App→Climb→TO)
    # ---------------------------------------------------------------------
    ff_cal = np.asarray(fuelflow_performance, dtype=float).copy()
    ei_cal = np.asarray(NOX_EI_matrix, dtype=float).copy()
    ff_eval = np.asarray(sls_equiv_fuel_flow, dtype=float).copy()

    ff_cal[ff_cal <= 0] = 1e-2
    ff_eval[ff_eval <= 0] = 1e-2

    # log-space abscissa / ordinate
    x_cal = np.log10(ff_cal)  # length-4, guaranteed order (Idle→TO)
    y_cal = np.log10(ei_cal)
    x_eval = np.log10(ff_eval)

    # 1a. In-range piece-wise linear interpolation
    y_interp = np.interp(x_eval, x_cal, y_cal)  # left/right handled next

    # 1b. Linear log-log extrapolation below Idle and above Take-off
    below = x_eval < x_cal[0]
    if below.any():
        slope_low = (y_cal[1] - y_cal[0]) / (x_cal[1] - x_cal[0])
        y_interp[below] = y_cal[0] + slope_low * (x_eval[below] - x_cal[0])

    above = x_eval > x_cal[-1]
    if above.any():
        slope_high = (y_cal[-1] - y_cal[-2]) / (x_cal[-1] - x_cal[-2])
        y_interp[above] = y_cal[-1] + slope_high * (x_eval[above] - x_cal[-1])

    NOxEI_sl = 10.0**y_interp  # back to linear space  g/kg fuel

    # 2) If cruiseCalc=True, apply the humidity/θ/δ correction (Eqs. 44–45)
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
        Pv = 0.014504 * (10.0**beta)  # [psia]
        phi = 0.6  # 60% relative humidity
        omega = (0.62198 * phi * Pv) / (Pamb_psia - (phi * Pv))
        H = -19.0 * (omega - 0.0063)

        # Eq. (45) ambient correction:
        correction = np.exp(H) * ((delta_amb**1.02) / (theta_amb**3.3)) ** 0.5
        NOxEI = NOxEI_sl * correction
    else:
        NOxEI = NOxEI_sl

    # ---------------------------------------------------------------------
    # 3) Map each evaluation point to thrust category
    #    Low  : ≤ Idle fuel-flow  (HC/CO “L”)
    #    Appr.: (Idle, Approach]  (NOx “A”)
    #    High : > Approach        (NOx “H”, includes Climb & TO)
    # ---------------------------------------------------------------------
    thrustCat = get_thrust_cat(sls_equiv_fuel_flow, fuelflow_performance, cruiseCalc)

    # 4) Speciation fractions (unchanged)
    noProp, no2Prop, honoProp = NOx_speciation(thrustCat)

    # ----------------------------------------------------------------------------
    # 5. Compute component EIs
    # ----------------------------------------------------------------------------
    if np.isnan(NOxEI).any():
        warnings.warn("NaN encountered in NOxEI calculation.", RuntimeWarning)

    NOEI = NOxEI * noProp  # g NO / kg fuel
    NO2EI = NOxEI * no2Prop  # g NO2 / kg fuel
    HONOEI = NOxEI * honoProp  # g HONO / kg fuel

    return BFFM2EINOxResult(
        NOxEI=NOxEI,
        NOEI=NOEI,
        NO2EI=NO2EI,
        HONOEI=HONOEI,
        noProp=noProp,
        no2Prop=no2Prop,
        honoProp=honoProp,
    )


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
#         a,b,c,d,e,f,g,h,i,j = [ 8.46329738,  0.00980137, -8.55054025,  0.00981223,
# 0.02928154,
#             0.01037376,  0.03666156,  0.01037419,  0.03664096,  0.01037464]

#         H = -19.0*(sp_humidity - 0.00634)

#         NOxEI = np.exp(H)*(P3_kPa**0.4) * (a * np.exp(b * T3_K) + c * \
# np.exp(d * T3_K) + e * np.exp(f * T3_K) + g * np.exp(h * T3_K) + i * np.exp(j * T3_K))
#     elif mode == "BFFM2":
#         return BFFM2_EINOx(fuel_flow_trajectory,NOX_EI_input,fuel_flow_performance,
# Tamb,Pamb,mach_number)
#     else:
#         raise Exception("Invalid mode input in EI NOx function (BFFM2, P3T3)")

#     # Thrust category assignment
#     thrustCat = get_thrust_cat(fuel_flow_trajectory,
# fuel_flow_performance, cruiseCalc)

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

import numpy as np

def get_thrust_cat(ff_eval, ff_cal, cruiseCalc):
    n_times = ff_eval.shape[0]
    thrustCat = np.zeros(n_times, dtype=int)

    if cruiseCalc:
        if ff_eval.size < 3:
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
        # LTO case: assume exactly 4 calibration points (LTO modes)
        # Categories fixed: [2, 2, 3, 1]
        if ff_eval.size != 4:
            raise ValueError("When cruiseCalc=False, fuelflow_KGperS must have length 11.")
        base = np.array([2, 2, 3, 1], dtype=int)
        # We linearly interpolate each fuelfactor against the 11-point calibration?
        # But MATLAB simply tiles these 11 categories across each column. Since here we have
        # 1D fuelfactor, we assume it also has length 11 in the pure LTO scenario.
        if n_times != 4:
            raise ValueError("When cruiseCalc=False, fuelfactor must have length 4.")
        thrustCat = base.copy()
    return thrustCat


def get_fuel_factor(fuel_flow, Pamb, mach_number, z = 3.8, P_SL = 101325.0):
    # -----------------------------
    # 0. Convert Wf_alt → Wf_SL (Eq. (40)):
    # -----------------------------
    # δ_amb
    delta_amb = Pamb / P_SL

    # Mach factor = exp(0.2 * M3^2)
    mach_term = np.exp(0.2 * mach_number**2)

    # Sea-level fuel flow = Wf_alt * (1/δ_amb)^z * Mach-term
    Wf_SL = fuel_flow * ( (1.0 / delta_amb) ** z ) * mach_term

    return Wf_SL
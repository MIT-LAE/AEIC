"""
Module with the BADA3 aircraft parameters class

- uses the BaseAircraftParameter class from fuel_burn_base.py
"""

from dataclasses import dataclass

from .fuel_burn_base import BaseAircraftParameters


@dataclass
class Bada3AircraftParameters(BaseAircraftParameters):
    """BADA3 aircraft parameters.

    This class implements the BADA3 aircraft parameters. It is based on the
    BADA3.16 User Manual

    """

    ac_type: str | None = None
    wake_cat: str | None = None
    c_fcr: float | None = None
    c_f1: float | None = None
    c_f2: float | None = None
    c_f3: float | None = None
    c_f4: float | None = None
    c_d0cr: float | None = None
    c_d2cr: float | None = None
    S_ref: float | None = None
    ref_mass: float | None = None
    min_mass: float | None = None
    max_mass: float | None = None
    max_payload: float | None = None
    V_MO: float | None = None
    M_MO: float | None = None
    H_MO: float | None = None
    c_tc1: float | None = None
    c_tc2: float | None = None
    c_tc3: float | None = None
    c_tc4: float | None = None
    c_tc5: float | None = None
    c_tcr: float | None = 0.95
    c_tdes_low: float | None = None
    c_tdes_high: float | None = None
    h_p_des: float | None = None
    c_tdes_app: float | None = None
    c_tdes_ld: float | None = None
    engine_type: str | None = None
    cas_cruise_lo: float | None = None
    cas_cruise_hi: float | None = None
    cas_cruise_mach: float | None = None

    def assign_parameters_fromdict(self, parameters: dict):
        """
        Assigns the parameters from a dictionary.
        """
        for key in parameters:
            setattr(self, key, parameters[key])

    def get_params_asdict(self):
        """
        Returns the parameters as a dictionary.
        """
        return self.__dict__

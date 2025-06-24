"""
Module with the BADA3 aircraft parameters class

- uses the BaseAircraftParameter class from fuel_burn_base.py
"""

from .fuel_burn_base import BaseAircraftParameters

from dataclasses import dataclass

from typing import Optional


@dataclass
class Bada3AircraftParameters(BaseAircraftParameters):
    """BADA3 aircraft parameters.

    This class implements the BADA3 aircraft parameters. It is based on the
    BADA3.16 User Manual

    """

    ac_type: Optional[str] = None
    wake_cat: Optional[str] = None
    c_fcr: Optional[float] = None
    c_f1: Optional[float] = None
    c_f2: Optional[float] = None
    c_f3: Optional[float] = None
    c_f4: Optional[float] = None
    c_d0cr: Optional[float] = None
    c_d2cr: Optional[float] = None
    S_ref: Optional[float] = None
    ref_mass: Optional[float] = None
    min_mass: Optional[float] = None
    max_mass: Optional[float] = None
    max_payload: Optional[float] = None
    V_MO: Optional[float] = None
    M_MO: Optional[float] = None
    H_MO: Optional[float] = None
    c_tc1: Optional[float] = None
    c_tc2: Optional[float] = None
    c_tc3: Optional[float] = None
    c_tc4: Optional[float] = None
    c_tc5: Optional[float] = None
    c_tcr: Optional[float] = 0.95
    c_tdes_low: Optional[float] = None
    c_tdes_high: Optional[float] = None
    h_p_des: Optional[float] = None
    c_tdes_app: Optional[float] = None
    c_tdes_ld: Optional[float] = None
    engine_type: Optional[str] = None
    cas_cruise_lo: Optional[float] = None
    cas_cruise_hi: Optional[float] = None
    cas_cruise_mach: Optional[float] = None
     
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

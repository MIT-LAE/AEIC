from dataclasses import dataclass
from functools import cached_property

import numpy as np


# TODO: Import this into environment config and full Config
@dataclass
class deltaT_config:
    # Specific heat of liquid water
    c_water = 4.2e3  # J*K^-1*kg^-1
    # Density of water
    rho = 1000  # kg/m^3

    # Specific heat of Mixed Layer (70m)
    specific_heat = 441333333.33  # J*K^-1*m^-2
    # Heat capacity of the Mixed Layer
    C1 = specific_heat * 0.71  # J*K^-1*m^-2
    # Heat Capacity of Deep Ocean (3000m)
    C2 = 14700000000  # J*K^-1*m^-2

    # Equilibrium Climate Sensitivity (ECS)
    # The equilibrium temperature change (K or C) that results from doubling CO2
    # Typical range**: 1.5 - 4.5 K (IPCC AR5)
    ECS = 4.0  # K
    # Radiative Forcing from CO₂ Doubling
    RF2xCO2 = 3.93  # W/m^2

    # Advective mass flux of water from boundary layer
    advective_mass = 1.435e-4  # kg*m^-2*s^-1
    # Diffusion coefficient for turbulent Mixing
    diffuse_coeff = 7e-5  # m^2*s^-1
    # Mixing depth
    z = 1166.6667  # m

    @cached_property
    def lambda_climate(self):
        """
        Climate Sensitivity Parameter: Temperature response per unit radiative forcing
        units: K/Wm^-2
        """
        return self.ECS / self.RF2xCO2

    @cached_property
    def tau_deep(self):
        """
        Ocean Mixed Layer Response Time
        units: seconds
        """
        return self.C1 * self.lambda_climate

    @cached_property
    def alpha_1(self):
        """
        Mixed Layer Heat Exchange Coefficient
        Rate of heat exchange between the surface mixed layer and deep ocean.
        units: seconds^-1
        """
        return self._get_alpha(self.C1)

    @cached_property
    def alpha_2(self):
        """
        Deep Ocean Heat Exchange Coefficient
        Rate of heat exchange between deep ocean and surface mixed layer
        units: seconds^-1
        """
        return self._get_alpha(self.C2)

    def _get_alpha(self, C):
        return (
            self.c_water
            / C
            * (self.advective_mass + (self.diffuse_coeff * self.rho) / self.z)
        )


@dataclass(frozen=True)
class deltaT_2box_output:
    deltaT_surface: np.ndarray
    deltaT_deep: np.ndarray


def deltaT_2box(RF, config):
    """
    Two-box temperature response model with:

    Mixed layer (ocean surface, ~70m): heat capacity C1
    Deep ocean (~3000m): heat capacity C2
    Climate sensitivity: lambda_climate = ECS / RF2xCO2 (K/(W/m²))
    Heat exchange: advection + diffusion

    For each forcer:

    Calculates surface temperature change: ΔT_surface
    Calculates deep ocean temperature change: ΔT_deep
    Uses iterative forward stepping through time
    """
    years = len(RF)
    sec_per_yr = 60 * 60 * 24 * 365.25  # seconds per year

    # Initialize arrays
    deltaT_surface = np.zeros_like(RF)
    deltaT_deep = np.zeros_like(RF)

    # Main calculation loop
    for j in range(years - 1):
        deltaT_surface[j + 1] = (
            sec_per_yr
            * (
                RF[j] / config.C1
                - deltaT_surface[j] / config.tau_deep
                - config.alpha_1 * (deltaT_surface[j] - deltaT_deep[j])
            )
            + deltaT_surface[j]
        )
        deltaT_deep[j + 1] = (
            sec_per_yr * config.alpha_2 * (deltaT_surface[j] - deltaT_deep[j])
            + deltaT_deep[j]
        )

    return deltaT_2box_output(deltaT_surface=deltaT_surface, deltaT_deep=deltaT_deep)

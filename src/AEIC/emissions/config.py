# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from typing import ClassVar

from AEIC.utils.models import CIBaseModel, CIStrEnum


class EINOxMethod(CIStrEnum):
    """Config for selecting input modes for NOx emissions"""

    # NOx emission method options
    BFFM2 = "bffm2"
    P3T3 = "p3t3"
    NONE = "none"


class PMvolMethod(CIStrEnum):
    """Config for selecting input modes for PMvol emissions"""

    # PMvol emission method options
    FUEL_FLOW = "fuel_flow"
    FOA3 = "foa3"
    NONE = "none"


class PMnvolMethod(CIStrEnum):
    """Config for selecting input modes for PMnvol emissions"""

    # PMnvol emission method options
    MEEM = "meem"
    SCOPE11 = "scope11"
    FOA3 = "foa3"
    NONE = "none"


class EmissionsConfig(CIBaseModel):
    """Validated user-configurable inputs for emissions modeling.

    Wraps the raw performance-model config/TOML, enforces defaults, and keeps
    every param needed to build emission settings (fuel selection, method
    choices, LTO input mode, and optional components like APU/GSE/lifecycle)."""

    DEFAULT_METHOD: ClassVar[EINOxMethod] = EINOxMethod.BFFM2

    # Fuel Info
    fuel: str  # Fuel used (conventional Jet-A, SAF, etc)

    # Trajectory emissions config
    climb_descent_usage: bool = True

    # Emission calculation flags for only fuel dependent emission calculations
    co2_enabled: bool = True
    h2o_enabled: bool = True
    sox_enabled: bool = True

    # Emission calculation method options for all other emmisions
    nox_method: EINOxMethod = DEFAULT_METHOD
    hc_method: EINOxMethod = DEFAULT_METHOD
    co_method: EINOxMethod = DEFAULT_METHOD
    pmvol_method: PMvolMethod = PMvolMethod.FUEL_FLOW
    pmnvol_method: PMnvolMethod = PMnvolMethod.MEEM

    # Non trajectory emission calculation flags
    apu_enabled: bool = True
    gse_enabled: bool = True
    lifecycle_enabled: bool = True

    @property
    def fuel_file(self) -> str:
        """Get the fuel file path."""
        # Delayed import to avoid circular dependency.
        from AEIC.config import config

        return config.file_location(f'fuels/{self.fuel}.toml')

    @property
    def nox_enabled(self) -> bool:
        return self.nox_method != EINOxMethod.NONE

    @property
    def hc_enabled(self) -> bool:
        return self.hc_method != EINOxMethod.NONE

    @property
    def co_enabled(self) -> bool:
        return self.co_method != EINOxMethod.NONE

    @property
    def pmvol_enabled(self) -> bool:
        return self.pmvol_method != PMvolMethod.NONE

    @property
    def pmnvol_enabled(self) -> bool:
        return self.pmnvol_method != PMnvolMethod.NONE

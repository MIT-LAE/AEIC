# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict

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
    """Configuration data for emissions module."""

    model_config = ConfigDict(frozen=True)
    """Configuration is frozen after creation."""

    DEFAULT_METHOD: ClassVar[EINOxMethod] = EINOxMethod.BFFM2
    """Default method for NOx, HC, and CO emissions calculations."""

    fuel: str
    """Fuel used (conventional Jet-A, SAF, etc.)."""

    # Trajectory emissions config

    climb_descent_usage: bool = True
    """Flag controlling flight phases for which emissions are calculated. If
    true, emissions are calculated for the entire trajectory (takeoff, climb,
    cruise, descent); if false, emissions are only calculated for cruise and
    LTO data is used for takeoff, climb and approach."""

    # Emission calculation flags for only fuel dependent emission calculations.

    co2_enabled: bool = True
    """CO2 emission calculation flag."""

    h2o_enabled: bool = True
    """H2O emission calculation flag."""

    sox_enabled: bool = True
    """SOx emission calculation flag."""

    # Emission calculation method options for all other emmisions

    nox_method: EINOxMethod = DEFAULT_METHOD
    """NOx emission calculation method."""

    hc_method: EINOxMethod = DEFAULT_METHOD
    """HC emission calculation method."""

    co_method: EINOxMethod = DEFAULT_METHOD
    """CO emission calculation method."""

    pmvol_method: PMvolMethod = PMvolMethod.FUEL_FLOW
    """PMvol emission calculation method."""

    pmnvol_method: PMnvolMethod = PMnvolMethod.MEEM
    """PMnvol emission calculation method."""

    # Non trajectory emission calculation flags.

    apu_enabled: bool = True
    """APU emission calculation flag."""

    gse_enabled: bool = True
    """GSE emission calculation flag."""

    lifecycle_enabled: bool = True
    """Lifecycle emission calculation flag."""

    @property
    def fuel_file(self) -> str:
        """Fuel file path."""
        # Delayed import to avoid circular dependency.
        from AEIC.config import config

        return config.file_location(f'fuels/{self.fuel}.toml')

    @property
    def nox_enabled(self) -> bool:
        """NOx emission calculation flag."""
        return self.nox_method != EINOxMethod.NONE

    @property
    def hc_enabled(self) -> bool:
        """HC emission calculation flag."""
        return self.hc_method != EINOxMethod.NONE

    @property
    def co_enabled(self) -> bool:
        """CO emission calculation flag."""
        return self.co_method != EINOxMethod.NONE

    @property
    def pmvol_enabled(self) -> bool:
        """PMvol emission calculation flag."""
        return self.pmvol_method != PMvolMethod.NONE

    @property
    def pmnvol_enabled(self) -> bool:
        """PMnvol emission calculation flag."""
        return self.pmnvol_method != PMnvolMethod.NONE

# TODO: Remove this when we migrate to Python 3.14.
from __future__ import annotations

from functools import cached_property
from typing import ClassVar

from pydantic import ConfigDict

from AEIC.types import Species
from AEIC.utils.models import CIBaseModel, CIStrEnum


class EINOxMethod(CIStrEnum):
    """NOₓ emissions calculation methods."""

    BFFM2 = 'bffm2'
    P3T3 = 'p3t3'
    NONE = 'none'


class PMvolMethod(CIStrEnum):
    """PMvol emissions calculation methods."""

    FUEL_FLOW = 'fuel_flow'
    FOA3 = 'foa3'
    NONE = 'none'


class PMnvolMethod(CIStrEnum):
    """PMnvol emissions calculation methods."""

    MEEM = 'meem'
    SCOPE11 = 'scope11'
    FOA3 = 'foa3'
    NONE = 'none'


class ClimbDescentMode(CIStrEnum):
    """Climb/descent emissions calculation mode."""

    TRAJECTORY = 'trajectory'
    """Calculate emissions for the entire trajectory (takeoff, climb, cruise,
    descent)."""

    LTO = 'lto'
    """Calculate emissions only for cruise and use LTO data for takeoff,
    climb, and approach."""


class EmissionsConfig(CIBaseModel):
    """Configuration data for emissions module."""

    model_config = ConfigDict(frozen=True)
    """Configuration is frozen after creation."""

    DEFAULT_METHOD: ClassVar[EINOxMethod] = EINOxMethod.BFFM2
    """Default method for NOₓ, HC, and CO emissions calculations."""

    fuel: str
    """Fuel used (conventional Jet-A, SAF, etc.). Should be the name of a file
    accessible as "fuels/{fuel}.toml" using the AEIC configuration system."""

    # Trajectory emissions config

    climb_descent_mode: ClimbDescentMode = ClimbDescentMode.TRAJECTORY
    """Flag controlling how emissions are calculated for non-cruise flight
    phases. If set to 'trajectory', emissions are calculated for the entire
    trajectory (takeoff, climb, cruise, descent); if set to 'lto', emissions
    are only calculated for cruise and LTO data is used for takeoff, climb and
    approach."""

    # Emission calculation flags for only fuel dependent emission calculations.

    co2_enabled: bool = True
    """Toggle fuel-dependent CO₂ emission calculation."""

    h2o_enabled: bool = True
    """Toggle fuel-dependeny H₂O emission calculation."""

    sox_enabled: bool = True
    """Toggle fuel-dependency SOₓ (SO₂ and SO₄) emission calculation."""

    # Emission calculation method options for all other emmisions

    nox_method: EINOxMethod = DEFAULT_METHOD
    """NOₓ emission calculation method. ("None" disables NOₓ emissions.)"""

    hc_method: EINOxMethod = DEFAULT_METHOD
    """HC emission calculation method. ("None" disables HC emissions.)"""

    co_method: EINOxMethod = DEFAULT_METHOD
    """CO emission calculation method. ("None" disables CO emissions.)"""

    pmvol_method: PMvolMethod = PMvolMethod.FUEL_FLOW
    """PMvol emission calculation method. ("None" disables PMvol emissions.)"""

    pmnvol_method: PMnvolMethod = PMnvolMethod.MEEM
    """PMnvol emission calculation method. ("None" disables PMnvol emissions.)"""

    # Non trajectory emission calculation flags.

    apu_enabled: bool = True
    """Toggle APU emission calculation. (APU emissions are only calculated if
    the performance model provides an APU name.)"""

    gse_enabled: bool = True
    """Toggle GSE emission calculation."""

    lifecycle_enabled: bool = True
    """Toggle CO₂ lifecycle emission calculation."""

    @property
    def fuel_file(self) -> str:
        """Fuel file path."""
        # Delayed import to avoid circular dependency.
        from AEIC.config import config

        return config.file_location(f'fuels/{self.fuel}.toml')

    @property
    def nox_enabled(self) -> bool:
        """NOₓ emission calculation flag."""
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

    @cached_property
    def enabled_species(self) -> set[Species]:
        result = set()

        def add(*species: Species, label: str | None = None):
            if label is None:
                label = species[0].name.lower()
            if getattr(self, f'{label}_enabled'):
                for s in species:
                    result.add(s)

        add(Species.CO2)
        add(Species.H2O)
        add(Species.HC)
        add(Species.CO)
        add(Species.NOx, Species.NO, Species.NO2, Species.HONO)
        add(Species.PMvol, Species.OCic)
        add(Species.PMnvol, Species.PMnvolGMD)
        if self.pmnvol_method in (PMnvolMethod.SCOPE11, PMnvolMethod.MEEM):
            add(Species.PMnvolN, label='pmnvol')
        add(Species.SO2, Species.SO4, label='sox')

        return result

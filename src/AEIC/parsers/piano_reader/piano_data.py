"""Parsed contents of a set of PIANO exports, and the user-supplied overrides."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from AEIC.parsers.piano_reader.climb_reader import _parse_climb
from AEIC.parsers.piano_reader.cruise_reader import _parse_cruise
from AEIC.parsers.piano_reader.descent_reader import _parse_descent
from AEIC.performance.types import SpeedData, TableInput


@dataclass(frozen=True)
class PianoOverrides:
    """User-supplied values PIANO files do not always contain."""

    climb_masses_kg: list[float] | None = None
    """Initial mass of each climb block, in file order [kg].

    All-or-nothing: supply one mass per block, including the blocks whose
    header already states one. A supplied mass that contradicts a stated one
    warns.
    """

    climb_cas_low_kts: float | None = None
    """Climb calibrated airspeed below FL100 [knots]."""

    climb_cas_high_kts: float | None = None
    """Climb calibrated airspeed between FL100 and the crossover [knots]."""

    climb_mach: float | None = None
    """Climb Mach number above the crossover altitude."""

    climb_crossover_altitude_ft: float | None = None
    """Altitude above which the climb flies its Mach number [feet]."""


@dataclass
class PianoData:
    """Internal representation of the contents of a set of PIANO exports."""

    aircraft_name: str
    """Aircraft name, from the cruise table title."""

    maximum_altitude_ft: int
    """Highest altitude any climb block reaches [feet]."""

    isa_offset: int
    """ISA temperature offset [deg C]."""

    climb_speeds: SpeedData
    """Climb airspeed schedule."""

    descent_speeds: SpeedData
    """Descent airspeed schedule."""

    climb: TableInput
    """Climb performance table."""

    cruise: TableInput
    """Cruise performance table."""

    descent: TableInput
    """Descent performance table."""

    cruise_reference_mach: TableInput
    """Reference Mach numbers PIANO labels in the cruise sweep."""

    descent_idle_thrust: TableInput
    """Altitude below which each descent block flies at idle thrust."""

    @classmethod
    def load(
        cls,
        cruise_file: str,
        climb_file: str,
        descent_file: str,
        *,
        overrides: PianoOverrides | None = None,
    ) -> Self:
        """Read a set of PIANO cruise, climb and descent exports.

        Args:
            cruise_file: Path to the PIANO cruise table export.
            climb_file: Path to the PIANO climb details export.
            descent_file: Path to the PIANO descent details export.
            overrides: Values PIANO does not always write to the files.

        Raises:
            ValueError: If the files cannot be parsed, if the climb mass count
                does not match the climb block count, or if the ISA offset is
                non-zero.
        """
        overrides = overrides if overrides is not None else PianoOverrides()

        aircraft_name, cruise, reference_mach = _parse_cruise(cruise_file)
        climb_speeds, isa_offset, maximum_altitude_ft, climb = _parse_climb(
            climb_file, overrides
        )
        descent_speeds, descent, idle_thrust = _parse_descent(descent_file)

        return cls(
            aircraft_name=aircraft_name,
            maximum_altitude_ft=maximum_altitude_ft,
            isa_offset=isa_offset,
            climb_speeds=climb_speeds,
            descent_speeds=descent_speeds,
            climb=climb,
            cruise=cruise,
            descent=descent,
            cruise_reference_mach=reference_mach,
            descent_idle_thrust=idle_thrust,
        )

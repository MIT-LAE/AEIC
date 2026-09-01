"""Reader for the PIANO climb details export."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from AEIC.parsers.piano_reader.common import (
    _Block,
    _cross_check_fuel_burn,
    _cross_check_tas,
    _deltas,
    _find,
    _parse_schedule,
    _read_lines,
    _reject_malformed,
    _Schedule,
    _split_blocks,
    _tas_from_schedule,
)
from AEIC.performance.types import SpeedData, TableInput
from AEIC.units import (
    FEET_TO_METERS,
    FPM_TO_MPS,
    NAUTICAL_MILES_TO_METERS,
    POUNDS_FORCE_TO_NEWTONS,
    POUNDS_TO_KG,
)

if TYPE_CHECKING:
    from AEIC.parsers.piano_reader.piano_data import PianoOverrides

logger = logging.getLogger(__name__)

#############################################################################
# PIANO climb file schema
#############################################################################
#
# A climb file is structured as one block per initial mass.
# Every block is in two parts: a header or halt note, and a "Climb details"
# table.
#
# If the climb reaches the target altitude, PIANO adds a header above
# the "Climb details table.".
# If it does not reach the target altitude, the header block is replaced by
# a halt note instead.
# `_split_blocks` therefore keys on the "Climb details" marker, and treats
# everything since the previous marker as the block's header lines.
#
# 1a. Block header, after a completed climb:
#
#     Climb from 0.feet to:   41000.feet
#     ---------------------------------------------------------------
#     Time                28.00     minutes
#     Fuel burn           4000.     lb.
#     Distance            200.0     n.miles
#
#     Initial mass        150000.   lb.
#     Airspeed schedule   250./ 280.kcas/ mach 0.750 above 30000.feet
#     Delta-ISA           +0.       deg.C.
#     ---------------------------------------------------------------
#
# The reader takes four values from the header:
#   - "Initial mass" gives the block's mass, unless the caller supplies one
#     (`_climb_masses`).
#   - "Airspeed schedule" gives the CAS below and above FL100, the Mach
#     number and the crossover altitude (`_climb_schedule`). It is a property
#     of the file, so the first block that states one sets it every block.
#   - "Delta-ISA" gives the ISA offset (`_parse_isa_offset`). A non-zero
#     offset is rejected, because TAS is derived for a standard atmosphere.
#   - "Fuel burn" is the block total the last data row is checked against
#     (`_cross_check_fuel_burn`).
#
# 1b. Halt note, written in place of the next block's header:
#
#     Climb to 41000.feet halted at 40000.feet,
#     Rate of Climb < 50.feet/min.
#     -----------------------------------------
#
# 2. Data table, one per block:
#
#     Climb details
#
#      Alt.     Time      Dist.      Burn      FN/eng    R.o.C.     Drag
#     (feet)    (sec)   (n.miles)    (lb.)     (lbf.)    (f.p.m)    (lbf.)
#
#         0.       0.       0.0         0.     30000.    2000.     1000.
#      1417.      60.       5.0        40.     30000.    2000.     1000.
#
# Rows run from the ground up, one per altitude step, at the mass the header
# states. "Time", "Dist." and "Burn" are cumulative from the start of the
# climb. "FN/eng", "R.o.C." and "Drag" are the value at that altitude.
# `_is_data_row` relies on the fact that only data rows start with a number
# to identify data rows.
#############################################################################

# Climb table columns
CLIMB_COLS = [
    'fl',
    'mass',
    'tas',
    'rocd',
    'fuel_flow',
    'time',
    'distance',
    'burn',
    'fn_per_engine',
    'drag',
]

# If user supplied masses and PIANO header masses are both present
# we check that they are consistent up to 1%
_CLIMB_MASS_CROSS_CHECK_TOLERANCE = 0.01

# Alt., Time, Dist., Burn, FN/eng, R.o.C., Drag
_CLIMB_ROW_COLS = 7

_CLIMB_MASS_RE = re.compile(r'Initial mass\s+([\d.]+)')
_DELTA_ISA_RE = re.compile(r'Delta-ISA\s+([+-]?[\d.]+)')
_HALTED_RE = re.compile(r'halted at\s+([\d.]+)feet')


# ---------------------------------------------------------------------------
# Climb
# ---------------------------------------------------------------------------


def _parse_climb(
    path: str, overrides: PianoOverrides
) -> tuple[SpeedData, int, int, TableInput]:
    """Parse a PIANO climb details export.

    Returns the climb speed schedule, the ISA offset, the highest altitude
    reached by any block, and the climb table.
    """
    lines = _read_lines(path)
    blocks = _split_blocks(lines, 'Climb details', _CLIMB_ROW_COLS)

    isa_offset = _parse_isa_offset(lines)
    masses = _climb_masses(blocks, overrides)
    schedule = _climb_schedule(blocks, overrides)

    for match in (_HALTED_RE.search(line) for line in lines):
        if match is not None:
            logger.info(
                'PIANO climb halted at %s feet, so the block stops there',
                match.group(1).rstrip('.'),
            )

    data = []
    maximum_altitude_ft = 0.0
    for n, (block, mass) in enumerate(zip(blocks, masses, strict=True)):
        label = f'Climb block {n + 1}'
        _reject_malformed(label, block, _CLIMB_ROW_COLS)

        rows = block.rows
        if not rows:
            logger.warning('%s: no data rows', label)
            continue

        altitude_ft = [row[0] for row in rows]
        maximum_altitude_ft = max(maximum_altitude_ft, max(altitude_ft))
        altitude_m = [ft * FEET_TO_METERS for ft in altitude_ft]
        tas = [_tas_from_schedule(alt, schedule) for alt in altitude_m]
        rocd = [row[5] * FPM_TO_MPS for row in rows]

        d_time = _deltas([row[1] for row in rows])
        d_distance = _deltas([row[2] * NAUTICAL_MILES_TO_METERS for row in rows])
        d_burn = _deltas([row[3] * POUNDS_TO_KG for row in rows])

        _cross_check_tas(label, tas, d_distance, d_time, rocd)
        _cross_check_fuel_burn(label, rows[-1][3] * POUNDS_TO_KG, block.header_lines)

        for i, row in enumerate(rows):
            if d_time[i] == 0:
                logger.warning(
                    '%s: dropping the row at %.0f feet, which spans no time '
                    'so has no defined fuel flow',
                    label,
                    row[0],
                )
                continue
            data.append(
                [
                    row[0] / 100,
                    mass,
                    tas[i],
                    rocd[i],
                    d_burn[i] / d_time[i],
                    row[1],
                    row[2] * NAUTICAL_MILES_TO_METERS,
                    row[3] * POUNDS_TO_KG,
                    row[4] * POUNDS_FORCE_TO_NEWTONS,
                    row[6] * POUNDS_FORCE_TO_NEWTONS,
                ]
            )

    table = TableInput(
        cols=CLIMB_COLS, data=sorted(data, key=lambda row: (row[1], row[0]))
    )
    return (
        schedule.to_speed_data(),
        isa_offset,
        int(maximum_altitude_ft),
        table,
    )


def _parse_isa_offset(lines: list[str]) -> int:
    """Parse the ISA temperature offset from a PIANO climb file.

    Raises:
        ValueError: If the offset is non-zero. Deriving TAS from the airspeed
            schedule assumes a standard atmosphere.
    """
    match = _find(_DELTA_ISA_RE, lines)
    if match is None:
        return 0
    offset = float(match.group(1))
    if offset != 0:
        raise ValueError(
            f'PIANO climb file reports a Delta-ISA of {offset:+g} deg C, but '
            'true airspeed is derived assuming a standard atmosphere'
        )
    return int(offset)


def _header_climb_mass(block: _Block) -> float | None:
    """Initial mass a climb block's header states [kg], or None if it states
    none. PIANO omits the header of a block that follows a halted climb."""
    match = _find(_CLIMB_MASS_RE, block.header_lines)
    return None if match is None else float(match.group(1)) * POUNDS_TO_KG


def _climb_masses(blocks: list[_Block], overrides: PianoOverrides) -> list[float]:
    """Resolve the initial mass of each climb block, in file order.

    A supplied mass wins over the header, because a user supplies masses
    when the headers do not carry them.
    The override is all-or-nothing: so a user passing any climb mass must pass masses
    for all blocks in file order. Warns when a supplied mass contradicts a mass the
    file states.

    Raises:
        ValueError: If the supplied mass count does not match the block count,
            or if a block has no header mass and none was supplied.
    """
    header_masses = [_header_climb_mass(block) for block in blocks]

    if overrides.climb_masses_kg is not None:
        if len(overrides.climb_masses_kg) != len(blocks):
            raise ValueError(
                f'Got {len(overrides.climb_masses_kg)} climb mass(es) for '
                f'{len(blocks)} climb block(s) in the PIANO climb file'
            )
        for n, (supplied, header) in enumerate(
            zip(overrides.climb_masses_kg, header_masses, strict=True)
        ):
            if header is None:
                continue
            if abs(supplied - header) > _CLIMB_MASS_CROSS_CHECK_TOLERANCE * header:
                logger.warning(
                    'Climb block %d: the supplied initial mass of %.1f kg does '
                    'not match the %.1f kg its header states; using the '
                    'supplied mass',
                    n + 1,
                    supplied,
                    header,
                )
        return list(overrides.climb_masses_kg)

    masses = []
    for n, header in enumerate(header_masses):
        if header is None:
            raise ValueError(
                f'Climb block {n + 1} has no "Initial mass" header, so the '
                'mass of every block must be supplied explicitly'
            )
        masses.append(header)
    return masses


def _climb_schedule(blocks: list[_Block], overrides: PianoOverrides) -> _Schedule:
    """Resolve the climb airspeed schedule.

    The schedule is a property of the file, so the first block that has one supplies it
    for every block.

    Raises:
        ValueError: If no block states a schedule and the overrides do not
            supply a complete one.
    """
    for block in blocks:
        schedule = _parse_schedule(block.header_lines, descending=False)
        if schedule is not None:
            return _apply_climb_overrides(schedule, overrides)

    cas_low_kts = overrides.climb_cas_low_kts
    cas_high_kts = overrides.climb_cas_high_kts
    mach = overrides.climb_mach
    crossover_ft = overrides.climb_crossover_altitude_ft

    missing = [
        name
        for name, value in (
            ('--climb-cas-low-kts', cas_low_kts),
            ('--climb-cas-high-kts', cas_high_kts),
            ('--climb-mach', mach),
            ('--climb-crossover-altitude-ft', crossover_ft),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            'No climb block states an airspeed schedule, so it must be '
            f'supplied explicitly. Missing: {", ".join(missing)}'
        )

    return _Schedule(
        cas_low_kts=cas_low_kts,
        cas_high_kts=cas_high_kts,
        mach=mach,
        crossover_ft=crossover_ft,
    )


def _apply_climb_overrides(schedule: _Schedule, overrides: PianoOverrides) -> _Schedule:
    """Overlay any user-supplied climb schedule values on a parsed schedule."""
    return _Schedule(
        cas_low_kts=(
            overrides.climb_cas_low_kts
            if overrides.climb_cas_low_kts is not None
            else schedule.cas_low_kts
        ),
        cas_high_kts=(
            overrides.climb_cas_high_kts
            if overrides.climb_cas_high_kts is not None
            else schedule.cas_high_kts
        ),
        mach=(
            overrides.climb_mach if overrides.climb_mach is not None else schedule.mach
        ),
        crossover_ft=(
            overrides.climb_crossover_altitude_ft
            if overrides.climb_crossover_altitude_ft is not None
            else schedule.crossover_ft
        ),
    )

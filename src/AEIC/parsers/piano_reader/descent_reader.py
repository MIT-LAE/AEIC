"""Reader for the PIANO descent details export."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import logging
import re

from AEIC.parsers.piano_reader.common import (
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

logger = logging.getLogger(__name__)

#############################################################################
# PIANO descent file schema
#############################################################################
#
# A descent file is structured as one block per mass, like the climb file.
# Unlike climb, PIANO writes a full header for every block, so a descent file
# should not have a halt note / no headerless block.
# `_split_blocks` keys on the "Descent details" marker.
#
# 1. Block header:
#
#     Descent from 41000.feet to:   0.feet
#     ---------------------------------------------------------------
#     Time                20.0      minutes
#     Fuel burn           800.      lb.
#     Distance            200.0     n.miles
#
#     Mass                100000.    lb.
#     Airspeed schedule   mach 0.750 above 30000.feet/ 280./ 250.kcas
#     Idle thrust below   30000.feet
#     ---------------------------------------------------------------
#
# The reader takes four values from the header:
#   - "Mass" gives the block's mass.
#   - "Airspeed schedule" gives the Mach number, the crossover altitude and
#     the CAS above and below FL100. The first block sets the schedule for
#     every block as speed schedule is constant for the file.
#   - "Idle thrust below" gives the altitude below which the descent flies at
#     idle thrust.
#   - "Fuel burn" is the block total the last data row is checked against
#     (`_cross_check_fuel_burn`).
#
# A descent file states no Delta-ISA, so the ISA offset comes from the climb
# file.
#
# 2. Data table, one per block:
#
#     Descent details
#
#      Alt.     Time      Dist.      Burn     R.o.D.     FN/eng
#     (feet)    (sec)    (n.miles)   (lb.)    (f.p.m)    (lbf.)
#
#     41000.      00.      200.        30.     1400.     1500.
#     39610.      01.      200.        30.     1400.     1500.
#
# Rows run from the top down, one per altitude step, at the mass the header
# states. "Time", "Dist." and "Burn" are cumulative from the start of the
# descent. "R.o.D." and "FN/eng" are the value at that altitude.
# PIANO reports rate of descent as a positive number.
#############################################################################

# Descent table columns
DESCENT_COLS = [
    'fl',
    'mass',
    'tas',
    'rocd',
    'fuel_flow',
    'time',
    'distance',
    'burn',
    'fn_per_engine',
]

# Descent idle thrust table columns
DESCENT_IDLE_THRUST_COLS = ['mass', 'idle_thrust_altitude']

# Alt., Time, Dist., Burn, R.o.D., FN/eng
_DESCENT_ROW_COLS = 6

_DESCENT_MASS_RE = re.compile(r'^\s*Mass\s+([\d.]+)')
_IDLE_THRUST_RE = re.compile(r'Idle thrust below\s+([\d.]+)feet')


# ---------------------------------------------------------------------------
# Descent
# ---------------------------------------------------------------------------


def _parse_descent(path: str) -> tuple[SpeedData, TableInput, TableInput]:
    """Parse a PIANO descent details export.

    Returns the descent speed schedule, the descent table and the idle thrust
    table.

    Raises:
        ValueError: If a block has no mass or no airspeed schedule header.
            Unlike climb, PIANO writes a full header for every descent block.
    """
    lines = _read_lines(path)
    blocks = _split_blocks(lines, 'Descent details', _DESCENT_ROW_COLS)

    data = []
    idle_thrust = []
    schedule: _Schedule | None = None

    for n, block in enumerate(blocks):
        label = f'Descent block {n + 1}'
        _reject_malformed(label, block, _DESCENT_ROW_COLS)

        mass_match = _find(_DESCENT_MASS_RE, block.header_lines)
        if mass_match is None:
            raise ValueError(f'{label} has no "Mass" header')
        mass = float(mass_match.group(1)) * POUNDS_TO_KG

        block_schedule = _parse_schedule(block.header_lines, descending=True)
        if block_schedule is None:
            raise ValueError(f'{label} has no "Airspeed schedule" header')
        if schedule is None:
            schedule = block_schedule
        elif block_schedule != schedule:
            logger.warning(
                '%s states an airspeed schedule that differs from the first '
                'block; using the first block for every block',
                label,
            )

        idle_match = _find(_IDLE_THRUST_RE, block.header_lines)
        if idle_match is None:
            logger.warning('%s has no "Idle thrust below" header', label)
        else:
            idle_thrust.append([mass, float(idle_match.group(1)) * FEET_TO_METERS])

        rows = block.rows
        if not rows:
            logger.warning('%s: no data rows', label)
            continue

        altitude_m = [row[0] * FEET_TO_METERS for row in rows]
        tas = [_tas_from_schedule(alt, schedule) for alt in altitude_m]
        # PIANO reports rate of descent as a positive number.
        rocd = [-row[4] * FPM_TO_MPS for row in rows]

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
                    row[5] * POUNDS_FORCE_TO_NEWTONS,
                ]
            )

    if schedule is None:
        raise ValueError(f'No descent block found in {path}')

    return (
        schedule.to_speed_data(),
        TableInput(
            cols=DESCENT_COLS, data=sorted(data, key=lambda row: (row[1], row[0]))
        ),
        TableInput(
            cols=DESCENT_IDLE_THRUST_COLS,
            data=sorted(idle_thrust, key=lambda row: row[0]),
        ),
    )

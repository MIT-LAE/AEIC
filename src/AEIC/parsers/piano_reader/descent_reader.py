"""Reader for the PIANO descent details export."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import logging
import re
from typing import NamedTuple

from AEIC.parsers.piano_reader.common import (
    _cross_check_fuel_burn,
    _cross_check_tas,
    _deltas,
    _find,
    _parse_schedule,
    _read_lines,
    _reject_malformed,
    _row_fuel_flows,
    _Schedule,
    _split_blocks,
    _tas_from_schedule,
)
from AEIC.parsers.piano_reader.rows import PianoDescentRow
from AEIC.performance.types import SpeedData, TableInput
from AEIC.units import (
    FEET_TO_METERS,
    FPM_TO_MPS,
    NAUTICAL_MILES_TO_METERS,
    POUNDS_FORCE_TO_NEWTONS,
    POUNDS_TO_KG,
)

logger = logging.getLogger(__name__)


class DescentTableRow(NamedTuple):
    """A row of the output descent table, in SI units except for the flight level.

    See the PIANO parser doc for the expected schema. Consumers read this table
    by column name, so the field order only sets the column order of the output.
    """

    fl: float
    mass: float
    tas: float
    rocd: float
    fuel_flow: float
    time: float
    distance: float
    burn: float
    fn_per_engine: float


class DescentIdleThrustTableRow(NamedTuple):
    """A row of the output descent idle thrust table, in SI units."""

    mass: float
    idle_thrust_altitude: float


DESCENT_COLS = list(DescentTableRow._fields)
DESCENT_IDLE_THRUST_COLS = list(DescentIdleThrustTableRow._fields)

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
        ValueError: If two blocks state different airspeed schedules. The
            schedule must be the same across the file.
    """
    lines = _read_lines(path)
    blocks = _split_blocks(lines, 'Descent details', PianoDescentRow.from_tokens)

    data: list[DescentTableRow] = []
    idle_thrust: list[DescentIdleThrustTableRow] = []
    schedule: _Schedule | None = None

    for n, block in enumerate(blocks):
        label = f'Descent block {n + 1}'
        _reject_malformed(label, block, len(PianoDescentRow.columns()))

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
            raise ValueError(
                f'{label} states the airspeed schedule {block_schedule}, which '
                f'differs from the first block\'s {schedule}; the airspeed '
                f'schedule must be the same across the file'
            )

        idle_match = _find(_IDLE_THRUST_RE, block.header_lines)
        if idle_match is None:
            logger.warning('%s has no "Idle thrust below" header', label)
        else:
            idle_thrust.append(
                DescentIdleThrustTableRow(
                    mass=mass,
                    idle_thrust_altitude=float(idle_match.group(1)) * FEET_TO_METERS,
                )
            )

        rows = block.rows
        if not rows:
            logger.warning('%s: no data rows', label)
            continue

        altitude_m = [row.altitude_ft * FEET_TO_METERS for row in rows]
        tas = [_tas_from_schedule(alt, schedule) for alt in altitude_m]
        # PIANO reports rate of descent as a positive number.
        rocd = [-row.rod_fpm * FPM_TO_MPS for row in rows]

        d_time = _deltas([row.time_s for row in rows])
        d_distance = _deltas(
            [row.distance_nm * NAUTICAL_MILES_TO_METERS for row in rows]
        )
        fuel_flow = _row_fuel_flows(
            [row.time_s for row in rows],
            [row.burn_lb * POUNDS_TO_KG for row in rows],
        )

        _cross_check_tas(label, tas, d_distance, d_time, rocd)
        _cross_check_fuel_burn(
            label, rows[-1].burn_lb * POUNDS_TO_KG, block.header_lines
        )

        for i, row in enumerate(rows):
            row_fuel_flow = fuel_flow[i]
            if row_fuel_flow is None:
                logger.warning(
                    '%s: dropping the row at %.0f feet, whose fuel flow '
                    'difference spans no time so is not defined',
                    label,
                    row.altitude_ft,
                )
                continue
            data.append(
                DescentTableRow(
                    fl=row.altitude_ft / 100,
                    mass=mass,
                    tas=tas[i],
                    rocd=rocd[i],
                    fuel_flow=row_fuel_flow,
                    time=row.time_s,
                    distance=row.distance_nm * NAUTICAL_MILES_TO_METERS,
                    burn=row.burn_lb * POUNDS_TO_KG,
                    fn_per_engine=row.fn_per_engine_lbf * POUNDS_FORCE_TO_NEWTONS,
                )
            )

    if schedule is None:
        raise ValueError(f'No descent block found in {path}')

    return (
        schedule.to_speed_data(),
        TableInput(
            cols=DESCENT_COLS,
            data=[list(row) for row in sorted(data, key=lambda r: (r.mass, r.fl))],
        ),
        TableInput(
            cols=DESCENT_IDLE_THRUST_COLS,
            data=[list(row) for row in sorted(idle_thrust, key=lambda r: r.mass)],
        ),
    )

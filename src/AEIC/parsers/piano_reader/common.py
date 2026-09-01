"""Parsing helpers shared by the PIANO cruise, climb and descent readers."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from AEIC.performance.types import SpeedData
from AEIC.units import (
    FEET_TO_METERS,
    FL_TO_METERS,
    KNOTS_TO_MPS,
    POUNDS_TO_KG,
)
from AEIC.utils.standard_atmosphere import (
    cas_to_tas,
    speed_of_sound_at_altitude,
)

logger = logging.getLogger(__name__)

# Relative deviation above which the TAS cross-check warns
# PIANO climb/descent does not produce TAS but we can estimate it
# and compare with a tolerance to ground speed and distance
_TAS_CROSS_CHECK_TOLERANCE = 0.05

# Relative deviation above which the block fuel burn cross-check warns.
_FUEL_BURN_CROSS_CHECK_TOLERANCE = 0.01

_FUEL_BURN_RE = re.compile(r'Fuel burn\s+([\d.]+)')
_SCHEDULE_LINE_RE = re.compile(r'Airspeed schedule\s+(.+?)\s*$')
_SCHEDULE_MACH_RE = re.compile(r'mach\s+([\d.]+)')
_SCHEDULE_CROSSOVER_RE = re.compile(r'above\s+([\d.]+)feet')
_SCHEDULE_CAS_RE = re.compile(r'([\d.]+)\s*/\s*([\d.]+)\s*kcas')


# ---------------------------------------------------------------------------
# Shared helpers, used by all three files
# ---------------------------------------------------------------------------


def _read_lines(path: str) -> list[str]:
    with open(path, encoding='utf-8', errors='ignore') as f:
        return f.readlines()


def _find(pattern: re.Pattern[str], lines: list[str]) -> re.Match[str] | None:
    """First match of `pattern` across `lines`, if any."""
    for line in lines:
        match = pattern.search(line)
        if match is not None:
            return match
    return None


def _is_data_row(tokens: list[str]) -> bool:
    """Whether PIANO wrote these tokens as a table row.
    PIANO opens every data row with a number.
    """
    if not tokens:
        return False
    try:
        float(tokens[0])
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Climb and descent detail blocks
#
# PIANO writes both files in the same shape: one block per initial
# mass, holding a metadata header and a table of cumulative values.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Schedule:
    """A PIANO airspeed schedule, in the units the file reports."""

    cas_low_kts: float
    cas_high_kts: float
    mach: float
    crossover_ft: float

    def to_speed_data(self) -> SpeedData:
        """Convert to the performance model's SI speed data."""
        return SpeedData(
            cas_low=self.cas_low_kts * KNOTS_TO_MPS,
            cas_high=self.cas_high_kts * KNOTS_TO_MPS,
            mach=self.mach,
            crossover_altitude_m=self.crossover_ft * FEET_TO_METERS,
        )


def _parse_schedule(lines: list[str], descending: bool) -> _Schedule | None:
    """Parse an airspeed schedule from a block's header lines.

    PIANO writes the schedule in the order the phase flies it, so a climb
    reads ``250./ 280.kcas/ mach 0.750 above 30000.feet`` and a descent reads
    ``mach 0.750 above 30000.feet/ 280./ 250.kcas``. Set `descending` for the
    latter, where the high CAS comes first.
    """
    match = _find(_SCHEDULE_LINE_RE, lines)
    if match is None:
        return None
    text = match.group(1)

    mach = _SCHEDULE_MACH_RE.search(text)
    crossover = _SCHEDULE_CROSSOVER_RE.search(text)
    cas = _SCHEDULE_CAS_RE.search(text)
    if mach is None or crossover is None or cas is None:
        return None

    first, second = float(cas.group(1)), float(cas.group(2))
    cas_high, cas_low = (first, second) if descending else (second, first)
    return _Schedule(
        cas_low_kts=cas_low,
        cas_high_kts=cas_high,
        mach=float(mach.group(1)),
        crossover_ft=float(crossover.group(1)),
    )


def _tas_from_schedule(altitude_m: float, schedule: _Schedule) -> float:
    """True airspeed at an altitude flown on a PIANO airspeed schedule."""
    crossover_m = schedule.crossover_ft * FEET_TO_METERS
    if altitude_m > crossover_m:
        return schedule.mach * float(speed_of_sound_at_altitude(altitude_m))

    # PIANO assumes per ATC rules that aircraft flying below FL 100 are at
    # their low CAS (which is always? 250 kts)
    ATC_FL_CAS_LOW = 100
    cas_kts = (
        schedule.cas_low_kts
        if altitude_m < ATC_FL_CAS_LOW * FL_TO_METERS
        else schedule.cas_high_kts
    )
    return float(cas_to_tas(cas_kts * KNOTS_TO_MPS, altitude_m))


@dataclass(frozen=True)
class _Block:
    """One detail block of a PIANO climb or descent file."""

    header_lines: list[str]
    """Lines between the previous block's marker and this block's."""

    rows: list[list[float]]
    """Data lines that parsed, in file order."""

    malformed: list[str]
    """Data lines that did not parse, in file order."""


def _numeric_row(tokens: list[str], ncols: int) -> list[float] | None:
    """Parse `tokens` as a row of exactly `ncols` numbers, or return None.

    Call this only on a row `_is_data_row` accepts. None then means the row is
    malformed, rather than that the line was never a row.
    """
    if len(tokens) != ncols:
        return None
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


def _split_blocks(lines: list[str], marker: str, ncols: int) -> list[_Block]:
    """Split a PIANO detail file into the blocks introduced by `marker`.

    A block's header lines are those between the previous block's marker and
    this one, which is where PIANO writes block metadata when it writes any.
    Blocks that follow a halted climb have no metadata there, so their header
    lines hold none.

    Raises:
        ValueError: If the file contains no block.
    """
    starts = [i for i, line in enumerate(lines) if marker in line]
    if not starts:
        raise ValueError(f'No "{marker}" block found in PIANO file')

    blocks = []
    for n, start in enumerate(starts):
        header_start = starts[n - 1] + 1 if n > 0 else 0
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        rows = []
        malformed = []
        for line in lines[start + 1 : end]:
            tokens = line.split()
            if not _is_data_row(tokens):
                continue
            row = _numeric_row(tokens, ncols)
            if row is None:
                malformed.append(line)
            else:
                rows.append(row)
        blocks.append(_Block(lines[header_start:start], rows, malformed))
    return blocks


def _reject_malformed(label: str, block: _Block, ncols: int) -> None:
    """Refuse a block holding a data line that did not parse.

    Dropping the line would lose more than one row. `_deltas` differences the
    rows that survive, so the row after the gap reports the fuel burn of two
    steps over the time of two steps. Its own cumulative columns stay right
    and neither cross-check here can see the gap, so the result looks sound.

    PIANO writes fixed-width columns, so the usual cause is a value that
    outgrew its field and ran into its neighbour, merging both into one token.

    Raises:
        ValueError: If the block holds any malformed data line.
    """
    if not block.malformed:
        return
    raise ValueError(
        f'{label}: {len(block.malformed)} line(s) start with a number but do '
        f'not hold {ncols} numbers, and dropping a row would silently average '
        'the fuel flow of the row after it. First such line: '
        f'"{block.malformed[0].strip()}"'
    )


def _deltas(values: list[float]) -> list[float]:
    """Per-row differences of a cumulative column.

    The first row takes the forward difference, so it gets the same step as
    the second row.
    """
    if len(values) < 2:
        return [0.0] * len(values)
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    return [diffs[0]] + diffs


def _cross_check_tas(
    label: str,
    tas: list[float],
    d_distance: list[float],
    d_time: list[float],
    rocd: list[float],
) -> None:
    """Compare derived TAS against the block's own distance and time.

    The speed schedule and the tabulated trajectory are independent, so a
    large disagreement means one of the two was misread. Warns only: PIANO
    tabulates at coarse altitude steps, so some deviation is expected.
    """
    deviations = []
    for i, dt in enumerate(d_time):
        if dt == 0 or tas[i] == 0:
            continue
        ground_speed = d_distance[i] / dt
        deviations.append(abs(math.hypot(ground_speed, rocd[i]) - tas[i]) / tas[i])

    if not deviations:
        logger.warning('%s: no usable rows for the TAS cross-check', label)
        return

    median = sorted(deviations)[len(deviations) // 2]
    if median > _TAS_CROSS_CHECK_TOLERANCE:
        logger.warning(
            '%s: median TAS deviation of %.1f%% between the speed schedule and '
            'the tabulated distance and time',
            label,
            100 * median,
        )


def _cross_check_fuel_burn(label: str, burn_kg: float, header_lines: list[str]) -> None:
    """Compare a block's final cumulative burn against its header total."""
    match = _find(_FUEL_BURN_RE, header_lines)
    if match is None:
        return
    total = float(match.group(1)) * POUNDS_TO_KG
    if total == 0:
        return
    if abs(burn_kg - total) > _FUEL_BURN_CROSS_CHECK_TOLERANCE * total:
        logger.warning(
            '%s: final cumulative burn of %.1f kg does not match the header '
            'total of %.1f kg',
            label,
            burn_kg,
            total,
        )

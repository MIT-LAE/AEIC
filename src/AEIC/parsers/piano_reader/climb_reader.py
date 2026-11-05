"""Reader for the PIANO climb details export."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import logging
import re
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from AEIC.parsers.piano_reader.common import (
    _ATC_FL_CAS_LOW,
    _Block,
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
from AEIC.parsers.piano_reader.rows import PianoClimbRow
from AEIC.performance.types import SpeedData, TableInput
from AEIC.units import (
    FEET_TO_METERS,
    FL_TO_METERS,
    FPM_TO_MPS,
    KNOTS_TO_MPS,
    METERS_TO_FEET,
    NAUTICAL_MILES_TO_METERS,
    POUNDS_FORCE_TO_NEWTONS,
    POUNDS_TO_KG,
)
from AEIC.utils.standard_atmosphere import crossover_altitude

if TYPE_CHECKING:
    from collections.abc import Callable

    from AEIC.parsers.piano_reader.piano_data import PianoOverrides

logger = logging.getLogger(__name__)


class ClimbTableRow(NamedTuple):
    """A row of the output climb table, in SI units except for the flight level.
    The field order defines the order of the climb table columns.
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
    drag: float


CLIMB_COLS = list(ClimbTableRow._fields)

# If user supplied masses and PIANO header masses are both present
# we check that they are consistent up to 0.1%
_CLIMB_MASS_CROSS_CHECK_TOLERANCE = 0.001

# If user supplied schedule and PIANO header schedule are both present
# we check that they are consistent up to 0.1%
_CLIMB_SCHEDULE_CROSS_CHECK_TOLERANCE = 0.001

# The crossover a climb header states must match the standard atmosphere
# crossover of its CAS and Mach up to 0.1%
_CLIMB_CROSSOVER_CROSS_CHECK_TOLERANCE = 0.001

# The line of column labels that opens a climb table, and the label of the
# NOx column some exports add to it.
_CLIMB_LABELS_RE = re.compile(r'^\s*Alt\.')
_NOX_LABEL_RE = re.compile(r'^NOx', re.IGNORECASE)

_CLIMB_MASS_RE = re.compile(r'Initial mass\s+([\d.]+)')
_DELTA_ISA_RE = re.compile(r'Delta-ISA\s+([+-]?[\d.]+)')
_HALTED_RE = re.compile(r'halted at\s+([\d.]+)feet')

# A halted block whose last row is more than this below the halt altitude
# gets an extra row extrapolated to the halt altitude [feet]
_HALT_GAP_MIN_FT = 10.0

# Number of altitude points for the integration of time and distance from the
# last row to the halt altitude
_HALT_GAP_INTEGRATION_POINTS = 101


# ---------------------------------------------------------------------------
# Climb
# ---------------------------------------------------------------------------


def _parse_climb(
    path: str, overrides: PianoOverrides, extrapolate_to_halt: bool = True
) -> tuple[SpeedData, int, int, TableInput]:
    """Parse a PIANO climb details export.

    If `extrapolate_to_halt` is set, a block whose climb halted more than
    `_HALT_GAP_MIN_FT` above its last row gets an extra row at the halt
    altitude. See `_extrapolate_to_halt`.

    Returns the climb speed schedule, the ISA offset, the highest altitude
    reached by any block and the climb table. The highest altitude includes
    any extrapolated row at a halt altitude.
    """
    lines = _read_lines(path)
    nox_col = _nox_column(lines)
    row_cols = len(PianoClimbRow.columns(nox_col))
    blocks = _split_blocks(
        lines, 'Climb details', partial(PianoClimbRow.from_tokens, nox_col=nox_col)
    )

    isa_offset = _parse_isa_offset(lines)
    masses = _climb_masses(blocks, overrides)
    schedule = _climb_schedule(blocks, overrides)

    data: list[ClimbTableRow] = []
    maximum_altitude_ft = 0.0
    for n, (block, mass) in enumerate(zip(blocks, masses, strict=True)):
        label = f'Climb block {n + 1}'
        _reject_malformed(label, block, row_cols)

        rows = block.rows
        if not rows:
            logger.warning('%s: no data rows', label)
            continue

        # Cross-check the rows PIANO wrote, before any extrapolated row.
        tas, rocd, d_time, d_distance, fuel_flow = _derived_columns(rows, schedule)
        _cross_check_tas(label, tas, d_distance, d_time, rocd)
        _cross_check_fuel_burn(
            label, rows[-1].burn_lb * POUNDS_TO_KG, block.header_lines
        )

        halt_ft = _halt_altitude_ft(block)
        if halt_ft is not None and halt_ft - rows[-1].altitude_ft > _HALT_GAP_MIN_FT:
            if not extrapolate_to_halt:
                logger.info(
                    '%s: PIANO halted the climb at %.0f feet, %.0f feet above '
                    'the last row, so the block stops at its last row',
                    label,
                    halt_ft,
                    halt_ft - rows[-1].altitude_ft,
                )
            else:
                row = _extrapolate_to_halt(label, rows, halt_ft, schedule)
                if row is not None:
                    logger.info(
                        '%s: PIANO halted the climb at %.0f feet, %.0f feet '
                        'above the last row, so a row is extrapolated to the '
                        'halt altitude',
                        label,
                        halt_ft,
                        halt_ft - rows[-1].altitude_ft,
                    )
                    rows = [*rows, row]
                    tas, rocd, d_time, d_distance, fuel_flow = _derived_columns(
                        rows, schedule
                    )

        maximum_altitude_ft = max(
            maximum_altitude_ft, max(row.altitude_ft for row in rows)
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
                ClimbTableRow(
                    fl=row.altitude_ft / 100,
                    mass=mass,
                    tas=tas[i],
                    rocd=rocd[i],
                    fuel_flow=row_fuel_flow,
                    time=row.time_s,
                    distance=row.distance_nm * NAUTICAL_MILES_TO_METERS,
                    burn=row.burn_lb * POUNDS_TO_KG,
                    fn_per_engine=row.fn_per_engine_lbf * POUNDS_FORCE_TO_NEWTONS,
                    drag=row.drag_lbf * POUNDS_FORCE_TO_NEWTONS,
                )
            )

    table = TableInput(
        cols=CLIMB_COLS,
        data=[list(row) for row in sorted(data, key=lambda r: (r.mass, r.fl))],
    )
    return (
        schedule.to_speed_data(),
        isa_offset,
        int(maximum_altitude_ft),
        table,
    )


def _derived_columns(
    rows: list[PianoClimbRow], schedule: _Schedule
) -> tuple[list[float], list[float], list[float], list[float], list[float | None]]:
    """TAS [m/s], rate of climb [m/s], the per-row time [s] and distance [m]
    steps, and the fuel flow [kg/s] of a climb block.

    The fuel flow of a row is None if it is not defined, see
    `_row_fuel_flows`."""
    tas = [
        _tas_from_schedule(row.altitude_ft * FEET_TO_METERS, schedule) for row in rows
    ]
    rocd = [row.rocd_fpm * FPM_TO_MPS for row in rows]
    d_time = _deltas([row.time_s for row in rows])
    d_distance = _deltas([row.distance_nm * NAUTICAL_MILES_TO_METERS for row in rows])
    fuel_flow = _row_fuel_flows(
        [row.time_s for row in rows], [row.burn_lb * POUNDS_TO_KG for row in rows]
    )
    return tas, rocd, d_time, d_distance, fuel_flow


def _halt_altitude_ft(block: _Block[PianoClimbRow]) -> float | None:
    """Altitude at which PIANO halted a block's climb [feet], or None if the
    climb reached its target. PIANO writes the halt message just before the
    table of the block it applies to, so it is in that block's header lines."""
    match = _find(_HALTED_RE, block.header_lines)
    return None if match is None else float(match.group(1).rstrip('.'))


def _extrapolate_to_halt(
    label: str, rows: list[PianoClimbRow], halt_ft: float, schedule: _Schedule
) -> PianoClimbRow | None:
    """Extrapolate a halted climb block to its halt altitude.

    PIANO halts a climb when its rate of climb gets too low, but tabulated values
    can stop at a lower altitude. The halt altitude is one the climb reached, so
    an extrapolated row there extends the table without going past the valid data.

    The fit uses the rows above the crossover, where the climb is at constant
    Mach:

    - Rate of climb and net thrust are linear through the last two rows.
    - Fuel flow is linear through the average fuel flows of the last two
      intervals, each at the midpoint altitude of its interval.
    - Drag is quadratic through the last three rows, because it passes
      through a minimum near the ceiling.
    - Time is the integral of dh / ROCD from the last row, distance is the
      integral of the ground speed over that time, and burn is the integral
      of the fuel flow over that time.

    Returns the row in the units of the PIANO table, or None if there are fewer
    than three rows above the crossover, fewer than two fuel flows to fit, or if
    the rate of climb, net thrust, drag or fuel flow at the halt altitude is not
    positive.
    """
    above = [row for row in rows if row.altitude_ft > schedule.crossover_ft]
    fuel_flow = _interval_fuel_flows(rows, schedule.crossover_ft)

    # Check we have enough data points to extrapolate
    if len(above) < 3 or len(fuel_flow) < 2:
        logger.warning(
            '%s: too few rows at constant Mach to extrapolate the climb to '
            'the halt altitude of %.0f feet',
            label,
            halt_ft,
        )
        return None

    last = rows[-1]

    # Construct interpolants for all variables
    rocd_fpm = _polynomial_through(
        [(row.altitude_ft, row.rocd_fpm) for row in above[-2:]]
    )
    fn_lbf = _polynomial_through(
        [(row.altitude_ft, row.fn_per_engine_lbf) for row in above[-2:]]
    )
    drag_lbf = _polynomial_through(
        [(row.altitude_ft, row.drag_lbf) for row in above[-3:]]
    )
    ff_lbfps = _polynomial_through(fuel_flow[-2:])

    # Evaluate extrapolation at the halt altitude
    values = {
        'rocd': rocd_fpm(halt_ft),
        'net thrust': fn_lbf(halt_ft),
        'drag': drag_lbf(halt_ft),
        'fuel flow': ff_lbfps(halt_ft),
    }

    # Sanity check: we should not get negative values here
    # and we skip interpolation if we do as these values are
    # non physical
    for name, value in values.items():
        if value <= 0:
            logger.warning(
                '%s: extrapolated %s at the halt altitude of %.0f feet is '
                'negative (%g), skipping extrapolation',
                label,
                name,
                halt_ft,
                value,
            )
            return None

    # Time and distance are cumulative metrics so we need to integrate
    # dt/dh = 1 / ROCD and ground_speed * dt/dh, over some discretization of dh
    altitude_ft = np.linspace(last.altitude_ft, halt_ft, _HALT_GAP_INTEGRATION_POINTS)

    rocd_mps = np.array([rocd_fpm(h) for h in altitude_ft]) * FPM_TO_MPS
    # We need dt/dh in s/ft to integrate with dh in ft
    dt_dh = 1 / (rocd_mps * METERS_TO_FEET)
    # TAS in m/s
    tas = np.array(
        [_tas_from_schedule(h * FEET_TO_METERS, schedule) for h in altitude_ft]
    )
    # Ground speed in m/s
    ground_speed = np.sqrt(tas**2 - rocd_mps**2)

    # Fuel flow in lb/s
    fuel_flow = np.array([ff_lbfps(h) for h in altitude_ft])

    # Integrate time, ground_speed and fuel flow
    d_time = float(np.trapezoid(dt_dh, altitude_ft))
    d_distance_m = float(np.trapezoid(ground_speed * dt_dh, altitude_ft))
    d_burn = float(np.trapezoid(fuel_flow * dt_dh, altitude_ft))

    # Everything needs to be in PIANO raw data units
    # ft, s, nm, lb, lbf, fp/min, lbf
    return PianoClimbRow(
        altitude_ft=halt_ft,
        time_s=last.time_s + d_time,
        distance_nm=last.distance_nm + d_distance_m / NAUTICAL_MILES_TO_METERS,
        burn_lb=last.burn_lb + d_burn,
        fn_per_engine_lbf=values['net thrust'],
        rocd_fpm=values['rocd'],
        drag_lbf=values['drag'],
    )


def _interval_fuel_flows(
    rows: list[PianoClimbRow], crossover_ft: float
) -> list[tuple[float, float]]:
    """(Altitude [feet], fuel flow [lb/s]) of each climb interval above the
    crossover, in row order.

    The average fuel flow of an interval is the fuel flow at its midpoint altitude.
    The interval that spans the crossover is skipped, because part of it is at
    constant CAS. Skip intervals that span no time (undefined fuel flow).
    """
    return [
        (
            (below.altitude_ft + row.altitude_ft) / 2,
            (row.burn_lb - below.burn_lb) / (row.time_s - below.time_s),
        )
        for below, row in zip(rows, rows[1:])
        if below.altitude_ft > crossover_ft and row.time_s != below.time_s
    ]


def _polynomial_through(
    points: list[tuple[float, float]],
) -> Callable[[float], float]:
    """The polynomial of lowest degree through `points`, as a function.
    Formulated as Lagrange polynomial."""

    def evaluate(x: float) -> float:
        total = 0.0
        for j, (xj, yj) in enumerate(points):
            term = yj
            for m, (xm, _) in enumerate(points):
                if m != j:
                    term *= (x - xm) / (xj - xm)
            total += term
        return total

    return evaluate


def _nox_column(lines: list[str]) -> int | None:
    """Index of the NOx column of the climb table, or None if it has none.

    Some PIANO exports add a NOx column to the climb table meaning the row
    length can vary between 7 and 8 elements. We need to detect this for
    later parsing to work
    """
    for line in lines:
        if _CLIMB_LABELS_RE.match(line) is None:
            continue
        labels = line.split()
        if len(labels) != len(PianoClimbRow.columns()) + 1:
            return None
        for i, label in enumerate(labels):
            if _NOX_LABEL_RE.match(label) is not None:
                return i
        return None
    return None


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


def _header_climb_mass(block: _Block[PianoClimbRow]) -> float | None:
    """Initial mass a climb block's header states [kg], or None if it states
    none. PIANO omits the header of a block whose climb halted."""
    match = _find(_CLIMB_MASS_RE, block.header_lines)
    return None if match is None else float(match.group(1)) * POUNDS_TO_KG


def _climb_masses(
    blocks: list[_Block[PianoClimbRow]], overrides: PianoOverrides
) -> list[float]:
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


def _climb_schedule(
    blocks: list[_Block[PianoClimbRow]], overrides: PianoOverrides
) -> _Schedule:
    """Resolve the climb airspeed schedule.

    The schedule is a property of the file, so the first block that has one supplies it
    for every block. A supplied schedule overrides the header, and its
    crossover is derived from the supplied CAS and Mach. Warns when a supplied
    value contradicts a value the file states.

    Raises:
        ValueError: If no block states a schedule and the overrides do not
            supply one.
    """
    supplied = _supplied_climb_schedule(overrides)

    header = None
    # Look for a header in any climb block
    for block in blocks:
        header = _parse_schedule(block.header_lines, descending=False)
        # Found a header
        if header is not None:
            break

    if header is not None:
        # Ensure crossover alt in header is consistent
        # with header climb CAS and climb Mach
        _cross_check_crossover(header)
        if supplied is None:
            return header
        # User also supplied the schedule, check they are
        # consistent
        _cross_check_supplied_schedule(supplied, header)
        return supplied

    # No header found so user must provide it
    if supplied is None:
        raise ValueError(
            'No climb block states an airspeed schedule, so it must be '
            'supplied explicitly with --climb-cas-low-kts, '
            '--climb-cas-high-kts and --climb-mach'
        )
    return supplied


def _supplied_climb_schedule(overrides: PianoOverrides) -> _Schedule | None:
    """Build the climb airspeed schedule the overrides supply, or None if
    they supply none.

    All three schedule values need to be passed for the override.

    Raises:
        ValueError: If the overrides supply only part of the schedule, or if
            the derived crossover is below FL100.
    """
    cas_low_kts = overrides.climb_cas_low_kts
    cas_high_kts = overrides.climb_cas_high_kts
    mach = overrides.climb_mach

    values = (
        ('--climb-cas-low-kts', cas_low_kts),
        ('--climb-cas-high-kts', cas_high_kts),
        ('--climb-mach', mach),
    )
    missing = [name for name, value in values if value is None]
    # All missing means we rely on the header if it exists
    if len(missing) == len(values):
        return None

    # If at least one is missing (but not all three) error out
    if missing:
        raise ValueError(
            'The climb airspeed schedule must be supplied in full or not at '
            f'all. Missing: {", ".join(missing)}'
        )

    crossover_ft = _derived_crossover_ft(cas_high_kts, mach)

    # Future TAS calculations assume that crossover altitude > FL100
    # so we enforce this here
    if crossover_ft * FEET_TO_METERS < _ATC_FL_CAS_LOW * FL_TO_METERS:
        raise ValueError(
            f'{cas_high_kts:g} kcas and Mach {mach:g} cross over at '
            f'{crossover_ft:.0f} feet, below FL{_ATC_FL_CAS_LOW}'
        )

    return _Schedule(
        cas_low_kts=cas_low_kts,
        cas_high_kts=cas_high_kts,
        mach=mach,
        crossover_ft=crossover_ft,
    )


def _cross_check_supplied_schedule(supplied: _Schedule, header: _Schedule) -> None:
    """Warn for each supplied schedule value that contradicts the header."""
    for name, supplied_value, header_value in (
        ('low CAS [kcas]', supplied.cas_low_kts, header.cas_low_kts),
        ('high CAS [kcas]', supplied.cas_high_kts, header.cas_high_kts),
        ('Mach', supplied.mach, header.mach),
    ):
        if (
            abs(supplied_value - header_value)
            > _CLIMB_SCHEDULE_CROSS_CHECK_TOLERANCE * header_value
        ):
            logger.warning(
                'The supplied climb %s of %g does not match the %g the '
                'airspeed schedule states; using the supplied value',
                name,
                supplied_value,
                header_value,
            )


def _derived_crossover_ft(cas_high_kts: float, mach: float) -> float:
    """Standard atmosphere altitude at which `cas_high_kts` equals `mach`
    [feet]."""
    return float(crossover_altitude(cas_high_kts * KNOTS_TO_MPS, mach)) / FEET_TO_METERS


def _cross_check_crossover(schedule: _Schedule) -> None:
    """Compare the crossover a header states against the one its CAS and
    Mach give in the standard atmosphere."""
    derived_ft = _derived_crossover_ft(schedule.cas_high_kts, schedule.mach)
    if (
        abs(schedule.crossover_ft - derived_ft)
        > _CLIMB_CROSSOVER_CROSS_CHECK_TOLERANCE * derived_ft
    ):
        logger.warning(
            'The climb airspeed schedule states a crossover of %.0f feet, but '
            '%.0f kcas and Mach %.3f cross over at %.0f feet in the standard '
            'atmosphere',
            schedule.crossover_ft,
            schedule.cas_high_kts,
            schedule.mach,
            derived_ft,
        )

"""Reader for PIANO aircraft performance text exports.

PIANO writes three files for an aircraft: a cruise table sweeping mass,
altitude and Mach number, plus climb and descent files holding one detail
block per initial mass. This module parses all three into SI tables ready to
be written to a performance model TOML file.

The files are richer than BADA PTF data, so the tables carry more columns than
the five a performance table requires. Every column PIANO reports is kept
except buffet onset and the NOx, HC and CO emission indices.

Parsing never fabricates rows. A climb block that halts below its target
altitude is kept exactly as PIANO wrote it, because the missing rows mean the
aircraft cannot climb there. A data line that does not parse is a different
matter. Dropping it would corrupt the fuel flow of the row after it, so a
climb or descent file holding one is refused.
"""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Self

from AEIC.performance.types import SpeedData, TableInput
from AEIC.units import (
    FEET_TO_METERS,
    FL_TO_METERS,
    FPM_TO_MPS,
    KNOTS_TO_MPS,
    NAUTICAL_MILES_TO_METERS,
    POUNDS_FORCE_TO_NEWTONS,
    POUNDS_PER_HOUR_TO_KG_PER_S,
    POUNDS_TO_KG,
)
from AEIC.utils.standard_atmosphere import (
    cas_to_tas,
    speed_of_sound_at_altitude,
)

logger = logging.getLogger(__name__)

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
"""Columns of the climb table."""

CRUISE_COLS = [
    'fl',
    'mass',
    'mach',
    'tas',
    'cas',
    'rocd',
    'fuel_flow',
    'drag',
    'mcr_pct',
    'lift_to_drag',
    'sfc',
    'sar',
    'mcl_avail_per_engine',
    'rocd_mcl_fix_mach',
    'rocd_mcl_fix_cas',
]
"""Columns of the cruise table."""

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
"""Columns of the descent table."""

CRUISE_REFERENCE_MACH_COLS = ['fl', 'mass', 'max_sar', 'sar_99', 'max_lim']
"""Columns of the cruise reference Mach table."""

DESCENT_IDLE_THRUST_COLS = ['mass', 'idle_thrust_altitude']
"""Columns of the descent idle thrust table."""

_REFERENCE_MACH_LABELS = {
    'maxSAR': 'max_sar',
    '99%SAR': 'sar_99',
    'maxLim': 'max_lim',
}
"""PIANO cruise reference Mach labels, mapped to column names."""

_TAS_CROSS_CHECK_TOLERANCE = 0.05
"""Relative deviation above which the TAS cross-check warns."""

_FUEL_BURN_CROSS_CHECK_TOLERANCE = 0.01
"""Relative deviation above which the block fuel burn cross-check warns."""

_CLIMB_MASS_CROSS_CHECK_TOLERANCE = 0.01
"""Relative deviation above which a supplied climb mass warns.

Wide enough to absorb a mass restated in rounded pounds, narrow enough to
catch a mistyped digit.
"""

_CLIMB_ROW_COLS = 7
"""Alt., Time, Dist., Burn, FN/eng, R.o.C., Drag."""

_DESCENT_ROW_COLS = 6
"""Alt., Time, Dist., Burn, R.o.D., FN/eng."""

_CRUISE_ROW_COLS = 15
"""Mass, Altitude, Mach and the discriminator, plus eleven swept values."""

_AIRCRAFT_NAME_RE = re.compile(r'Cruise table for\s+(.+?)\s*$')
_CLIMB_MASS_RE = re.compile(r'Initial mass\s+([\d.]+)')
_DESCENT_MASS_RE = re.compile(r'^\s*Mass\s+([\d.]+)')
_FUEL_BURN_RE = re.compile(r'Fuel burn\s+([\d.]+)')
_DELTA_ISA_RE = re.compile(r'Delta-ISA\s+([+-]?[\d.]+)')
_IDLE_THRUST_RE = re.compile(r'Idle thrust below\s+([\d.]+)feet')
_SCHEDULE_LINE_RE = re.compile(r'Airspeed schedule\s+(.+?)\s*$')
_SCHEDULE_MACH_RE = re.compile(r'mach\s+([\d.]+)')
_SCHEDULE_CROSSOVER_RE = re.compile(r'above\s+([\d.]+)feet')
_SCHEDULE_CAS_RE = re.compile(r'([\d.]+)\s*/\s*([\d.]+)\s*kcas')
_HALTED_RE = re.compile(r'halted at\s+([\d.]+)feet')


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

    PIANO opens every data row with a number, and opens no heading, unit line
    or metadata line with one. So the first token tells a row that failed to
    parse apart from the text around the table. Without that test a parse
    failure cannot be reported, because the column headings fail to parse too.
    """
    if not tokens:
        return False
    try:
        float(tokens[0])
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Cruise table
# ---------------------------------------------------------------------------


def _parse_cruise(path: str) -> tuple[str, TableInput, TableInput]:
    """Parse a PIANO cruise table export.

    Returns the aircraft name, the cruise sweep and the reference Mach table.

    Raises:
        ValueError: If the file has no title, or if a data row holds anything
            but the "|" separator or a reference Mach label in its fourth
            column, which means the export does not match the layout read
            here.
    """
    lines = _read_lines(path)

    name_match = _find(_AIRCRAFT_NAME_RE, lines)
    if name_match is None:
        raise ValueError(f'No "Cruise table for" title found in {path}')
    aircraft_name = name_match.group(1).strip()

    # Keyed on (fl, mass, mach), so a labelled row landing exactly on the swept
    # grid can be detected rather than silently duplicating a key.
    rows: dict[tuple[float, float, float], tuple[list[float], bool]] = {}
    labelled: dict[tuple[float, float], dict[str, float]] = {}

    malformed = []
    for line in lines:
        tokens = line.split()
        if not _is_data_row(tokens):
            continue
        values = _cruise_values(tokens)
        if values is None:
            malformed.append(line)
            continue

        # The fourth column holds the "|" separator or a reference Mach label.
        # Any other token means the export does not follow the layout read
        # here, so no column can be trusted to hold what is expected of it.
        discriminator = tokens[3]
        is_swept = discriminator == '|'
        column = _REFERENCE_MACH_LABELS.get(discriminator)
        if not is_swept and column is None:
            labels = ', '.join(f'"{label}"' for label in _REFERENCE_MACH_LABELS)
            raise ValueError(
                f'Cruise row in {path} holds "{discriminator}" where "|" or a '
                f'reference Mach label ({labels}) is expected, so the row does '
                f'not match the PIANO cruise layout: {line.strip()}'
            )

        mass_lb, altitude_ft, mach = values[0], values[1], values[2]
        (
            tas_kts,
            cas_kts,
            drag_lbf,
            mcr_pct,
            lift_to_drag,
            fuel_flow_lbh,
            sfc_lb_h_lbf,
            sar_nm_lb,
            mcl_lbf,
            rocd_mcl_fix_mach_fpm,
            rocd_mcl_fix_cas_fpm,
        ) = values[3:]

        fl = altitude_ft / 100
        mass = mass_lb * POUNDS_TO_KG
        row = [
            fl,
            mass,
            mach,
            tas_kts * KNOTS_TO_MPS,
            cas_kts * KNOTS_TO_MPS,
            # Cruise is level flight, so every phase table carries the five
            # columns a performance table requires.
            0.0,
            fuel_flow_lbh * POUNDS_PER_HOUR_TO_KG_PER_S,
            drag_lbf * POUNDS_FORCE_TO_NEWTONS,
            mcr_pct,
            lift_to_drag,
            # lb/(h.lbf) -> kg/(N.s)
            sfc_lb_h_lbf * POUNDS_PER_HOUR_TO_KG_PER_S / POUNDS_FORCE_TO_NEWTONS,
            # nm/lb -> m/kg
            sar_nm_lb * NAUTICAL_MILES_TO_METERS / POUNDS_TO_KG,
            mcl_lbf * POUNDS_FORCE_TO_NEWTONS,
            # PIANO already reports these signed, so keep its sign.
            rocd_mcl_fix_mach_fpm * FPM_TO_MPS,
            rocd_mcl_fix_cas_fpm * FPM_TO_MPS,
        ]

        if column is not None:
            labelled.setdefault((fl, mass), {})[column] = mach

        _insert_cruise_row(rows, (fl, mass, mach), row, is_swept)

    if malformed:
        logger.warning(
            'Dropping %d cruise line(s) that start with a number but do not '
            'hold %d columns, which thins the interpolation grid. The first '
            'is: "%s"',
            len(malformed),
            _CRUISE_ROW_COLS,
            malformed[0].strip(),
        )

    cruise = TableInput(
        cols=CRUISE_COLS,
        data=[
            row
            for _, (row, _) in sorted(
                rows.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])
            )
        ],
    )
    return aircraft_name, cruise, _reference_mach_table(labelled)


def _cruise_values(tokens: list[str]) -> list[float] | None:
    """Parse the numeric columns of a cruise row, or return None.

    The fourth column holds a separator or a label rather than a number, so it
    is left to the caller.
    """
    if len(tokens) < _CRUISE_ROW_COLS:
        return None
    try:
        return [float(t) for t in tokens[:3] + tokens[4:_CRUISE_ROW_COLS]]
    except ValueError:
        return None


def _insert_cruise_row(
    rows: dict[tuple[float, float, float], tuple[list[float], bool]],
    key: tuple[float, float, float],
    row: list[float],
    is_swept: bool,
) -> None:
    """Add a cruise row, resolving a collision in favour of the swept row.

    A labelled reference Mach can land exactly on the swept grid, and PIANO
    does not always report the same values for the two. Keeping the swept row
    keeps the sweep self-consistent.
    """
    existing = rows.get(key)
    if existing is None:
        rows[key] = (row, is_swept)
        return

    existing_row, existing_is_swept = existing
    differing = [
        CRUISE_COLS[i] for i in range(len(CRUISE_COLS)) if existing_row[i] != row[i]
    ]
    if differing:
        logger.warning(
            'Duplicate cruise row at (fl=%.2f, mass=%.1f kg, mach=%.3f); '
            'keeping the swept row. Columns that disagree: %s',
            *key,
            ', '.join(differing),
        )
    if is_swept and not existing_is_swept:
        rows[key] = (row, is_swept)


def _reference_mach_table(
    labelled: dict[tuple[float, float], dict[str, float]],
) -> TableInput:
    """Build the reference Mach table from the labelled cruise rows.

    Only groups holding every label are emitted, because table data is a list
    of numbers and TOML has no null.
    """
    required = CRUISE_REFERENCE_MACH_COLS[2:]
    data = []
    incomplete = []
    for (fl, mass), machs in labelled.items():
        if all(column in machs for column in required):
            data.append([fl, mass] + [machs[column] for column in required])
        else:
            incomplete.append((fl, mass))

    if incomplete:
        logger.warning(
            'Dropping %d incomplete cruise reference Mach group(s), at '
            '(fl, mass) of %s',
            len(incomplete),
            ', '.join(f'({fl:.2f}, {mass:.1f})' for fl, mass in sorted(incomplete)),
        )

    return TableInput(
        cols=CRUISE_REFERENCE_MACH_COLS,
        data=sorted(data, key=lambda row: (row[1], row[0])),
    )


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
    precisely when the headers do not carry them. The override is
    all-or-nothing though, so a user correcting the later blocks must restate
    the first one too, and a typo there would otherwise pass unseen. Warn
    when a supplied mass contradicts a mass the file does state.

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

    The schedule is a property of the export rather than of one run, so the
    first block that states one supplies it for every block.

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

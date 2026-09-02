"""Reader for the PIANO cruise table export."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import logging
import math
import re

from AEIC.parsers.piano_reader.common import _find, _is_data_row, _read_lines
from AEIC.performance.types import TableInput
from AEIC.units import (
    FPM_TO_MPS,
    KNOTS_TO_MPS,
    NAUTICAL_MILES_TO_METERS,
    POUNDS_FORCE_TO_NEWTONS,
    POUNDS_PER_HOUR_TO_KG_PER_S,
    POUNDS_TO_KG,
)

logger = logging.getLogger(__name__)

#############################################################################
# PIANO cruise file schema
#############################################################################
#
# A cruise file is single table without blocks. PIANO sweeps Mach for
# every (mass, altitude) pair.
#
# 1. Title, stating the aircraft and the engine:
#
#     Cruise table for some_airplane, some_engine
#
# 2. Column headings and unit:
#
#     Mass              lb.
#     Altitude          feet
#     Mach              -
#     |                 discriminator, see below
#     TAS               kts
#     CAS               kts
#     Drag              lbf.
#     MCR.%             percent
#     L/D               -
#     FuelFlow          lb/hr
#     SFC               lb/h/lbf
#     SAR               nm/lb
#     MCL/eng avail.    lbf.
#     RoC@MCL fix.CAS   feet/min
#     RoC@MCL fix.Mach  feet/min
#     Buffet onset      Gs
#     NOx               lb/hr
#     HC                lb/hr
#     CO                lb/hr
#
# `_cruise_values` reads the first `_CRUISE_ROW_COLS` tokens of a row, so
# buffet onset and the three emission indices are dropped. PIANO writes "..."
# in a column it has no value for.
#
# 3. Data rows, one per (mass, altitude, Mach) point:
#
#     93000.   15000.  0.350    |    100.0   200.0      3000.   ...
#     93000.   15000.  0.450 maxSAR  100.0   200.0      3000.   ...
#
# The fourth column discriminates the two kinds of row. "|" marks a swept row.
# "maxSAR", "99%SAR" and "maxLim" mark a reference Mach of the (mass,
# altitude) pair the sweep just covered.
#
# A line that starts with a number but holds too few entries is dropped with a
# warning. The rationale is that the cruise table is an interpolation grid, so
# a lost row thins the grid but leaves its neighbours.
#############################################################################

# Cruise table columns
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

# Cruise reference Mach table columns
CRUISE_REFERENCE_MACH_COLS = ['fl', 'mass', 'max_sar', 'sar_99', 'max_lim']

# PIANO cruise reference Mach specific points mapped to their column name
_REFERENCE_MACH_LABELS = {
    'maxSAR': 'max_sar',
    '99%SAR': 'sar_99',
    'maxLim': 'max_lim',
}

# (Mass, Altitude, Mach) and the discriminator, plus eleven swept values.
_CRUISE_ROW_COLS = 15

_AIRCRAFT_NAME_RE = re.compile(r'Cruise table for\s+(.+?)\s*$')


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
    # Labelled rows are the rows for the specific operating points saved by PIANO
    # A dict: (fl, mass) -> another dict keyed (row_type) -> mach
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
        # If the discriminator is another symbol then raise as this not the
        # expected schema.
        discriminator = tokens[3]
        is_swept = discriminator == '|'
        row_type = _REFERENCE_MACH_LABELS.get(discriminator)
        if not is_swept and row_type is None:
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
            # Cruise is level flight so rate of climb/descend is 0.
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

        if row_type is not None:
            # Create the inner dict if it does not exist yet
            # and store the mapping row_type to corresponding mach
            labelled.setdefault((fl, mass), {})[row_type] = mach

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
    """Add a cruise row, resolving a collision in favour of the labelled row.

    A labelled reference Mach can land exactly on the swept grid, and PIANO
    does not always report the same values for the two. The labelled row is the
    operating point PIANO solved for, while the swept row only shares its key
    because the export rounds Mach, so the labelled row wins.
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
            'keeping the labelled row. Columns that disagree: %s',
            *key,
            ', '.join(differing),
        )
    if existing_is_swept and not is_swept:
        rows[key] = (row, is_swept)


def _reference_mach_table(
    labelled: dict[tuple[float, float], dict[str, float]],
) -> TableInput:
    """Build the reference Mach table from the labelled cruise rows.

    Every (flight level, mass) group is emitted. A label the file does not
    hold for a group is written as NaN, so a group keeps the reference Machs
    PIANO did report.
    """
    required = CRUISE_REFERENCE_MACH_COLS[2:]
    data = []
    incomplete = []
    for (fl, mass), machs in labelled.items():
        data.append([fl, mass] + [machs.get(column, math.nan) for column in required])
        missing = [column for column in required if column not in machs]
        if missing:
            incomplete.append((fl, mass, missing))

    if incomplete:
        logger.warning(
            'Setting a NaN Mach in %d incomplete cruise reference Mach '
            'group(s), at (fl, mass) of %s',
            len(incomplete),
            ', '.join(
                f'({fl:.2f}, {mass:.1f}): {", ".join(missing)}'
                for fl, mass, missing in sorted(incomplete)
            ),
        )

    return TableInput(
        cols=CRUISE_REFERENCE_MACH_COLS,
        data=sorted(data, key=lambda row: (row[1], row[0])),
    )

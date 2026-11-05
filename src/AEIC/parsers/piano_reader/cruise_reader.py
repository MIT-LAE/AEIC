"""Reader for the PIANO cruise table export."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import logging
import math
import re

from AEIC.parsers.piano_reader.common import _find, _is_data_row, _read_lines
from AEIC.parsers.piano_reader.rows import CruiseRow, UnexpectedDiscriminatorError
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

# See PIANO parser doc for expected schema
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
        try:
            # ValidationError means that the row is malformed
            # and CruiseRow.from_tokens(tokens) returns None.
            # This is fine and we skip the row.
            # But if a there is an unexpected discriminator it means
            # that the file itself does not use the convention we expect
            # so raise ValueError instead.
            piano_row = CruiseRow.from_tokens(tokens)
        except UnexpectedDiscriminatorError as error:
            raise ValueError(f'Cruise row in {path} {error}: {line.strip()}') from error
        if piano_row is None:
            malformed.append(line)
            continue

        fl = piano_row.altitude_ft / 100
        mass = piano_row.mass_lb * POUNDS_TO_KG
        mach = piano_row.mach
        row = [
            fl,
            mass,
            mach,
            piano_row.tas_kts * KNOTS_TO_MPS,
            piano_row.cas_kts * KNOTS_TO_MPS,
            # Cruise is level flight so rate of climb/descend is 0.
            0.0,
            piano_row.fuel_flow_lb_per_h * POUNDS_PER_HOUR_TO_KG_PER_S,
            piano_row.drag_lbf * POUNDS_FORCE_TO_NEWTONS,
            piano_row.mcr_pct,
            piano_row.lift_to_drag,
            # lb/(h.lbf) -> kg/(N.s)
            piano_row.sfc_lb_per_h_per_lbf
            * POUNDS_PER_HOUR_TO_KG_PER_S
            / POUNDS_FORCE_TO_NEWTONS,
            # nm/lb -> m/kg
            piano_row.sar_nm_per_lb * NAUTICAL_MILES_TO_METERS / POUNDS_TO_KG,
            piano_row.mcl_avail_per_engine_lbf * POUNDS_FORCE_TO_NEWTONS,
            # PIANO already reports these signed, so keep its sign.
            piano_row.rocd_mcl_fix_mach_fpm * FPM_TO_MPS,
            piano_row.rocd_mcl_fix_cas_fpm * FPM_TO_MPS,
        ]

        row_type = piano_row.reference_mach_column
        if row_type is not None:
            # Create the inner dict if it does not exist yet
            # and store the mapping row_type to corresponding mach
            labelled.setdefault((fl, mass), {})[row_type] = mach

        _insert_cruise_row(rows, (fl, mass, mach), row, piano_row.is_swept)

    if malformed:
        logger.warning(
            'Dropping %d cruise line(s) that open a data row but do not hold '
            '%d numeric columns, which thins the interpolation grid. The '
            'first is: "%s"',
            len(malformed),
            len(CruiseRow.columns()),
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


def _insert_cruise_row(
    rows: dict[tuple[float, float, float], tuple[list[float], bool]],
    key: tuple[float, float, float],
    row: list[float],
    is_swept: bool,
    verbose: bool = False,
) -> None:
    """Add a cruise row, resolving a collision in favour of the labelled row.

    A labelled reference Mach can land exactly on the swept grid, and PIANO
    does not always report the same values for the two. The labelled row is the
    operating point PIANO solved for, while the swept row only shares its key
    because the export rounds Mach, so the labelled row is kept over the sweep.
    """
    existing = rows.get(key)
    if existing is None:
        rows[key] = (row, is_swept)
        return

    existing_row, existing_is_swept = existing

    # Log the rows where the labeled point and swept point differ
    # Off by default because it produces a lot of noise
    if verbose:
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

    # Replace the swept point by the labeled point
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

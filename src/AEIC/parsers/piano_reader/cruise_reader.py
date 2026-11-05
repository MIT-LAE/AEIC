"""Reader for the PIANO cruise table export."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import logging
import math
import re
from typing import NamedTuple

from AEIC.parsers.piano_reader.common import _find, _is_data_row, _read_lines
from AEIC.parsers.piano_reader.rows import (
    _REFERENCE_MACH_LABELS,
    PianoCruiseRow,
    UnexpectedDiscriminatorError,
)
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


class CruiseTableRow(NamedTuple):
    """A row of the output cruise table, in SI units except for the flight level.
    The field order defines the order of the cruise table columns.
    """

    fl: float
    mass: float
    mach: float
    tas: float
    cas: float
    rocd: float
    fuel_flow: float
    drag: float
    mcr_pct: float
    lift_to_drag: float
    sfc: float
    sar: float
    mcl_avail_per_engine: float
    rocd_mcl_fix_mach: float
    rocd_mcl_fix_cas: float


class CruiseReferenceMachTableRow(NamedTuple):
    """A row of the output cruise reference Mach table.

    The fields after `mass` are the values of `_REFERENCE_MACH_LABELS`.
    """

    fl: float
    mass: float
    max_sar: float
    sar_99: float
    max_lim: float


CRUISE_COLS = list(CruiseTableRow._fields)
CRUISE_REFERENCE_MACH_COLS = list(CruiseReferenceMachTableRow._fields)

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
    rows: dict[tuple[float, float, float], tuple[CruiseTableRow, bool]] = {}
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
            # and PianoCruiseRow.from_tokens(tokens) returns None.
            # This is fine and we skip the row.
            # But if a there is an unexpected discriminator it means
            # that the file itself does not use the convention we expect
            # so raise ValueError instead.
            piano_row = PianoCruiseRow.from_tokens(tokens)
        except UnexpectedDiscriminatorError as error:
            raise ValueError(f'Cruise row in {path} {error}: {line.strip()}') from error
        if piano_row is None:
            malformed.append(line)
            continue

        fl = piano_row.altitude_ft / 100
        mass = piano_row.mass_lb * POUNDS_TO_KG
        mach = piano_row.mach
        row = CruiseTableRow(
            fl=fl,
            mass=mass,
            mach=mach,
            tas=piano_row.tas_kts * KNOTS_TO_MPS,
            cas=piano_row.cas_kts * KNOTS_TO_MPS,
            # Cruise is level flight so rate of climb/descend is 0.
            rocd=0.0,
            fuel_flow=piano_row.fuel_flow_lb_per_h * POUNDS_PER_HOUR_TO_KG_PER_S,
            drag=piano_row.drag_lbf * POUNDS_FORCE_TO_NEWTONS,
            mcr_pct=piano_row.mcr_pct,
            lift_to_drag=piano_row.lift_to_drag,
            # lb/(h.lbf) -> kg/(N.s)
            sfc=piano_row.sfc_lb_per_h_per_lbf
            * POUNDS_PER_HOUR_TO_KG_PER_S
            / POUNDS_FORCE_TO_NEWTONS,
            # nm/lb -> m/kg
            sar=piano_row.sar_nm_per_lb * NAUTICAL_MILES_TO_METERS / POUNDS_TO_KG,
            mcl_avail_per_engine=piano_row.mcl_avail_per_engine_lbf
            * POUNDS_FORCE_TO_NEWTONS,
            # PIANO already reports these signed, so keep its sign.
            rocd_mcl_fix_mach=piano_row.rocd_mcl_fix_mach_fpm * FPM_TO_MPS,
            rocd_mcl_fix_cas=piano_row.rocd_mcl_fix_cas_fpm * FPM_TO_MPS,
        )

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
            len(PianoCruiseRow.columns()),
            malformed[0].strip(),
        )

    cruise = TableInput(
        cols=CRUISE_COLS,
        data=[
            list(row)
            for row in sorted(
                (row for row, _ in rows.values()),
                key=lambda r: (r.mass, r.fl, r.mach),
            )
        ],
    )
    return aircraft_name, cruise, _reference_mach_table(labelled)


def _insert_cruise_row(
    rows: dict[tuple[float, float, float], tuple[CruiseTableRow, bool]],
    key: tuple[float, float, float],
    row: CruiseTableRow,
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
            column
            for column, old, new in zip(CRUISE_COLS, existing_row, row, strict=True)
            if old != new
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
    required = _REFERENCE_MACH_LABELS.values()
    data: list[CruiseReferenceMachTableRow] = []
    incomplete = []
    for (fl, mass), machs in labelled.items():
        data.append(
            CruiseReferenceMachTableRow(
                fl=fl,
                mass=mass,
                **{column: machs.get(column, math.nan) for column in required},
            )
        )
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
        data=[list(row) for row in sorted(data, key=lambda r: (r.mass, r.fl))],
    )

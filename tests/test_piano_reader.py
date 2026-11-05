"""Tests for the PIANO performance export reader.

The fixtures in `tests/data/performance/piano` are anonymized dummy data.
Their numbers are internally inconsistent on purpose: the four climb blocks
repeat the same rows truncated at different altitudes, the `Time` column
repeats values so most rows span no time, and the descent blocks' final
cumulative `Burn` disagrees with their header total. Assert structure and
units here, never physical plausibility.
"""

import logging
import math
from pathlib import Path
from typing import ClassVar

import pytest

from AEIC.config import config
from AEIC.parsers.piano_reader import PianoData, PianoOverrides
from AEIC.parsers.piano_reader.climb_reader import (
    CLIMB_COLS,
    _extrapolate_to_halt,
)
from AEIC.parsers.piano_reader.common import _Schedule
from AEIC.parsers.piano_reader.cruise_reader import (
    CRUISE_COLS,
    CRUISE_REFERENCE_MACH_COLS,
)
from AEIC.parsers.piano_reader.descent_reader import (
    DESCENT_COLS,
    DESCENT_IDLE_THRUST_COLS,
)
from AEIC.parsers.piano_reader.rows import (
    PianoClimbRow,
    PianoCruiseRow,
    PianoDescentRow,
    UnexpectedDiscriminatorError,
    _PianoRow,
)
from AEIC.units import (
    FEET_TO_METERS,
    FPM_TO_MPS,
    KNOTS_TO_MPS,
    NAUTICAL_MILES_TO_METERS,
    POUNDS_FORCE_TO_NEWTONS,
    POUNDS_PER_HOUR_TO_KG_PER_S,
    POUNDS_TO_KG,
)
from AEIC.utils.standard_atmosphere import speed_of_sound_at_altitude

PIANO_CRUISE_FILE = 'performance/piano/cruise.txt'
PIANO_CLIMB_FILE = 'performance/piano/climb.txt'
PIANO_DESCENT_FILE = 'performance/piano/descent.txt'

CLIMB_MASSES_KG = [68039.0, 50000.0, 46000.0, 42000.0]
"""Climb block masses, shared with the `piano_data` fixture so that the data
it holds and the data `load()` returns are the same."""

DESCENT_MASSES_LB = [100000.0, 100001.0, 100002.0, 100003.0]


def load(**overrides) -> PianoData:
    """Re-parse the exports behind `piano_data`, with overrides applied.

    Use `piano_data` to read the parsed data. Use this for the tests that
    apply an override, and for the tests that assert on warnings: the fixture
    caches its parse, so it emits its log records once per session only.
    """
    return PianoData.load(
        str(config.file_location(PIANO_CRUISE_FILE)),
        str(config.file_location(PIANO_CLIMB_FILE)),
        str(config.file_location(PIANO_DESCENT_FILE)),
        overrides=PianoOverrides(**{'climb_masses_kg': CLIMB_MASSES_KG, **overrides}),
    )


def test_descent_block_masses_come_from_the_headers(piano_data):
    """Each of the four descent blocks states its mass in its header, in lb.
    The climb masses come from the overrides, so they are not checked here."""
    assert sorted(set(piano_data.descent.column('mass'))) == [
        pytest.approx(lb * POUNDS_TO_KG) for lb in DESCENT_MASSES_LB
    ]


def test_climb_mass_count_mismatch_names_both_counts():
    with pytest.raises(ValueError, match=r'Got 2 climb mass\(es\) for 4 climb'):
        PianoData.load(
            str(config.file_location(PIANO_CRUISE_FILE)),
            str(config.file_location(PIANO_CLIMB_FILE)),
            str(config.file_location(PIANO_DESCENT_FILE)),
            overrides=PianoOverrides(climb_masses_kg=[54000.0, 50000.0]),
        )


def test_climb_masses_required_when_blocks_have_no_header():
    """Only the first climb block states an initial mass, so all four must be
    supplied together."""
    with pytest.raises(ValueError, match='Climb block 2 has no "Initial mass"'):
        PianoData.load(
            str(config.file_location(PIANO_CRUISE_FILE)),
            str(config.file_location(PIANO_CLIMB_FILE)),
            str(config.file_location(PIANO_DESCENT_FILE)),
        )


def test_supplied_climb_mass_contradicting_the_header_warns(caplog):
    """The masses are all-or-nothing, so a user correcting the headerless
    blocks must restate the first block's own mass. A typo there is caught."""
    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        load(climb_masses_kg=[54000.0, 50000.0, 46000.0, 42000.0])
    assert any(
        'Climb block 1: the supplied initial mass of 54000.0 kg does not match '
        'the 68038.9 kg its header states' in r.getMessage()
        for r in caplog.records
    )


def test_climb_speed_schedule(piano_data):
    """250./ 280.kcas/ mach 0.750 above 30523.feet."""
    assert piano_data.climb_speeds.cas_low == pytest.approx(250 * KNOTS_TO_MPS)
    assert piano_data.climb_speeds.cas_high == pytest.approx(280 * KNOTS_TO_MPS)
    assert piano_data.climb_speeds.mach == pytest.approx(0.750)
    assert piano_data.climb_speeds.crossover_altitude_m == pytest.approx(
        30523 * FEET_TO_METERS
    )


def test_descent_speed_schedule(piano_data):
    """`mach 0.750 above 30000.feet/ 280./ 250.kcas`, written high to low."""
    assert piano_data.descent_speeds.cas_low == pytest.approx(250 * KNOTS_TO_MPS)
    assert piano_data.descent_speeds.cas_high == pytest.approx(280 * KNOTS_TO_MPS)
    assert piano_data.descent_speeds.mach == pytest.approx(0.750)
    assert piano_data.descent_speeds.crossover_altitude_m == pytest.approx(
        30000 * FEET_TO_METERS
    )


def test_climb_schedule_override_derives_crossover():
    """275 kcas and Mach 0.740 cross over at 30680 feet."""
    overridden = load(
        climb_cas_low_kts=240.0, climb_cas_high_kts=275.0, climb_mach=0.74
    )
    assert overridden.climb_speeds.cas_low == pytest.approx(240 * KNOTS_TO_MPS)
    assert overridden.climb_speeds.cas_high == pytest.approx(275 * KNOTS_TO_MPS)
    assert overridden.climb_speeds.mach == pytest.approx(0.74)
    assert overridden.climb_speeds.crossover_altitude_m == pytest.approx(
        30680 * FEET_TO_METERS, abs=2 * FEET_TO_METERS
    )


def test_supplied_climb_schedule_contradicting_the_header_warns(caplog):
    """Each supplied value that differs from the header warns. The high CAS
    matches the header, so it does not."""
    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        load(climb_cas_low_kts=240.0, climb_cas_high_kts=280.0, climb_mach=0.72)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        'supplied climb low CAS [kcas] of 240 does not match the 250' in m
        for m in messages
    )
    assert any(
        'supplied climb Mach of 0.72 does not match the 0.75' in m for m in messages
    )
    assert not any('high CAS' in m for m in messages)


def test_partial_climb_schedule_override_raises():
    with pytest.raises(
        ValueError, match='Missing: --climb-cas-low-kts, --climb-cas-high-kts'
    ):
        load(climb_mach=0.72)


def test_climb_crossover_below_fl100_raises():
    """330 kcas reaches Mach 0.550 below FL100, so the schedule would never fly
    its high CAS."""
    with pytest.raises(ValueError, match='below FL100'):
        load(climb_cas_low_kts=250.0, climb_cas_high_kts=330.0, climb_mach=0.55)


def _climb_without_schedule(tmp_path: Path) -> Path:
    """The climb fixture with its one airspeed schedule line removed."""
    lines = config.file_location(PIANO_CLIMB_FILE).read_text().splitlines(keepends=True)
    climb_file = tmp_path / 'climb.txt'
    kept = [line for line in lines if 'Airspeed schedule' not in line]
    climb_file.write_text(''.join(kept))
    return climb_file


def test_climb_schedule_required_without_header(tmp_path):
    with pytest.raises(ValueError, match='No climb block states an airspeed'):
        _load_climb(_climb_without_schedule(tmp_path))


def test_climb_schedule_supplied_without_header(tmp_path):
    piano = _load_climb(
        _climb_without_schedule(tmp_path),
        climb_cas_low_kts=250.0,
        climb_cas_high_kts=280.0,
        climb_mach=0.75,
    )
    assert piano.climb_speeds.crossover_altitude_m == pytest.approx(
        30523 * FEET_TO_METERS, abs=2 * FEET_TO_METERS
    )


def test_inconsistent_climb_crossover_warns(tmp_path, caplog):
    """280 kcas and Mach 0.750 cross over at 30523 feet, not 30000."""
    text = (
        config.file_location(PIANO_CLIMB_FILE)
        .read_text()
        .replace('above 30523.feet', 'above 30000.feet')
    )
    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(text)

    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        PianoData.load(
            str(config.file_location(PIANO_CRUISE_FILE)),
            str(climb_file),
            str(config.file_location(PIANO_DESCENT_FILE)),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )
    assert any(
        'states a crossover of 30000 feet' in r.getMessage()
        and 'cross over at 30523 feet' in r.getMessage()
        for r in caplog.records
    )


def test_consistent_climb_crossover_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        load()
    assert not any('crossover' in r.getMessage() for r in caplog.records)


def test_labelled_mach_on_the_swept_grid_keeps_one_row(piano_data):
    """A labelled reference Mach can land exactly on the swept grid. The
    fixture labels maxSAR at 0.450, which the sweep also visits."""
    matching = [
        row
        for row in piano_data.cruise.data
        if row[0] == 150.0
        and row[2] == 0.450
        and row[1] == pytest.approx(93000 * POUNDS_TO_KG)
    ]
    assert len(matching) == 1


def test_unrecognised_cruise_label_raises(tmp_path):
    """An unknown label in the fourth column means the export comes from a
    PIANO layout this parser was not written against. Nothing in the file can
    be trusted after that, so refuse it rather than drop rows."""
    text = (
        config.file_location(PIANO_CRUISE_FILE).read_text().replace('99%SAR', '98%SAR')
    )
    cruise_file = tmp_path / 'cruise.txt'
    cruise_file.write_text(text)

    with pytest.raises(ValueError, match='does not match the expected PIANO cruise'):
        PianoData.load(
            str(cruise_file),
            str(config.file_location(PIANO_CLIMB_FILE)),
            str(config.file_location(PIANO_DESCENT_FILE)),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )


def test_cruise_row_without_the_separator_raises(tmp_path):
    """If the "|" separator is missing, the fourth column holds a number and
    every value after it is read one column early. Refuse the file instead."""
    cruise_file = tmp_path / 'cruise.txt'
    cruise_file.write_text(
        '  Cruise table for some_airplane, some_engine\n'
        # Fifteen numeric columns, so the row still parses once shifted.
        '   93000.  15000.  0.350  100.0  200.0  3000.  40.0  150.0  6000. '
        '  0.7000  0.08000  19000.  1000.  2000.  1.0\n'
    )

    with pytest.raises(ValueError, match='does not match the expected PIANO cruise'):
        PianoData.load(
            str(cruise_file),
            str(config.file_location(PIANO_CLIMB_FILE)),
            str(config.file_location(PIANO_DESCENT_FILE)),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )


def test_cruise_reference_mach_complete_groups(piano_data):
    """The fixture labels all three reference Machs for every (fl, mass)."""
    assert piano_data.cruise_reference_mach.cols == CRUISE_REFERENCE_MACH_COLS
    assert len(piano_data.cruise_reference_mach.data) == 6
    assert piano_data.cruise_reference_mach.data[0] == [
        150.0,
        pytest.approx(93000 * POUNDS_TO_KG),
        0.450,
        0.490,
        0.690,
    ]


def test_cruise_reference_mach_fills_missing_labels_with_nan(tmp_path, caplog):
    """A group the file does not label for every reference Mach is kept, with
    a NaN Mach in the columns PIANO did not report."""
    lines = (
        config.file_location(PIANO_CRUISE_FILE).read_text().splitlines(keepends=True)
    )
    trimmed = [line for line in lines if 'maxLim' not in line]
    cruise_file = tmp_path / 'cruise.txt'
    cruise_file.write_text(''.join(trimmed))

    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        piano = PianoData.load(
            str(cruise_file),
            str(config.file_location(PIANO_CLIMB_FILE)),
            str(config.file_location(PIANO_DESCENT_FILE)),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )

    reference_mach = piano.cruise_reference_mach
    assert len(reference_mach.data) == 6
    assert all(math.isnan(mach) for mach in reference_mach.column('max_lim'))
    assert not any(math.isnan(mach) for mach in reference_mach.column('max_sar'))
    assert not any(math.isnan(mach) for mach in reference_mach.column('sar_99'))
    assert any(
        'incomplete cruise reference Mach group' in r.getMessage()
        for r in caplog.records
    )


def test_zero_time_rows_are_dropped(piano_data, caplog):
    """The fixture's `Time` column repeats values, so the difference of most
    rows spans no time and they have no defined fuel flow. The clock reads
    0 s, then 60 s from 1417 feet, 600 s from 7086 feet and 6000 s from 35431
    feet. A row keeps its fuel flow if the time differs between the rows on
    each side of it, or between the edge row and its neighbour."""
    for mass in CLIMB_MASSES_KG:
        kept = [row for row in piano_data.climb.data if row[1] == mass]
        assert [row[0] for row in kept] == [
            0.0,
            14.17,
            56.69,
            70.86,
            340.14,
            354.31,
        ]

    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        load()
    assert any('spans no time' in r.getMessage() for r in caplog.records)


def _first_block_header(path: str) -> str:
    """The lines of a fixture file up to the first data row of its first
    block."""
    lines = config.file_location(path).read_text().splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if 'details' in line)
    first_row = next(
        i
        for i in range(start + 1, len(lines))
        if lines[i].split() and lines[i].split()[0][0].isdigit()
    )
    return ''.join(lines[:first_row])


def _rows_text(rows: list[list[float]]) -> str:
    return ''.join(' '.join(f'{value:10.1f}' for value in row) + '\n' for row in rows)


def test_climb_fuel_flow_is_the_central_difference_of_burn(tmp_path):
    """The burn is B(t) = 2 t + 0.01 t^2 lb at 60 s steps, so the fuel flow
    is B'(t) = 2 + 0.02 t lb/s. The central difference is exact at the two
    interior rows: 3.2 lb/s at 60 s and 4.4 lb/s at 120 s. The edge rows take
    the one-sided difference: 156 lb over the first 60 s, 300 lb over the
    last 60 s."""
    # Alt., Time, Dist., Burn, FN/eng, R.o.C., Drag
    rows = [
        [0.0, 0.0, 0.0, 0.0, 30000.0, 1000.0, 1000.0],
        [1000.0, 60.0, 1.0, 156.0, 30000.0, 1000.0, 1000.0],
        [2000.0, 120.0, 2.0, 384.0, 30000.0, 1000.0, 1000.0],
        [3000.0, 180.0, 3.0, 684.0, 30000.0, 1000.0, 1000.0],
    ]
    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(_first_block_header(PIANO_CLIMB_FILE) + _rows_text(rows))

    piano = _load_climb(climb_file, climb_masses_kg=CLIMB_MASSES_KG[:1])

    assert piano.climb.column('fuel_flow') == pytest.approx(
        [ff * POUNDS_TO_KG for ff in (2.6, 3.2, 4.4, 5.0)]
    )


def test_descent_fuel_flow_is_the_central_difference_of_burn(tmp_path):
    """The burn is B(t) = 0.4 t + 0.002 t^2 lb at 60 s steps, so the fuel
    flow is B'(t) = 0.4 + 0.004 t lb/s: 0.64 lb/s at 60 s and 0.88 lb/s at
    120 s. The edge rows take the one-sided difference: 31.2 lb over the first
    60 s, 60 lb over the last 60 s. Rows are in flight order, from the top of
    descent down."""
    # Alt., Time, Dist., Burn, R.o.D., FN/eng
    rows = [
        [29000.0, 0.0, 0.0, 0.0, 1000.0, 1500.0],
        [28000.0, 60.0, 4.0, 31.2, 1000.0, 1500.0],
        [27000.0, 120.0, 8.0, 76.8, 1000.0, 1500.0],
        [26000.0, 180.0, 12.0, 136.8, 1000.0, 1500.0],
    ]
    descent_file = tmp_path / 'descent.txt'
    descent_file.write_text(_first_block_header(PIANO_DESCENT_FILE) + _rows_text(rows))

    piano = PianoData.load(
        str(config.file_location(PIANO_CRUISE_FILE)),
        str(config.file_location(PIANO_CLIMB_FILE)),
        str(descent_file),
        overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
    )

    # The table is sorted by flight level, so the bottom row comes first.
    assert piano.descent.column('fuel_flow') == pytest.approx(
        [ff * POUNDS_TO_KG for ff in (1.0, 0.88, 0.64, 0.52)]
    )


def test_cruise_row_units(piano_data):
    """Spot-check every converted column of one cruise row."""
    row = piano_data.cruise.data[0]
    assert dict(zip(CRUISE_COLS, row, strict=True)) == {
        'fl': 150.0,
        'mass': pytest.approx(93000 * POUNDS_TO_KG),
        'mach': 0.350,
        'tas': pytest.approx(100 * KNOTS_TO_MPS),
        'cas': pytest.approx(200 * KNOTS_TO_MPS),
        'rocd': 0.0,
        'fuel_flow': pytest.approx(6000 * POUNDS_PER_HOUR_TO_KG_PER_S),
        'drag': pytest.approx(3000 * POUNDS_FORCE_TO_NEWTONS),
        'mcr_pct': 40.0,
        'lift_to_drag': 150.0,
        'sfc': pytest.approx(
            0.7 * POUNDS_PER_HOUR_TO_KG_PER_S / POUNDS_FORCE_TO_NEWTONS
        ),
        'sar': pytest.approx(0.08 * NAUTICAL_MILES_TO_METERS / POUNDS_TO_KG),
        'mcl_avail_per_engine': pytest.approx(19000 * POUNDS_FORCE_TO_NEWTONS),
        'rocd_mcl_fix_mach': pytest.approx(1000 * FPM_TO_MPS),
        'rocd_mcl_fix_cas': pytest.approx(2000 * FPM_TO_MPS),
    }


def test_climb_row_units(piano_data):
    """The first climb row is at sea level, where CAS and TAS coincide."""
    row = piano_data.climb.data[0]
    assert dict(zip(CLIMB_COLS, row, strict=True)) == {
        'fl': 0.0,
        'mass': 42000.0,
        'tas': pytest.approx(250 * KNOTS_TO_MPS, rel=1e-3),
        'rocd': pytest.approx(2000 * FPM_TO_MPS),
        # 40 lb burned over the 60 s to the next row.
        'fuel_flow': pytest.approx(40 * POUNDS_TO_KG / 60),
        'time': 0.0,
        'distance': 0.0,
        'burn': 0.0,
        'fn_per_engine': pytest.approx(30000 * POUNDS_FORCE_TO_NEWTONS),
        'drag': pytest.approx(1000 * POUNDS_FORCE_TO_NEWTONS),
    }


def test_descent_row_units(piano_data):
    """Descent rows carry no drag column, and PIANO's rate of descent is
    negated."""
    row = piano_data.descent.data[-1]
    assert dict(zip(DESCENT_COLS, row, strict=True)) == {
        'fl': 410.0,
        'mass': pytest.approx(100003 * POUNDS_TO_KG),
        'tas': pytest.approx(
            0.750 * float(speed_of_sound_at_altitude(41000 * FEET_TO_METERS))
        ),
        'rocd': pytest.approx(-1400 * FPM_TO_MPS),
        'fuel_flow': pytest.approx(0.0),
        'time': 0.0,
        'distance': pytest.approx(200 * NAUTICAL_MILES_TO_METERS),
        'burn': pytest.approx(30 * POUNDS_TO_KG),
        'fn_per_engine': pytest.approx(1500 * POUNDS_FORCE_TO_NEWTONS),
    }


def test_non_zero_delta_isa_raises(tmp_path):
    """Deriving TAS from the airspeed schedule assumes a standard
    atmosphere."""
    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(
        config.file_location(PIANO_CLIMB_FILE)
        .read_text()
        .replace('Delta-ISA           +0.', 'Delta-ISA           +10.')
    )

    with pytest.raises(ValueError, match=r'Delta-ISA of \+10 deg C'):
        PianoData.load(
            str(config.file_location(PIANO_CRUISE_FILE)),
            str(climb_file),
            str(config.file_location(PIANO_DESCENT_FILE)),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )


def test_descent_schedule_mismatch_raises(tmp_path):
    """The descent airspeed schedule must be the same across the file."""
    lines = config.file_location(PIANO_DESCENT_FILE).read_text().splitlines(True)
    marker = 'Airspeed schedule'
    second = [n for n, line in enumerate(lines) if marker in line][1]
    lines[second] = lines[second].replace('mach 0.750', 'mach 0.800')

    descent_file = tmp_path / 'descent.txt'
    descent_file.write_text(''.join(lines))

    with pytest.raises(ValueError, match='must be the same across the file'):
        PianoData.load(
            str(config.file_location(PIANO_CRUISE_FILE)),
            str(config.file_location(PIANO_CLIMB_FILE)),
            str(descent_file),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )


def test_aircraft_name_and_maximum_altitude(piano_data):
    assert piano_data.aircraft_name == 'some_airplane, some_engine'
    # The highest altitude actually reached, not the header's request of
    # 41000 feet. Only the first climb block gets there.
    assert piano_data.maximum_altitude_ft == 41100
    assert piano_data.isa_offset == 0


def test_descent_idle_thrust(piano_data):
    assert piano_data.descent_idle_thrust.cols == DESCENT_IDLE_THRUST_COLS
    assert piano_data.descent_idle_thrust.data == [
        [pytest.approx(mass_lb * POUNDS_TO_KG), pytest.approx(30000 * FEET_TO_METERS)]
        for mass_lb in DESCENT_MASSES_LB
    ]


def _merge_first_two_columns(line: str) -> str:
    """Rewrite a data line as PIANO writes it when column 1 overflows.

    PIANO writes fixed-width columns, so a value that outgrows its field runs
    into its neighbour and the two arrive as one token.
    """
    tokens = line.split()
    return '  ' + '  '.join([tokens[0] + tokens[1]] + tokens[2:]) + '\n'


def test_climb_row_with_merged_first_column_raises(tmp_path):
    """A merge of `Alt.` and `Time` leaves a token that is not a number, so
    the row can only be seen as malformed by how it starts. Dropping it would
    give the next row the fuel burn of two altitude steps over one step of
    time, which neither cross-check can see."""
    lines = config.file_location(PIANO_CLIMB_FILE).read_text().splitlines(True)
    merged = [
        _merge_first_two_columns(line) if line.split()[:2] == ['1417.', '60.'] else line
        for line in lines
    ]
    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(''.join(merged))

    with pytest.raises(ValueError, match=r'1 line\(s\) open a data row'):
        PianoData.load(
            str(config.file_location(PIANO_CRUISE_FILE)),
            str(climb_file),
            str(config.file_location(PIANO_DESCENT_FILE)),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )


def _with_extra_column(text: str, label: str) -> str:
    """The climb file with one more column, labelled `label`, on its table.

    Appends the column to the line of column labels, to the line of units and
    to every data row, as PIANO writes an export that also reports NOx.
    """
    lines = []
    for line in text.splitlines(True):
        stripped = line.strip()
        if stripped.startswith('Alt.'):
            lines.append(f'{line.rstrip()}     {label}\n')
        elif stripped.startswith('(feet)'):
            lines.append(f'{line.rstrip()}     (gm.)\n')
        elif stripped[:1].isdigit():
            lines.append(f'{line.rstrip()}      12.\n')
        else:
            lines.append(line)
    return ''.join(lines)


def _load_climb(climb_file: Path, **overrides) -> PianoData:
    """Parse the exports behind `piano_data`, with the climb file replaced and
    overrides applied."""
    return PianoData.load(
        str(config.file_location(PIANO_CRUISE_FILE)),
        str(climb_file),
        str(config.file_location(PIANO_DESCENT_FILE)),
        overrides=PianoOverrides(**{'climb_masses_kg': CLIMB_MASSES_KG, **overrides}),
    )


def test_climb_nox_column_is_dropped(tmp_path, piano_data):
    """Some PIANO climb exports carry an eighth NOx column. The reader has no
    use for it, so it drops the column and reads the rest as before."""
    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(
        _with_extra_column(config.file_location(PIANO_CLIMB_FILE).read_text(), 'NOx')
    )

    parsed = _load_climb(climb_file)

    assert parsed.climb.cols == CLIMB_COLS
    assert parsed.climb.data == piano_data.climb.data
    assert parsed.maximum_altitude_ft == piano_data.maximum_altitude_ft


def test_climb_extra_column_that_is_not_nox_raises(tmp_path):
    """NOx is the only extra column the reader knows how to drop. Any other
    one leaves rows it cannot map onto the expected columns."""
    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(
        _with_extra_column(config.file_location(PIANO_CLIMB_FILE).read_text(), 'HC')
    )

    with pytest.raises(ValueError, match='do not hold 7 numbers'):
        _load_climb(climb_file)


# ---------------------------------------------------------------------------
# Extrapolation of a halted climb to its halt altitude
# ---------------------------------------------------------------------------

SCHEDULE = _Schedule(
    cas_low_kts=250.0, cas_high_kts=280.0, mach=0.75, crossover_ft=30523.0
)
"""280 kcas and Mach 0.750 cross over at 30523 feet."""


def _rocd_fpm(h: float) -> float:
    return 1000.0 - 0.2 * (h - 31000.0)


def _fn_lbf(h: float) -> float:
    return 8000.0 - 0.5 * (h - 31000.0)


def _drag_lbf(h: float) -> float:
    return 10000.0 + 1.0e-4 * (h - 32000.0) ** 2


def _fuel_flow_lbps(h: float) -> float:
    return 2.0 - 1.0e-4 * (h - 31000.0)


def _synthetic_rows(altitudes_ft: list[float]) -> list[PianoClimbRow]:
    """Climb rows in PIANO units on known functions of altitude.

    Each interval takes 100 s, and burns the fuel flow at its midpoint
    altitude over that time. The fuel flow is linear in altitude, so the
    average fuel flow of each interval is the fuel flow at its midpoint. The
    first row is off every trend, so a fit that uses it fails the tests.
    """
    rows = [
        PianoClimbRow(
            altitude_ft=altitudes_ft[0],
            time_s=0.0,
            distance_nm=0.0,
            burn_lb=0.0,
            fn_per_engine_lbf=99999.0,
            rocd_fpm=9999.0,
            drag_lbf=99999.0,
        )
    ]
    for h in altitudes_ft[1:]:
        below = rows[-1]
        midpoint = (below.altitude_ft + h) / 2
        rows.append(
            PianoClimbRow(
                altitude_ft=h,
                time_s=below.time_s + 100.0,
                distance_nm=below.distance_nm + 10.0,
                burn_lb=below.burn_lb + 100.0 * _fuel_flow_lbps(midpoint),
                fn_per_engine_lbf=_fn_lbf(h),
                rocd_fpm=_rocd_fpm(h),
                drag_lbf=_drag_lbf(h),
            )
        )
    return rows


def test_extrapolate_to_halt_follows_the_fitted_forms():
    """Rate of climb, thrust and fuel flow are linear and drag is quadratic,
    so the extrapolation reproduces them. Time is the integral of dh / ROCD,
    and burn is the integral of the fuel flow over that time."""
    rows = _synthetic_rows([30000.0, 31000.0, 32000.0, 33000.0, 34000.0])

    row = _extrapolate_to_halt('block', rows, 35000.0, SCHEDULE)

    assert row is not None
    last = rows[-1]
    assert row.altitude_ft == 35000.0
    assert row.rocd_fpm == pytest.approx(_rocd_fpm(35000.0))
    assert row.fn_per_engine_lbf == pytest.approx(_fn_lbf(35000.0))
    assert row.drag_lbf == pytest.approx(_drag_lbf(35000.0))

    # dh / ROCD for a rate of climb linear from 400 to 200 ft/min over 1000 ft.
    d_time = 60.0 * 1000.0 / (200.0 - 400.0) * math.log(200.0 / 400.0)
    assert row.time_s == pytest.approx(last.time_s + d_time, rel=1e-4)
    # Integral of FF dh / ROCD, with u = h - 31000 feet from 3000 to 4000:
    # (2 - 1e-4 u) / (1000 - 0.2 u) = 5e-4 + 1.5 / (1000 - 0.2 u).
    d_burn = 60.0 * (5e-4 * 1000.0 + 1.5 / 0.2 * math.log(400.0 / 200.0))
    assert row.burn_lb == pytest.approx(last.burn_lb + d_burn, rel=1e-4)

    # The climb is at constant Mach and close to level, so the ground speed is
    # close to the TAS at the middle of the step.
    tas_kts = 0.75 * speed_of_sound_at_altitude(34500.0 * FEET_TO_METERS)
    tas_kts /= KNOTS_TO_MPS
    assert row.distance_nm - last.distance_nm == pytest.approx(
        tas_kts * d_time / 3600, rel=1e-2
    )


def test_extrapolate_to_halt_needs_two_usable_fuel_flows(caplog):
    """Three rows are above the crossover, so rate of climb, thrust and drag
    could be fitted. But the first of their intervals starts below the
    crossover, and the last interval spans no time. Only one interval is left
    for the fuel flow, which is too few to fit."""
    rows = _synthetic_rows([30000.0, 31000.0, 32000.0, 33000.0])
    rows[-1] = rows[-1].model_copy(update={'time_s': rows[-2].time_s})

    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        row = _extrapolate_to_halt('block', rows, 34000.0, SCHEDULE)

    assert row is None
    assert any('too few rows' in r.getMessage() for r in caplog.records)


def test_extrapolate_to_halt_skips_a_negative_rate_of_climb(caplog):
    """The rate of climb reaches zero at 36000 feet, so it is negative at a
    halt altitude above that. The climb is not extrapolated."""
    rows = _synthetic_rows([30000.0, 31000.0, 32000.0, 33000.0, 34000.0])

    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        row = _extrapolate_to_halt('block', rows, 36500.0, SCHEDULE)

    assert row is None
    assert any(
        'extrapolated rocd' in r.getMessage()
        and 'skipping extrapolation' in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Row models
# ---------------------------------------------------------------------------

CRUISE_TOKENS = [str(float(i)) for i in range(15)]
"""Numeric tokens for a cruise row, where each value is its column index."""
CRUISE_TOKENS[3] = '|'


def test_climb_row_columns_follow_the_file_order():
    row = PianoClimbRow.from_tokens(['1', '2', '3', '4', '5', '6', '7'])

    assert row == PianoClimbRow(
        altitude_ft=1.0,
        time_s=2.0,
        distance_nm=3.0,
        burn_lb=4.0,
        fn_per_engine_lbf=5.0,
        rocd_fpm=6.0,
        drag_lbf=7.0,
    )


def test_descent_row_columns_follow_the_file_order():
    """Unlike climb, a descent row has its rate column before its thrust."""
    row = PianoDescentRow.from_tokens(['1', '2', '3', '4', '5', '6'])

    assert row == PianoDescentRow(
        altitude_ft=1.0,
        time_s=2.0,
        distance_nm=3.0,
        burn_lb=4.0,
        rod_fpm=5.0,
        fn_per_engine_lbf=6.0,
    )


def test_cruise_row_columns_follow_the_file_order():
    """PIANO writes L/D before the fuel flow, and mass before altitude."""
    row = PianoCruiseRow.from_tokens(CRUISE_TOKENS)

    assert row == PianoCruiseRow(
        mass_lb=0.0,
        altitude_ft=1.0,
        mach=2.0,
        discriminator='|',
        tas_kts=4.0,
        cas_kts=5.0,
        drag_lbf=6.0,
        mcr_pct=7.0,
        lift_to_drag=8.0,
        fuel_flow_lb_per_h=9.0,
        sfc_lb_per_h_per_lbf=10.0,
        sar_nm_per_lb=11.0,
        mcl_avail_per_engine_lbf=12.0,
        rocd_mcl_fix_mach_fpm=13.0,
        rocd_mcl_fix_cas_fpm=14.0,
    )


@pytest.mark.parametrize('nox_col', [0, 4, 7])
def test_climb_row_reads_the_nox_column_at_any_position(nox_col):
    """The remaining columns keep their order wherever the file puts NOx."""
    tokens = ['1', '2', '3', '4', '5', '6', '7']
    tokens.insert(nox_col, '99')

    row = PianoClimbRow.from_tokens(tokens, nox_col=nox_col)

    assert row is not None
    assert row.nox == 99.0
    assert row == PianoClimbRow.from_tokens(
        ['1', '2', '3', '4', '5', '6', '7']
    ).model_copy(update={'nox': 99.0})


@pytest.mark.parametrize(
    'tokens',
    [
        ['1', '2', '3', '4', '5', '6'],
        ['1', '2', '3', '4', '5', '6', '7', '8'],
        ['1', '2', '3', '4.5.6', '5', '6', '7'],
    ],
)
def test_climb_row_is_malformed_without_seven_numbers(tokens):
    assert PianoClimbRow.from_tokens(tokens) is None


def test_climb_row_is_malformed_if_its_nox_value_is_not_a_number():
    """The reader has no use for the value, but a broken NOx token can mean
    that the columns around it merged."""
    tokens = ['1', '2', '3', '4', '5', '6', '7', '***']

    assert PianoClimbRow.from_tokens(tokens, nox_col=7) is None


def test_cruise_row_ignores_columns_after_those_it_reads():
    """PIANO writes buffet onset and emissions after the columns that the
    reader uses."""
    row = PianoCruiseRow.from_tokens([*CRUISE_TOKENS, 'buffet', 'NOx'])

    assert row == PianoCruiseRow.from_tokens(CRUISE_TOKENS)


def test_cruise_row_with_an_unknown_discriminator_raises():
    tokens = list(CRUISE_TOKENS)
    tokens[3] = '98%SAR'

    with pytest.raises(UnexpectedDiscriminatorError, match='98%SAR'):
        PianoCruiseRow.from_tokens(tokens)


@pytest.mark.parametrize('bad_column', [0, 14])
def test_malformed_cruise_row_does_not_report_its_discriminator(bad_column):
    """A broken number comes first: the reader drops this row, rather than
    refuse the file for the fourth column of a row that it cannot read."""
    tokens = list(CRUISE_TOKENS)
    tokens[3] = '98%SAR'
    tokens[bad_column] = '1.2.3'

    assert PianoCruiseRow.from_tokens(tokens) is None


def test_cruise_row_with_too_few_columns_is_malformed():
    assert PianoCruiseRow.from_tokens(CRUISE_TOKENS[:-1]) is None


@pytest.mark.parametrize(
    ('discriminator', 'is_swept', 'column'),
    [
        ('|', True, None),
        ('maxSAR', False, 'max_sar'),
        ('99%SAR', False, 'sar_99'),
        ('maxLim', False, 'max_lim'),
    ],
)
def test_cruise_row_classifies_its_discriminator(discriminator, is_swept, column):
    tokens = list(CRUISE_TOKENS)
    tokens[3] = discriminator

    row = PianoCruiseRow.from_tokens(tokens)

    assert row is not None
    assert row.is_swept is is_swept
    assert row.reference_mach_column == column


def test_field_declaration_order_does_not_affect_parsing():
    """Only `COLUMNS` states the file order, so a refactor that reorders the
    fields cannot swap two columns."""

    class Row(_PianoRow):
        COLUMNS: ClassVar[tuple[str, ...]] = ('first', 'second')

        second: float
        first: float

    row = Row.from_tokens(['1', '2'])

    assert row is not None
    assert (row.first, row.second) == (1.0, 2.0)


@pytest.mark.parametrize(
    'columns',
    [
        pytest.param(('first',), id='missing field'),
        pytest.param(('first', 'second', 'first'), id='duplicate'),
        pytest.param(('first', 'second', 'third'), id='unknown name'),
        pytest.param(('first', 'second', 'optional'), id='optional field'),
    ],
)
def test_row_class_with_columns_that_do_not_match_its_fields_raises(columns):
    with pytest.raises(TypeError, match='must name each required field once'):

        class Row(_PianoRow):
            COLUMNS: ClassVar[tuple[str, ...]] = columns

            first: float
            second: float
            optional: float | None = None

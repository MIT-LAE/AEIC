"""Tests for the PIANO performance export reader.

The fixtures in `tests/data/performance/piano` are anonymized dummy data.
Their numbers are internally inconsistent on purpose: the four climb blocks
repeat the same rows truncated at different altitudes, the `Time` column
repeats values so most rows span no time, and the descent blocks' final
cumulative `Burn` disagrees with their header total. Assert structure and
units here, never physical plausibility.
"""

import logging
from pathlib import Path

import pytest

from AEIC.config import config
from AEIC.parsers.piano_reader import PianoData, PianoOverrides
from AEIC.parsers.piano_reader.climb_reader import CLIMB_COLS
from AEIC.parsers.piano_reader.cruise_reader import (
    CRUISE_COLS,
    CRUISE_REFERENCE_MACH_COLS,
)
from AEIC.parsers.piano_reader.descent_reader import (
    DESCENT_COLS,
    DESCENT_IDLE_THRUST_COLS,
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


def test_block_counts(piano_data):
    """One climb block and one descent block per mass, four of each."""
    assert sorted(set(piano_data.climb.column('mass'))) == sorted(CLIMB_MASSES_KG)
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
    """`250./ 280.kcas/ mach 0.750 above 30000.feet`, inherited by every
    block."""
    assert piano_data.climb_speeds.cas_low == pytest.approx(250 * KNOTS_TO_MPS)
    assert piano_data.climb_speeds.cas_high == pytest.approx(280 * KNOTS_TO_MPS)
    assert piano_data.climb_speeds.mach == pytest.approx(0.750)
    assert piano_data.climb_speeds.crossover_altitude_m == pytest.approx(
        30000 * FEET_TO_METERS
    )


def test_descent_speed_schedule(piano_data):
    """`mach 0.750 above 30000.feet/ 280./ 250.kcas`, written high to low."""
    assert piano_data.descent_speeds.cas_low == pytest.approx(250 * KNOTS_TO_MPS)
    assert piano_data.descent_speeds.cas_high == pytest.approx(280 * KNOTS_TO_MPS)
    assert piano_data.descent_speeds.mach == pytest.approx(0.750)
    assert piano_data.descent_speeds.crossover_altitude_m == pytest.approx(
        30000 * FEET_TO_METERS
    )


def test_climb_schedule_overrides(piano_data):
    overridden = load(climb_cas_low_kts=240.0, climb_mach=0.72)
    assert overridden.climb_speeds.cas_low == pytest.approx(240 * KNOTS_TO_MPS)
    assert overridden.climb_speeds.mach == pytest.approx(0.72)
    # Untouched values still come from the file.
    assert overridden.climb_speeds.cas_high == piano_data.climb_speeds.cas_high


def test_crossover_altitude_override(piano_data):
    overridden = load(climb_crossover_altitude_ft=28000.0)
    assert overridden.climb_speeds.crossover_altitude_m == pytest.approx(
        28000 * FEET_TO_METERS
    )
    # Untouched values still come from the file.
    assert overridden.climb_speeds.mach == piano_data.climb_speeds.mach


def _climb_without_schedule(tmp_path: Path) -> Path:
    """The climb fixture with its one airspeed schedule line removed."""
    lines = config.file_location(PIANO_CLIMB_FILE).read_text().splitlines(keepends=True)
    climb_file = tmp_path / 'climb.txt'
    kept = [line for line in lines if 'Airspeed schedule' not in line]
    climb_file.write_text(''.join(kept))
    return climb_file


def test_labelled_mach_on_the_swept_grid_keeps_one_row(piano_data, caplog):
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


def test_duplicate_cruise_row_disagreement_warns(tmp_path, caplog):
    """PIANO does not always report the same values for a labelled row and
    the swept row it lands on. The labelled row wins and the disagreeing
    columns are named."""
    cruise_text = config.file_location(PIANO_CRUISE_FILE).read_text()
    lines = []
    for line in cruise_text.splitlines(keepends=True):
        tokens = line.split()
        if len(tokens) > 12 and tokens[3] == 'maxSAR':
            # Halve the maximum climb thrust available on the labelled row.
            tokens[12] = '9000.'
            line = ' '.join(tokens) + '\n'
        lines.append(line)
    cruise_file = tmp_path / 'cruise.txt'
    cruise_file.write_text(''.join(lines))

    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        piano = PianoData.load(
            str(cruise_file),
            str(config.file_location(PIANO_CLIMB_FILE)),
            str(config.file_location(PIANO_DESCENT_FILE)),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )

    matching = [row for row in piano.cruise.data if row[0] == 150.0 and row[2] == 0.450]
    assert len(matching) == 3
    # The labelled row wins, so its lower thrust replaces the swept row's.
    assert all(
        row[CRUISE_COLS.index('mcl_avail_per_engine')]
        == pytest.approx(9000 * POUNDS_FORCE_TO_NEWTONS)
        for row in matching
    )
    assert any(
        'Duplicate cruise row' in r.getMessage()
        and 'mcl_avail_per_engine' in r.getMessage()
        for r in caplog.records
    )


def test_unrecognised_cruise_label_raises(tmp_path):
    """An unknown label in the fourth column means the export comes from a
    PIANO layout this parser was not written against. Nothing in the file can
    be trusted after that, so refuse it rather than drop rows."""
    text = (
        config.file_location(PIANO_CRUISE_FILE).read_text().replace('99%SAR', '98%SAR')
    )
    cruise_file = tmp_path / 'cruise.txt'
    cruise_file.write_text(text)

    with pytest.raises(ValueError, match='does not match the PIANO cruise'):
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

    with pytest.raises(ValueError, match='does not match the PIANO cruise'):
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


def test_cruise_reference_mach_drops_incomplete_groups(tmp_path, caplog):
    """A group is emitted only when all three labels are present, because
    table data is a list of numbers and TOML has no null."""
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

    assert piano.cruise_reference_mach.data == []
    assert any(
        'incomplete cruise reference Mach group' in r.getMessage()
        for r in caplog.records
    )


def test_zero_time_rows_are_dropped(piano_data, caplog):
    """The fixture's `Time` column repeats values, so most rows span no time
    and have no defined fuel flow. Of each block's 30 rows, the four at 0,
    1417, 7086 and 35431 feet are the only ones that advance the clock."""
    for mass in CLIMB_MASSES_KG:
        kept = [row for row in piano_data.climb.data if row[1] == mass]
        assert [row[0] for row in kept] == [0.0, 14.17, 70.86, 354.31]

    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        load()
    assert any('spans no time' in r.getMessage() for r in caplog.records)


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


def test_rocd_signs(piano_data):
    """Climb rates stay positive, descent rates are negated."""
    assert all(rocd > 0 for rocd in piano_data.climb.column('rocd'))
    assert all(rocd < 0 for rocd in piano_data.descent.column('rocd'))
    assert all(rocd == 0 for rocd in piano_data.cruise.column('rocd'))


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


def test_row_order(piano_data):
    """Cruise sorts by (mass, fl, mach); climb and descent by (mass, fl)."""
    cruise_keys = [(row[1], row[0], row[2]) for row in piano_data.cruise.data]
    assert cruise_keys == sorted(cruise_keys)
    for table in (piano_data.climb, piano_data.descent):
        keys = [(row[1], row[0]) for row in table.data]
        assert keys == sorted(keys)

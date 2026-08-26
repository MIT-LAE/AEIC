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

from AEIC.parsers.piano_reader import (
    CLIMB_COLS,
    CRUISE_COLS,
    CRUISE_REFERENCE_MACH_COLS,
    DESCENT_COLS,
    DESCENT_IDLE_THRUST_COLS,
    PianoData,
    PianoOverrides,
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

PIANO_DIR = (Path(__file__).parent / 'data' / 'performance' / 'piano').resolve()

CLIMB_MASSES_KG = [54000.0, 50000.0, 46000.0, 42000.0]
"""Climb block masses. The fixture states one only for the first block."""

DESCENT_MASSES_LB = [100000.0, 100001.0, 100002.0, 100003.0]


def load(**overrides) -> PianoData:
    return PianoData.load(
        str(PIANO_DIR / 'cruise.txt'),
        str(PIANO_DIR / 'climb.txt'),
        str(PIANO_DIR / 'descent.txt'),
        overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG, **overrides),
    )


@pytest.fixture(scope='module')
def piano() -> PianoData:
    return load()


def test_block_counts(piano):
    """One climb block and one descent block per mass, four of each."""
    assert sorted(set(piano.climb.column('mass'))) == sorted(CLIMB_MASSES_KG)
    assert sorted(set(piano.descent.column('mass'))) == [
        pytest.approx(lb * POUNDS_TO_KG) for lb in DESCENT_MASSES_LB
    ]


def test_climb_mass_count_mismatch_names_both_counts():
    with pytest.raises(ValueError, match=r'Got 2 climb mass\(es\) for 4 climb'):
        PianoData.load(
            str(PIANO_DIR / 'cruise.txt'),
            str(PIANO_DIR / 'climb.txt'),
            str(PIANO_DIR / 'descent.txt'),
            overrides=PianoOverrides(climb_masses_kg=[54000.0, 50000.0]),
        )


def test_climb_masses_required_when_blocks_have_no_header():
    """Only the first climb block states an initial mass, so all four must be
    supplied together."""
    with pytest.raises(ValueError, match='Climb block 2 has no "Initial mass"'):
        PianoData.load(
            str(PIANO_DIR / 'cruise.txt'),
            str(PIANO_DIR / 'climb.txt'),
            str(PIANO_DIR / 'descent.txt'),
        )


def test_climb_speed_schedule(piano):
    """`250./ 280.kcas/ mach 0.750 above 30000.feet`, inherited by every
    block."""
    assert piano.climb_speeds.cas_low == pytest.approx(250 * KNOTS_TO_MPS)
    assert piano.climb_speeds.cas_high == pytest.approx(280 * KNOTS_TO_MPS)
    assert piano.climb_speeds.mach == pytest.approx(0.750)
    assert piano.climb_speeds.crossover_altitude_m == pytest.approx(
        30000 * FEET_TO_METERS
    )


def test_descent_speed_schedule(piano):
    """`mach 0.750 above 30000.feet/ 280./ 250.kcas`, written high to low."""
    assert piano.descent_speeds.cas_low == pytest.approx(250 * KNOTS_TO_MPS)
    assert piano.descent_speeds.cas_high == pytest.approx(280 * KNOTS_TO_MPS)
    assert piano.descent_speeds.mach == pytest.approx(0.750)
    assert piano.descent_speeds.crossover_altitude_m == pytest.approx(
        30000 * FEET_TO_METERS
    )


def test_climb_schedule_overrides(piano):
    overridden = load(climb_cas_low_kts=240.0, climb_mach=0.72)
    assert overridden.climb_speeds.cas_low == pytest.approx(240 * KNOTS_TO_MPS)
    assert overridden.climb_speeds.mach == pytest.approx(0.72)
    # Untouched values still come from the file.
    assert overridden.climb_speeds.cas_high == piano.climb_speeds.cas_high


def test_labelled_mach_on_the_swept_grid_keeps_one_row(piano, caplog):
    """A labelled reference Mach can land exactly on the swept grid. The
    fixture labels maxSAR at 0.450, which the sweep also visits."""
    matching = [
        row
        for row in piano.cruise.data
        if row[0] == 150.0
        and row[2] == 0.450
        and row[1] == pytest.approx(93000 * POUNDS_TO_KG)
    ]
    assert len(matching) == 1

    # The fixture's two rows agree on every column, so nothing is reported.
    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        load()
    assert not any('Duplicate cruise row' in r.getMessage() for r in caplog.records)


def test_duplicate_cruise_row_disagreement_warns(tmp_path, caplog):
    """PIANO does not always report the same values for a labelled row and
    the swept row it lands on. The swept row wins and the disagreeing columns
    are named."""
    lines = []
    for line in (PIANO_DIR / 'cruise.txt').read_text().splitlines(keepends=True):
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
            str(PIANO_DIR / 'climb.txt'),
            str(PIANO_DIR / 'descent.txt'),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )

    matching = [row for row in piano.cruise.data if row[0] == 150.0 and row[2] == 0.450]
    assert len(matching) == 3
    # The swept row wins, so the labelled row's lower thrust is gone.
    assert all(
        row[CRUISE_COLS.index('mcl_avail_per_engine')]
        == pytest.approx(19000 * POUNDS_FORCE_TO_NEWTONS)
        for row in matching
    )
    assert any(
        'Duplicate cruise row' in r.getMessage()
        and 'mcl_avail_per_engine' in r.getMessage()
        for r in caplog.records
    )


def test_cruise_reference_mach_complete_groups(piano):
    """The fixture labels all three reference Machs for every (fl, mass)."""
    assert piano.cruise_reference_mach.cols == CRUISE_REFERENCE_MACH_COLS
    assert len(piano.cruise_reference_mach.data) == 6
    assert piano.cruise_reference_mach.data[0] == [
        150.0,
        pytest.approx(93000 * POUNDS_TO_KG),
        0.450,
        0.490,
        0.690,
    ]


def test_cruise_reference_mach_drops_incomplete_groups(tmp_path, caplog):
    """A group is emitted only when all three labels are present, because
    table data is a list of numbers and TOML has no null."""
    lines = (PIANO_DIR / 'cruise.txt').read_text().splitlines(keepends=True)
    trimmed = [line for line in lines if 'maxLim' not in line]
    cruise_file = tmp_path / 'cruise.txt'
    cruise_file.write_text(''.join(trimmed))

    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        piano = PianoData.load(
            str(cruise_file),
            str(PIANO_DIR / 'climb.txt'),
            str(PIANO_DIR / 'descent.txt'),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )

    assert piano.cruise_reference_mach.data == []
    assert any(
        'incomplete cruise reference Mach group' in r.getMessage()
        for r in caplog.records
    )


def test_zero_time_rows_are_dropped(piano, caplog):
    """The fixture's `Time` column repeats values, so most rows span no time
    and have no defined fuel flow. Of each block's 30 rows, the four at 0,
    1417, 7086 and 35431 feet are the only ones that advance the clock."""
    for mass in CLIMB_MASSES_KG:
        kept = [row for row in piano.climb.data if row[1] == mass]
        assert [row[0] for row in kept] == [0.0, 14.17, 70.86, 354.31]

    with caplog.at_level(logging.WARNING, logger='AEIC.parsers.piano_reader'):
        load()
    assert any('spans no time' in r.getMessage() for r in caplog.records)


def test_cruise_row_units(piano):
    """Spot-check every converted column of one cruise row."""
    row = piano.cruise.data[0]
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


def test_climb_row_units(piano):
    """The first climb row is at sea level, where CAS and TAS coincide."""
    row = piano.climb.data[0]
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


def test_descent_row_units(piano):
    """Descent rows carry no drag column, and PIANO's rate of descent is
    negated."""
    row = piano.descent.data[-1]
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


def test_rocd_signs(piano):
    """Climb rates stay positive, descent rates are negated."""
    assert all(rocd > 0 for rocd in piano.climb.column('rocd'))
    assert all(rocd < 0 for rocd in piano.descent.column('rocd'))
    assert all(rocd == 0 for rocd in piano.cruise.column('rocd'))


def test_non_zero_delta_isa_raises(tmp_path):
    """Deriving TAS from the airspeed schedule assumes a standard
    atmosphere."""
    climb_file = tmp_path / 'climb.txt'
    climb_file.write_text(
        (PIANO_DIR / 'climb.txt')
        .read_text()
        .replace('Delta-ISA           +0.', 'Delta-ISA           +10.')
    )

    with pytest.raises(ValueError, match=r'Delta-ISA of \+10 deg C'):
        PianoData.load(
            str(PIANO_DIR / 'cruise.txt'),
            str(climb_file),
            str(PIANO_DIR / 'descent.txt'),
            overrides=PianoOverrides(climb_masses_kg=CLIMB_MASSES_KG),
        )


def test_aircraft_name_and_maximum_altitude(piano):
    assert piano.aircraft_name == 'some_airplane, some_engine'
    # The highest altitude actually reached, not the header's request of
    # 41000 feet. Only the first climb block gets there.
    assert piano.maximum_altitude_ft == 41100
    assert piano.isa_offset == 0


def test_descent_idle_thrust(piano):
    assert piano.descent_idle_thrust.cols == DESCENT_IDLE_THRUST_COLS
    assert piano.descent_idle_thrust.data == [
        [pytest.approx(mass_lb * POUNDS_TO_KG), pytest.approx(30000 * FEET_TO_METERS)]
        for mass_lb in DESCENT_MASSES_LB
    ]


def test_row_order(piano):
    """Cruise sorts by (mass, fl, mach); climb and descent by (mass, fl)."""
    cruise_keys = [(row[1], row[0], row[2]) for row in piano.cruise.data]
    assert cruise_keys == sorted(cruise_keys)
    for table in (piano.climb, piano.descent):
        keys = [(row[1], row[0]) for row in table.data]
        assert keys == sorted(keys)

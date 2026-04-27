# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

import pytest

from AEIC.config import config
from AEIC.performance.models import LegacyPerformanceModel, PerformanceModel
from AEIC.performance.models.legacy import (
    PerformanceTable,
    PerformanceTableInput,
    ROCDFilter,
)
from AEIC.performance.types import AircraftState
from AEIC.units import METERS_TO_FL

# Per-phase per-input-row recovery checks. A single-cell verification
# would still pass a regression that, e.g., transposed FL ↔ mass on
# construction if it happened to keep one coordinate fixed. Loop over
# every input row and assert each is recoverable at its (fl, mass) key
# with the original tas / fuel_flow / rocd values. The phase-split
# means we cover three tables (climb / cruise / descent) of differing
# shape, including the descent single-mass case.


def _verify_rows_recoverable(rows, model):
    for fl, ff, tas, rocd, mass in rows:
        match = model.df[(model.df.fl == fl) & (model.df.mass == mass)]
        assert len(match) == 1, f'no row at fl={fl}, mass={mass}'
        assert match.tas.values[0] == tas
        assert match.fuel_flow.values[0] == ff
        assert match.rocd.values[0] == rocd


def test_create_performance_table_cruise():
    rows = _cruise_rows()
    model = _build(rows, ROCDFilter.ZERO)
    assert model.fl == [330, 350]
    assert model.mass == [60000, 70000, 80000]
    _verify_rows_recoverable(rows, model)


def test_create_performance_table_climb():
    rows = _climb_rows()
    model = _build(rows, ROCDFilter.POSITIVE)
    assert model.fl == [330, 350]
    assert model.mass == [60000, 70000, 80000]
    _verify_rows_recoverable(rows, model)


def test_create_performance_table_descent():
    rows = _descent_rows()
    model = _build(rows, ROCDFilter.NEGATIVE)
    assert model.fl == [330, 350]
    assert model.mass == [70000]
    _verify_rows_recoverable(rows, model)


@pytest.mark.parametrize(
    'cols, data, match',
    [
        (
            ['fl', 'fl', 'fuel_flow', 'tas', 'rocd', 'mass'],
            [[330.0, 330.0, 0.5, 220.0, 0.0, 60000.0]],
            r'Duplicate column names in performance table',
        ),
        (
            ['FUEL_FLOW', 'TAS', 'ROCD', 'MASS'],
            [[0.5, 220.0, 0.0, 60000.0]],
            r'Missing required "fl" column',
        ),
        (
            ['FL', 'FUEL_FLOW', 'ROCD', 'MASS'],
            [[330.0, 0.5, 0.0, 60000.0]],
            r'Missing required "tas" column',
        ),
        (
            ['FL', 'FUEL_FLOW', 'TAS', 'MASS'],
            [[330.0, 0.5, 220.0, 60000.0]],
            r'Missing required "rocd" column',
        ),
        (
            ['FL', 'FUEL_FLOW', 'TAS', 'ROCD'],
            [[330.0, 0.5, 220.0, 0.0]],
            r'Missing required "mass" column',
        ),
        (
            ['FL', 'TAS'],
            [[330.0, 220.0]],
            r'Missing required "fuel_flow" column',
        ),
        (
            ['FL', 'FUEL_FLOW', 'TAS', 'ROCD', 'MASS', 'EXTRA'],
            [[330.0, 0.5, 220.0, 0.0, 60000.0]],
            r'Not enough data columns in performance table',
        ),
        (
            ['FL', 'FUEL_FLOW', 'TAS', 'ROCD', 'MASS'],
            [[330.0, 0.5, 220.0, 0.0, 60000.0], [330.0, 0.5, 220.0, 0.0]],
            r'Inconsistent number of data columns in performance table',
        ),
    ],
)
def test_performance_table_input_rejects(cols, data, match):
    with pytest.raises(ValueError, match=match):
        _ = PerformanceTableInput(cols=cols, data=data)


# Shared scaffolding for PerformanceTable.__post_init__ negative-path tests.
#
# Each helper builds a per-phase row set whose TAS, fuel_flow, and ROCD
# satisfy what __post_init__ requires for that phase:
#   - climb (POSITIVE): ROCD ≥ 0; TAS and fuel_flow depend only on FL;
#     ROCD may vary with mass.
#   - cruise (ZERO):    |ROCD| ≤ ZERO_ROCD_TOL; TAS depends only on FL;
#     fuel_flow may vary with mass.
#   - descent (NEGATIVE): ROCD < 0; exactly one mass; everything FL-only.
# Each negative test mutates one cell or drops one row to trip a single
# named raise branch.

_COLS = ['FL', 'FUEL_FLOW', 'TAS', 'ROCD', 'MASS']
_COL_IDX = {name: i for i, name in enumerate(_COLS)}


def _climb_rows():
    """2 FLs × 3 masses, all positive ROCD (varies with mass)."""
    rows = []
    for fl in (330, 350):
        for mass in (60000, 70000, 80000):
            tas = 200 + (fl - 300) // 10
            ff = round(0.5 + 0.001 * fl, 6)
            rocd = 1500.0 - 0.01 * mass
            rows.append([fl, ff, tas, rocd, mass])
    return rows


def _cruise_rows():
    """2 FLs × 3 masses, ROCD = 0 (cruise)."""
    rows = []
    for fl in (330, 350):
        for mass in (60000, 70000, 80000):
            tas = 220 + (fl - 300) // 10
            ff = round(0.5 + 0.001 * fl + 0.000001 * mass, 6)
            rows.append([fl, ff, tas, 0.0, mass])
    return rows


def _descent_rows():
    """2 FLs × 1 mass, all negative ROCD (descent shape)."""
    rows = []
    mass = 70000
    for fl in (330, 350):
        tas = 240 + (fl - 300) // 10
        ff = round(0.5 + 0.001 * fl, 6)
        rocd = -500.0 - (fl - 300) * 1.0
        rows.append([fl, ff, tas, rocd, mass])
    return rows


def _build(rows, rocd_type):
    return PerformanceTable.from_input(
        PerformanceTableInput(cols=_COLS, data=rows), rocd_type=rocd_type
    )


def _drop_cell(fl, mass):
    return lambda rows: [
        r for r in rows if not (r[_COL_IDX['FL']] == fl and r[_COL_IDX['MASS']] == mass)
    ]


def _mutate_cell(fl, mass, col, new):
    idx = _COL_IDX[col]

    def _apply(rows):
        out = [list(r) for r in rows]
        for r in out:
            if r[_COL_IDX['FL']] == fl and r[_COL_IDX['MASS']] == mass:
                r[idx] = new
        return out

    return _apply


def test_performance_table_baselines_valid():
    # Per-phase baselines must load — otherwise every mutation below
    # would trip an unrelated branch.
    _build(_climb_rows(), ROCDFilter.POSITIVE)
    _build(_cruise_rows(), ROCDFilter.ZERO)
    _build(_descent_rows(), ROCDFilter.NEGATIVE)


# ROCD-sign-mismatch raises (one per phase, distinct messages).
@pytest.mark.parametrize(
    'rows_fn, rocd_type, mutate, match',
    [
        (
            _descent_rows,
            ROCDFilter.NEGATIVE,
            _mutate_cell(fl=330, mass=70000, col='ROCD', new=0.0),
            r'ROCD values in descent performance table are not all negative',
        ),
        (
            _cruise_rows,
            ROCDFilter.ZERO,
            _mutate_cell(fl=330, mass=60000, col='ROCD', new=10.0),
            r'ROCD values in cruise performance table are not all zero',
        ),
        (
            _climb_rows,
            ROCDFilter.POSITIVE,
            _mutate_cell(fl=330, mass=60000, col='ROCD', new=-100.0),
            r'some ROCD values in climb performance table are negative',
        ),
    ],
)
def test_performance_table_rocd_sign_rejects(rows_fn, rocd_type, mutate, match):
    with pytest.raises(ValueError, match=match):
        _build(mutate(rows_fn()), rocd_type)


# Mass-count rejects (one per phase). Climb/cruise allow (2, 3); descent
# requires exactly 1.
@pytest.mark.parametrize(
    'rocd_type, build, match',
    [
        (
            ROCDFilter.POSITIVE,
            lambda: [
                [
                    fl,
                    0.5 + 0.001 * fl,
                    200 + (fl - 300) // 10,
                    1500.0 - 0.01 * mass,
                    mass,
                ]
                for fl in (330, 350)
                for mass in (60000, 65000, 70000, 80000)  # 4 masses ∉ (2, 3)
            ],
            r'Legacy performance table \(climb\) has wrong number of mass values',
        ),
        (
            ROCDFilter.ZERO,
            lambda: [
                [fl, 0.5 + 0.001 * fl, 220 + (fl - 300) // 10, 0.0, 70000]
                for fl in (330, 350)  # 1 mass ∉ (2, 3)
            ],
            r'Legacy performance table \(cruise\) has wrong number of mass values',
        ),
        (
            ROCDFilter.NEGATIVE,
            lambda: [
                [
                    fl,
                    0.5 + 0.001 * fl,
                    240 + (fl - 300) // 10,
                    -500.0 - (fl - 300),
                    mass,
                ]
                for fl in (330, 350)
                for mass in (60000, 70000)  # 2 masses ≠ 1
            ],
            r'Legacy performance table \(descent\) has wrong number of mass values',
        ),
    ],
)
def test_performance_table_wrong_mass_count(rocd_type, build, match):
    with pytest.raises(ValueError, match=match):
        _build(build(), rocd_type)


# Coverage and FL-only-dependency raises. Descent's coverage and
# per-variable FL-only checks are unreachable from valid descent data
# once the mass-count == 1 invariant holds (any (fl, mass) drop also
# drops the corresponding FL, keeping #FL × #mass == #rows; with one
# mass each FL trivially maps to a single value of every column). They
# remain in __post_init__ as belt-and-braces and are intentionally not
# covered here.
@pytest.mark.parametrize(
    'rows_fn, rocd_type, mutate, match',
    [
        # Coverage gaps (climb, cruise — descent unreachable per note above).
        (
            _climb_rows,
            ROCDFilter.POSITIVE,
            _drop_cell(fl=330, mass=60000),
            r'Performance data for climb does not have full coverage',
        ),
        (
            _cruise_rows,
            ROCDFilter.ZERO,
            _drop_cell(fl=330, mass=60000),
            r'Performance data for cruise does not have full coverage',
        ),
        # FL-only-dependency violations (cruise checks tas only; climb
        # checks tas + fuel_flow; descent's fl-only checks are
        # unreachable per note above).
        (
            _cruise_rows,
            ROCDFilter.ZERO,
            _mutate_cell(fl=330, mass=60000, col='TAS', new=999.0),
            r'tas for cruise phase depends on variables other than FL',
        ),
        (
            _climb_rows,
            ROCDFilter.POSITIVE,
            _mutate_cell(fl=330, mass=60000, col='TAS', new=999.0),
            r'tas for climb phase depends on variables other than FL',
        ),
        (
            _climb_rows,
            ROCDFilter.POSITIVE,
            _mutate_cell(fl=330, mass=60000, col='FUEL_FLOW', new=9.999),
            r'fuel_flow for climb phase depends on variables other than FL',
        ),
    ],
)
def test_performance_table_post_init_rejects(rows_fn, rocd_type, mutate, match):
    with pytest.raises(ValueError, match=match):
        _build(mutate(rows_fn()), rocd_type)


def _sample_model() -> LegacyPerformanceModel:
    model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )
    assert isinstance(model, LegacyPerformanceModel)
    return model


def test_sample_model_per_phase_contracts():
    """Pin the BADA-3 per-phase shape on the sample model: cruise is
    non-empty with all-zero ROCD, climb carries three masses, descent
    carries one. The validator's mass-count rule on main allows climb /
    cruise to be (2, 3); a TOML edit that silently dropped one cruise
    or climb mass would fall through that rule. This test pins the
    sample's actual shape so any drift surfaces as a clear failure.
    """
    model = _sample_model()

    cruise = model.performance_table(ROCDFilter.ZERO)
    assert len(cruise) > 0
    assert all(abs(rocd) <= PerformanceTable.ZERO_ROCD_TOL for rocd in cruise.df.rocd)
    assert len(cruise.mass) == 3

    climb = model.performance_table(ROCDFilter.POSITIVE)
    assert len(climb.mass) == 3

    descent = model.performance_table(ROCDFilter.NEGATIVE)
    assert len(descent.mass) == 1


def test_interpolate_bilinear_recovers_cell_and_midpoint():
    """The cruise (ROCDFilter.ZERO) phase table on the sample model has
    n_masses>1, so interpolate goes through the bilinear branch of
    `Interpolator.__call__`. Pin both: at an exact (fl, mass) cell the
    returned values equal the cell's stored values; at the centroid of
    four corners the result equals the simple corner average.
    """
    table = _sample_model().performance_table(ROCDFilter.ZERO)
    fl_a, fl_b = table.fl[0], table.fl[1]
    mass_a, mass_b = table.mass[0], table.mass[1]

    def cell(fl, mass, col):
        match = table.df[(table.df.fl == fl) & (table.df.mass == mass)]
        return match[col].values[0]

    # Exact cell.
    state_corner = AircraftState(altitude=fl_a / METERS_TO_FL, aircraft_mass=mass_a)
    perf_corner = table.interpolate(state_corner)
    assert table._interpolator.n_masses > 1  # bilinear path
    assert perf_corner.true_airspeed == pytest.approx(cell(fl_a, mass_a, 'tas'))
    assert perf_corner.fuel_flow == pytest.approx(cell(fl_a, mass_a, 'fuel_flow'))

    # Centroid of four corners — bilinear result is the simple mean.
    expected_tas = (
        sum(cell(f, m, 'tas') for f in (fl_a, fl_b) for m in (mass_a, mass_b)) / 4
    )
    expected_ff = (
        sum(cell(f, m, 'fuel_flow') for f in (fl_a, fl_b) for m in (mass_a, mass_b)) / 4
    )
    state_mid = AircraftState(
        altitude=(fl_a + fl_b) / 2 / METERS_TO_FL,
        aircraft_mass=(mass_a + mass_b) / 2,
    )
    perf_mid = table.interpolate(state_mid)
    assert perf_mid.true_airspeed == pytest.approx(expected_tas)
    assert perf_mid.fuel_flow == pytest.approx(expected_ff)


def test_interpolate_fl_only_fallback_for_descent():
    """The descent (ROCDFilter.NEGATIVE) phase table on a BADA-shape
    sample has n_masses==1, so interpolate goes through the FL-only
    fallback in `Interpolator`. At an exact FL cell the returned values
    equal the cell's stored values.
    """
    table = _sample_model().performance_table(ROCDFilter.NEGATIVE)
    assert len(table.mass) == 1
    mass = table.mass[0]
    fl = table.fl[0]
    cell = table.df[table.df.fl == fl].iloc[0]

    state = AircraftState(altitude=fl / METERS_TO_FL, aircraft_mass=mass)
    perf = table.interpolate(state)
    assert table._interpolator.n_masses == 1  # FL-only path
    assert perf.true_airspeed == pytest.approx(cell.tas)
    assert perf.fuel_flow == pytest.approx(cell.fuel_flow)
    assert perf.rate_of_climb == pytest.approx(cell.rocd)

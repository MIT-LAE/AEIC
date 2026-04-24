# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

import pytest

from AEIC.performance.models.legacy import (
    PerformanceTable,
    PerformanceTableInput,
    ROCDFilter,
)


def test_create_performance_table():
    def fuel_flow(fl: int, tas: int) -> float:
        return round(0.5 + 0.001 * fl + 0.0001 * tas, 6)

    def tas(fl: int) -> int:
        return 220 + (fl - 300) // 10

    cols = ['FL', 'FUEL_FLOW', 'TAS', 'ROCD', 'MASS']
    rows = []
    for fl in (330, 350):
        for mass in (60000, 70000, 80000):
            v = tas(fl)
            ff = fuel_flow(fl, v)
            rows.append([fl, ff, v, 0.0, mass])
    model = PerformanceTable.from_input(
        PerformanceTableInput(cols=cols, data=rows), rocd_type=ROCDFilter.ZERO
    )

    assert model.fl == [330, 350]
    assert model.mass == [60000, 70000, 80000]

    assert model.df[
        (model.df.fl == 350)
        & (model.df.tas == tas(350))
        & (model.df.rocd == 0)
        & (model.df.mass == 60000)
    ].fuel_flow.values[0] == fuel_flow(350, tas(350))


def test_create_performance_table_missing_output_column():
    with pytest.raises(
        ValueError, match='Missing required "fuel_flow" column in performance table'
    ):
        _ = PerformanceTableInput(cols=['FL', 'TAS'], data=[[330, 220]])


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

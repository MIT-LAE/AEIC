"""Tests for `RaggedInterpolator`, on synthetic tables."""

import re

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import RegularGridInterpolator

from AEIC.performance.models.ragged import Interval, RaggedInterpolator


def table(cols: tuple[str, ...], rows: list[tuple[float, ...]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(cols))


def raises_message(message: str):
    return pytest.raises(ValueError, match=f'^{re.escape(message)}$')


# Two masses with different FL points. Mass 3 stops at a lower FL and mass 1
# starts at a higher FL.
RAGGED_2D = table(
    ('mass', 'fl', 'tas'),
    [
        (1.0, 5.0, 5.0),
        (1.0, 10.0, 10.0),
        (1.0, 20.0, 20.0),
        (3.0, 0.0, 100.0),
        (3.0, 5.0, 110.0),
        (3.0, 15.0, 130.0),
    ],
)

# A cruise-like table. The Mach sweep of the (fl=400, mass=70) cell starts at
# a higher Mach than the other cells.
RAGGED_3D = table(
    ('fl', 'mass', 'mach', 'fuel_flow'),
    [
        (300.0, 50.0, 0.70, 1.0),
        (300.0, 50.0, 0.75, 1.5),
        (300.0, 50.0, 0.80, 2.5),
        (300.0, 70.0, 0.70, 2.0),
        (300.0, 70.0, 0.75, 2.5),
        (300.0, 70.0, 0.80, 3.0),
        (400.0, 50.0, 0.70, 4.0),
        (400.0, 50.0, 0.75, 4.5),
        (400.0, 50.0, 0.80, 6.0),
        (400.0, 70.0, 0.75, 5.0),
        (400.0, 70.0, 0.80, 7.0),
    ],
)


def test_rectilinear_2d_is_bilinear():
    df = table(
        ('mass', 'fl', 'tas'),
        [
            (1.0, 0.0, 1.0),
            (1.0, 10.0, 2.0),
            (1.0, 20.0, 4.0),
            (3.0, 0.0, 5.0),
            (3.0, 10.0, 11.0),
            (3.0, 20.0, 13.0),
        ],
    )
    interp = RaggedInterpolator(df, ('mass', 'fl'), ('tas',), 'test')

    # mass 1 at FL 12.5: 2.5. Mass 3 at FL 12.5: 11.5. Weight 0.25 on mass 3.
    assert interp(1.5, 12.5) == pytest.approx((0.75 * 2.5 + 0.25 * 11.5,))


def test_rectilinear_3d_is_trilinear():
    df = table(
        ('fl', 'mass', 'mach', 'fuel_flow'),
        [
            (300.0, 50.0, 0.7, 1.0),
            (300.0, 50.0, 0.8, 3.0),
            (300.0, 70.0, 0.7, 2.0),
            (300.0, 70.0, 0.8, 8.0),
            (400.0, 50.0, 0.7, 4.0),
            (400.0, 50.0, 0.8, 4.0),
            (400.0, 70.0, 0.7, 0.0),
            (400.0, 70.0, 0.8, 10.0),
        ],
    )
    interp = RaggedInterpolator(df, ('fl', 'mass', 'mach'), ('fuel_flow',), 'test')

    # Weights: 0.5 in FL, 0.25 in mass, 0.75 in Mach.
    fl300 = 0.75 * (1.0 + 0.75 * 2.0) + 0.25 * (2.0 + 0.75 * 6.0)
    fl400 = 0.75 * (4.0 + 0.75 * 0.0) + 0.25 * (0.0 + 0.75 * 10.0)
    assert interp(350.0, 55.0, 0.775) == pytest.approx((0.5 * fl300 + 0.5 * fl400,))


def test_regular_grid_matches_scipy():
    # Can't test multilinear interpolation on a ragged grid directly vs scipy
    # Instead we make synthetic data that is rectangular such that we can
    # verify accuracy against scipy's RegularGridInterpolator
    # The ragged-specific checks are tested in the other tests

    # Uneven spacing on each axis.
    grid = (
        np.array([300.0, 330.0, 410.0]),
        np.array([50.0, 58.0, 61.0, 75.0]),
        np.array([0.7, 0.725, 0.78, 0.8, 0.85]),
    )
    rng = np.random.default_rng(0)
    values = rng.uniform(1.0, 2.0, (*(len(g) for g in grid), 2))
    mesh = np.meshgrid(*grid, indexing='ij')
    df = pd.DataFrame(
        {
            'fl': mesh[0].ravel(),
            'mass': mesh[1].ravel(),
            'mach': mesh[2].ravel(),
            'tas': values[..., 0].ravel(),
            'fuel_flow': values[..., 1].ravel(),
        }
    )
    # Give the rows in random order, so that the test also checks the sort.
    df = df.sample(frac=1.0, random_state=0)
    interp = RaggedInterpolator(
        df, ('fl', 'mass', 'mach'), ('tas', 'fuel_flow'), 'test'
    )

    # On each axis, take every node and every midpoint, so that the points
    # mix exact nodes on some axes with interpolation on the others. Add
    # random points in the grid.
    nodes_and_midpoints = [np.sort(np.r_[g, (g[:-1] + g[1:]) / 2]) for g in grid]
    mixed = np.stack(
        [m.ravel() for m in np.meshgrid(*nodes_and_midpoints, indexing='ij')],
        axis=1,
    )
    lo = np.array([g[0] for g in grid])
    hi = np.array([g[-1] for g in grid])
    random = lo + rng.random((200, 3)) * (hi - lo)
    points = np.concatenate([mixed, random])

    expected = RegularGridInterpolator(grid, values)(points)
    for point, value in zip(points, expected):
        assert interp(*point) == pytest.approx(tuple(value), rel=1e-15, abs=0.0)


@pytest.mark.parametrize(
    ('df', 'axes', 'outputs'),
    [
        (RAGGED_2D, ('mass', 'fl'), ('tas',)),
        (RAGGED_3D, ('fl', 'mass', 'mach'), ('fuel_flow',)),
    ],
)
def test_every_row_returns_its_tabulated_outputs(df, axes, outputs):
    interp = RaggedInterpolator(df, axes, outputs, 'test')
    for row in df.itertuples(index=False):
        coords = [getattr(row, a) for a in axes]
        expected = tuple(getattr(row, o) for o in outputs)
        assert interp(*coords) == expected


def test_ragged_inner_axis_interpolates_each_curve_at_the_query_fl():
    interp = RaggedInterpolator(RAGGED_2D, ('mass', 'fl'), ('tas',), 'test')

    # Mass 1 at FL 12.5: 12.5. Mass 3 at FL 12.5: 110 + 0.75 * 20 = 125.
    assert interp(2.0, 12.5) == pytest.approx((0.5 * 12.5 + 0.5 * 125.0,))


def test_ragged_ceiling():
    interp = RaggedInterpolator(RAGGED_2D, ('mass', 'fl'), ('tas',), 'test')

    with raises_message('test table: fl=18.0 outside [0.0, 15.0] for mass=3.0'):
        interp(2.0, 18.0)

    # On the lighter mass, the heavier mass is not used.
    assert interp(1.0, 18.0) == pytest.approx((18.0,))


def test_ragged_floor():
    interp = RaggedInterpolator(RAGGED_2D, ('mass', 'fl'), ('tas',), 'test')

    with raises_message('test table: fl=2.0 outside [5.0, 20.0] for mass=1.0'):
        interp(2.0, 2.0)

    # On the heavier mass, the lighter mass is not used.
    assert interp(3.0, 2.0) == pytest.approx((104.0,))


def test_ragged_lower_mach_bound_matters_only_at_a_corner():
    interp = RaggedInterpolator(
        RAGGED_3D, ('fl', 'mass', 'mach'), ('fuel_flow',), 'test'
    )

    with raises_message(
        'test table: mach=0.72 outside [0.75, 0.8] for fl=400.0, mass=70.0'
    ):
        interp(350.0, 60.0, 0.72)

    # The (fl=400, mass=70) cell is not a corner of these queries.
    assert interp(350.0, 50.0, 0.72) == pytest.approx((0.5 * 1.2 + 0.5 * 4.2,))
    assert interp(300.0, 60.0, 0.72) == pytest.approx((0.5 * 1.2 + 0.5 * 2.2,))


@pytest.mark.parametrize(
    ('coords', 'message'),
    [
        ((0.5, 10.0), 'test table: mass=0.5 outside [1.0, 3.0]'),
        ((3.5, 10.0), 'test table: mass=3.5 outside [1.0, 3.0]'),
        ((1.0, 4.0), 'test table: fl=4.0 outside [5.0, 20.0] for mass=1.0'),
        ((1.0, 21.0), 'test table: fl=21.0 outside [5.0, 20.0] for mass=1.0'),
        ((3.0, -1.0), 'test table: fl=-1.0 outside [0.0, 15.0] for mass=3.0'),
        ((3.0, 16.0), 'test table: fl=16.0 outside [0.0, 15.0] for mass=3.0'),
    ],
)
def test_out_of_range_message(coords, message):
    interp = RaggedInterpolator(RAGGED_2D, ('mass', 'fl'), ('tas',), 'test')
    with raises_message(message):
        interp(*coords)


@pytest.mark.parametrize(
    ('coords', 'message'),
    [
        ((float('nan'), 10.0), 'test table: mass=nan outside [1.0, 3.0]'),
        ((1.0, float('nan')), 'test table: fl=nan outside [5.0, 20.0] for mass=1.0'),
    ],
)
def test_nan_coordinate_raises(coords, message):
    interp = RaggedInterpolator(RAGGED_2D, ('mass', 'fl'), ('tas',), 'test')
    with raises_message(message):
        interp(*coords)


def test_one_node_axis_accepts_only_that_value():
    df = table(
        ('mass', 'fl', 'tas'),
        [(2.0, 0.0, 10.0), (2.0, 10.0, 20.0), (2.0, 20.0, 40.0)],
    )
    interp = RaggedInterpolator(df, ('mass', 'fl'), ('tas',), 'test')

    assert interp(2.0, 15.0) == pytest.approx((30.0,))
    with raises_message('test table: mass=2.5 outside [2.0, 2.0]'):
        interp(2.5, 15.0)


def test_outputs_follow_the_requested_order():
    df = table(
        ('fl', 'tas', 'fuel_flow'),
        [(0.0, 100.0, 1.0), (10.0, 200.0, 3.0)],
    )
    forward = RaggedInterpolator(df, ('fl',), ('tas', 'fuel_flow'), 'test')
    backward = RaggedInterpolator(df, ('fl',), ('fuel_flow', 'tas'), 'test')

    assert forward(5.0) == pytest.approx((150.0, 2.0))
    assert backward(5.0) == pytest.approx((2.0, 150.0))


def test_empty_table_raises_on_evaluation():
    df = table(('fl', 'tas'), [])
    interp = RaggedInterpolator(df, ('fl',), ('tas',), 'test')
    with raises_message('test table: no data'):
        interp(5.0)


# ---------------------------------------------------------------------------
# Valid range of one axis
# ---------------------------------------------------------------------------

# The middle masses do not cover Mach 0.72, so the valid masses at that Mach
# have a gap.
TABLE_WITH_GAP = table(
    ('mass', 'mach', 'fuel_flow'),
    [
        (1.0, 0.70, 1.0),
        (1.0, 0.80, 2.0),
        (2.0, 0.75, 3.0),
        (2.0, 0.80, 4.0),
        (3.0, 0.70, 5.0),
        (3.0, 0.80, 6.0),
        (4.0, 0.70, 7.0),
        (4.0, 0.80, 8.0),
        (5.0, 0.70, 9.0),
        (5.0, 0.80, 10.0),
    ],
)


# At Mach 0.72, the valid masses are [1, 1] and [3, 5] at FL300, and [1, 4]
# at FL400. Mass 2 at FL300 and mass 5 at FL400 do not cover Mach 0.72.
TABLE_WITH_GAPS_AT_TWO_FLS = table(
    ('fl', 'mass', 'mach', 'fuel_flow'),
    [
        (fl, mass, mach, fl / 100 + mass + mach)
        for fl, start_mach_by_mass in (
            (300.0, {1.0: 0.70, 2.0: 0.75, 3.0: 0.70, 4.0: 0.70, 5.0: 0.70}),
            (400.0, {1.0: 0.70, 2.0: 0.70, 3.0: 0.70, 4.0: 0.70, 5.0: 0.75}),
        )
        for mass, start_mach in start_mach_by_mass.items()
        for mach in (start_mach, 0.80)
    ],
)


def ragged_2d() -> RaggedInterpolator:
    return RaggedInterpolator(RAGGED_2D, ('mass', 'fl'), ('tas',), 'test')


def ragged_3d() -> RaggedInterpolator:
    return RaggedInterpolator(RAGGED_3D, ('fl', 'mass', 'mach'), ('fuel_flow',), 'test')


def table_with_gap() -> RaggedInterpolator:
    return RaggedInterpolator(TABLE_WITH_GAP, ('mass', 'mach'), ('fuel_flow',), 'test')


@pytest.mark.parametrize(
    ('mass', 'expected'),
    [
        pytest.param(1.0, [Interval(5.0, 20.0)], id='light_node'),
        pytest.param(3.0, [Interval(0.0, 15.0)], id='heavy_node'),
        # Between the masses, the FL must be in the range of both.
        pytest.param(2.0, [Interval(5.0, 15.0)], id='between_nodes'),
        pytest.param(0.5, [], id='outside'),
    ],
)
def test_valid_range_of_the_inner_axis(mass, expected):
    assert ragged_2d().valid_range('fl', mass=mass) == expected


@pytest.mark.parametrize(
    ('fl', 'expected'),
    [
        pytest.param(10.0, [Interval(1.0, 3.0)], id='both_masses'),
        # Above the heavy ceiling and below the light floor, only the exact
        # mass of the node that covers the FL is valid.
        pytest.param(18.0, [Interval(1.0, 1.0)], id='above_heavy_ceiling'),
        pytest.param(2.0, [Interval(3.0, 3.0)], id='below_light_floor'),
        pytest.param(25.0, [], id='above_all'),
    ],
)
def test_valid_range_of_the_outer_axis(fl, expected):
    assert ragged_2d().valid_range('mass', fl=fl) == expected


@pytest.mark.parametrize(
    ('fl', 'mach', 'expected'),
    [
        pytest.param(300.0, 0.72, [Interval(50.0, 70.0)], id='full_fl_node'),
        pytest.param(400.0, 0.72, [Interval(50.0, 50.0)], id='ragged_fl_node'),
        # Between the FL nodes, the mass must be valid in both.
        pytest.param(350.0, 0.72, [Interval(50.0, 50.0)], id='between_fl_nodes'),
        pytest.param(350.0, 0.78, [Interval(50.0, 70.0)], id='mach_in_all_cells'),
        pytest.param(350.0, 0.90, [], id='mach_outside'),
    ],
)
def test_valid_range_of_a_middle_axis(fl, mach, expected):
    assert ragged_3d().valid_range('mass', fl=fl, mach=mach) == expected


def test_valid_range_of_the_innermost_axis_of_three():
    # At FL 400, mass 70 starts at Mach 0.75, so it raises the lower bound.
    assert ragged_3d().valid_range('mach', fl=350.0, mass=60.0) == [
        Interval(0.75, 0.80)
    ]


def test_valid_range_of_the_outermost_axis_of_three():
    assert ragged_3d().valid_range('fl', mass=70.0, mach=0.72) == [
        Interval(300.0, 300.0)
    ]


def test_valid_range_has_a_gap_where_a_middle_node_does_not_cover():
    assert table_with_gap().valid_range('mass', mach=0.72) == [
        Interval(1.0, 1.0),
        Interval(3.0, 5.0),
    ]


def test_valid_range_between_two_nodes_intersects_their_gapped_ranges():
    """Between FL300 and FL400, a mass must be valid at both FLs. The first
    FL300 interval ends before the FL400 interval, so the intersection must
    move to the next FL300 interval and not stop."""
    interp = RaggedInterpolator(
        TABLE_WITH_GAPS_AT_TWO_FLS, ('fl', 'mass', 'mach'), ('fuel_flow',), 'test'
    )

    assert interp.valid_range('mass', fl=350.0, mach=0.72) == [
        Interval(1.0, 1.0),
        Interval(3.0, 4.0),
    ]


def test_valid_range_of_a_one_axis_table():
    df = table(('fl', 'tas'), [(0.0, 1.0), (10.0, 2.0)])
    interp = RaggedInterpolator(df, ('fl',), ('tas',), 'test')

    assert interp.valid_range('fl') == [Interval(0.0, 10.0)]


def test_valid_range_of_an_empty_table_is_empty():
    df = table(('fl', 'tas'), [])
    interp = RaggedInterpolator(df, ('fl',), ('tas',), 'test')

    assert interp.valid_range('fl') == []


@pytest.mark.parametrize(
    ('axis', 'coords', 'error', 'match'),
    [
        pytest.param('mach', {'fl': 10.0}, ValueError, 'no axis "mach"', id='axis'),
        pytest.param('mass', {}, TypeError, 'expected coordinates', id='coords'),
    ],
)
def test_valid_range_with_wrong_arguments_raises(axis, coords, error, match):
    with pytest.raises(error, match=match):
        ragged_2d().valid_range(axis, **coords)


@pytest.mark.parametrize(
    ('df', 'axes'),
    [
        pytest.param(RAGGED_2D, ('mass', 'fl'), id='2d'),
        pytest.param(RAGGED_3D, ('fl', 'mass', 'mach'), id='3d'),
        pytest.param(TABLE_WITH_GAP, ('mass', 'mach'), id='gapped'),
        pytest.param(
            TABLE_WITH_GAPS_AT_TWO_FLS, ('fl', 'mass', 'mach'), id='gapped_3d'
        ),
    ],
)
def test_valid_range_matches_evaluation(df, axes):
    """Check that evaluate and valid_range are consistent:
    * successful interpolation <-> valid
    * failed interpolation <-> invalid).
    """
    interp = RaggedInterpolator(df, axes, (df.columns[-1],), 'test')
    rng = np.random.default_rng(0)

    def samples(axis: str, n: int) -> list[float]:
        # Random values a little outside the axis range, and the grid values.
        values = df[axis]
        span = values.max() - values.min()
        low, high = values.min() - 0.1 * span, values.max() + 0.1 * span
        return [*rng.uniform(low, high, n), *values.unique()]

    for axis in interp.axes:
        others = [a for a in interp.axes if a != axis]
        fixed_samples = {a: samples(a, 10) for a in others}
        for _ in range(200):
            fixed = {a: float(rng.choice(fixed_samples[a])) for a in others}
            valid = interp.valid_range(axis, **fixed)
            ends = [end for interval in valid for end in interval]
            for value in [*samples(axis, 10), *ends]:
                coords = {**fixed, axis: value}
                in_range = any(low <= value <= high for low, high in valid)
                try:
                    interp(*(coords[a] for a in interp.axes))
                    evaluated = True
                except ValueError:
                    evaluated = False
                assert evaluated == in_range, coords

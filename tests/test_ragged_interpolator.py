"""Tests for `RaggedInterpolator`, on synthetic tables."""

import re

import pandas as pd
import pytest

from AEIC.performance.models.ragged import RaggedInterpolator


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

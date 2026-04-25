import numpy as np
import pytest

from AEIC.performance.types import ThrustMode, ThrustModeArray, ThrustModeValues


@pytest.fixture
def tm1():
    return ThrustModeValues(
        {
            ThrustMode.IDLE: 1.0,
            ThrustMode.APPROACH: 2.0,
            ThrustMode.CLIMB: 3.0,
            ThrustMode.TAKEOFF: 4.0,
        }
    )


@pytest.fixture
def tm2():
    return ThrustModeValues(
        {
            ThrustMode.IDLE: 0.5,
            ThrustMode.APPROACH: 1.5,
            ThrustMode.CLIMB: 2.5,
            ThrustMode.TAKEOFF: 3.5,
        }
    )


@pytest.fixture
def tm3():
    return ThrustModeValues({ThrustMode.IDLE: 10.0, ThrustMode.TAKEOFF: 40.0})


def test_thrust_mode_values_defaults(tm3):
    assert tm3[ThrustMode.APPROACH] == 0.0


@pytest.mark.parametrize(
    'name,lhs,rhs,expect_equal',
    [
        # Equal TMVs: same data, mutability flag does not matter.
        (
            'equal_same_data',
            ThrustModeValues({ThrustMode.IDLE: 1.0, ThrustMode.CLIMB: 2.0}),
            ThrustModeValues({ThrustMode.IDLE: 1.0, ThrustMode.CLIMB: 2.0}),
            True,
        ),
        # Differing values.
        (
            'unequal_values',
            ThrustModeValues({ThrustMode.IDLE: 1.0}),
            ThrustModeValues({ThrustMode.IDLE: 2.0}),
            False,
        ),
        # Differing key sets even with overlapping values.
        (
            'unequal_keys',
            ThrustModeValues({ThrustMode.IDLE: 1.0}),
            ThrustModeValues({ThrustMode.CLIMB: 1.0}),
            False,
        ),
        # Non-TMV operands always return False — the `isinstance` guard
        # short-circuits before any data comparison.
        ('vs_dict', ThrustModeValues(), {}, False),
        ('vs_zero_float', ThrustModeValues(), 0.0, False),
        ('vs_none', ThrustModeValues(), None, False),
    ],
)
def test_thrust_mode_values_comparison(name, lhs, rhs, expect_equal):
    assert (lhs == rhs) is expect_equal
    assert (lhs != rhs) is not expect_equal


def test_add_thrust_mode_values(tm1, tm2):
    result = tm1 + tm2
    assert result[ThrustMode.IDLE] == 1.5
    assert result[ThrustMode.APPROACH] == 3.5
    assert result[ThrustMode.CLIMB] == 5.5
    assert result[ThrustMode.TAKEOFF] == 7.5


def test_add_float_thrust_mode_values(tm1):
    result = 1.0 + tm1
    assert result[ThrustMode.IDLE] == 2.0
    assert result[ThrustMode.APPROACH] == 3.0
    assert result[ThrustMode.CLIMB] == 4.0
    assert result[ThrustMode.TAKEOFF] == 5.0


def test_mul_float_thrust_mode_values(tm1):
    result = 2.0 * tm1
    assert result[ThrustMode.IDLE] == 2.0
    assert result[ThrustMode.APPROACH] == 4.0
    assert result[ThrustMode.CLIMB] == 6.0
    assert result[ThrustMode.TAKEOFF] == 8.0


def test_div_thrust_mode_values(tm1, tm3):
    result = tm3 / tm1
    assert result[ThrustMode.IDLE] == 10.0
    assert result[ThrustMode.TAKEOFF] == 10.0


def test_div_float_thrust_mode_values(tm1):
    result = tm1 / 2.0
    assert result[ThrustMode.IDLE] == 0.5
    assert result[ThrustMode.APPROACH] == 1.0
    assert result[ThrustMode.CLIMB] == 1.5
    assert result[ThrustMode.TAKEOFF] == 2.0


def test_add_int_and_right_directions(tm1):
    """Cover the int branch of `__add__` (only float was tested) and the
    `tm + scalar` direction (only `scalar + tm` was tested via
    `__radd__`).
    """
    # Right-hand: `tm + scalar` goes through `__add__` directly.
    result = tm1 + 1
    assert result[ThrustMode.IDLE] == 2.0
    assert result[ThrustMode.TAKEOFF] == 5.0
    # Left-hand int: `int + tm` goes through `__radd__` → `__add__` int branch.
    result = 1 + tm1
    assert result[ThrustMode.IDLE] == 2.0
    assert result[ThrustMode.TAKEOFF] == 5.0


def test_mul_thrust_mode_values_pairwise(tm1, tm2):
    """The TMV × TMV branch of `__mul__` was untested (only float scalar)."""
    result = tm1 * tm2
    assert result[ThrustMode.IDLE] == 0.5
    assert result[ThrustMode.APPROACH] == 3.0
    assert result[ThrustMode.CLIMB] == 7.5
    assert result[ThrustMode.TAKEOFF] == 14.0


@pytest.mark.parametrize(
    'args, expected',
    [
        (
            (np.array([1.0, 2.0, 3.0, 4.0]),),
            {
                ThrustMode.IDLE: 1.0,
                ThrustMode.APPROACH: 2.0,
                ThrustMode.CLIMB: 3.0,
                ThrustMode.TAKEOFF: 4.0,
            },
        ),
        (
            (5.0,),
            {
                ThrustMode.IDLE: 5.0,
                ThrustMode.APPROACH: 5.0,
                ThrustMode.CLIMB: 5.0,
                ThrustMode.TAKEOFF: 5.0,
            },
        ),
        (
            (1.0, 2.0, 3.0, 4.0),
            {
                ThrustMode.IDLE: 1.0,
                ThrustMode.APPROACH: 2.0,
                ThrustMode.CLIMB: 3.0,
                ThrustMode.TAKEOFF: 4.0,
            },
        ),
    ],
    ids=['ndarray', 'scalar_float', 'four_positional'],
)
def test_thrust_mode_values_constructor_shapes(args, expected):
    tm = ThrustModeValues(*args)
    for mode, value in expected.items():
        assert tm[mode] == value


def test_thrust_mode_values_invalid_init():
    # Two-positional is not a recognized constructor shape.
    with pytest.raises(ValueError, match='Invalid initialization of ThrustModeValues'):
        ThrustModeValues(1.0, 2.0)


def test_thrust_mode_values_immutable_by_default(tm1):
    """Default `mutable=False` makes `__setitem__` raise — the SUT comment
    at types.py:71-73 explicitly warns that flipping the default would
    silently let downstream code mutate shared LTO data.
    """
    with pytest.raises(TypeError, match='frozen and cannot be modified'):
        tm1[ThrustMode.IDLE] = 99.0


def test_thrust_mode_values_copy_mutable_escape_hatch(tm1):
    """`copy(mutable=True)` is the documented escape hatch from a frozen
    instance — must produce an independent, writable copy.
    """
    writable = tm1.copy(mutable=True)
    writable[ThrustMode.IDLE] = 99.0
    assert writable[ThrustMode.IDLE] == 99.0
    assert tm1[ThrustMode.IDLE] == 1.0  # original is independent


def test_thrust_mode_values_broadcast(tm1):
    """`broadcast` projects per-mode values onto an array shaped like a
    `ThrustModeArray` — the SUT path used to lay out LTO data along a
    trajectory's per-point thrust-mode axis. Verify both that each cell
    carries its mode's value and that the output preserves the input
    shape.
    """
    modes = ThrustModeArray(
        np.array(
            [
                ThrustMode.IDLE.value,
                ThrustMode.CLIMB.value,
                ThrustMode.TAKEOFF.value,
                ThrustMode.APPROACH.value,
                ThrustMode.IDLE.value,
            ]
        )
    )
    out = tm1.broadcast(modes)
    assert out.shape == (5,)
    assert out.tolist() == [1.0, 3.0, 4.0, 2.0, 1.0]


def test_thrust_mode_array_rejects_invalid_values():
    """`ThrustModeArray.__post_init__` is the construction-time guard
    against arrays that mix valid mode codes with anything else. A
    regression that loosened the check would let downstream `as_enum`
    raise far from the source.
    """
    with pytest.raises(ValueError, match='invalid ThrustMode values'):
        ThrustModeArray(np.array([ThrustMode.IDLE.value, 9999]))


def test_or_thrust_mode_values(tm1, tm3):
    # `__or__` iterates `self._data.items()`, so the result's key set is
    # the lhs's key set — `__getitem__`'s default-to-zero would mask a
    # regression that flipped iteration to `other`. Pin the key sets
    # explicitly in addition to the value checks.
    result1 = tm3 | tm1
    assert set(iter(result1)) == {ThrustMode.IDLE, ThrustMode.TAKEOFF}
    assert result1[ThrustMode.IDLE] == 11.0
    assert result1[ThrustMode.TAKEOFF] == 44.0

    result2 = tm1 | tm3
    assert set(iter(result2)) == {
        ThrustMode.IDLE,
        ThrustMode.APPROACH,
        ThrustMode.CLIMB,
        ThrustMode.TAKEOFF,
    }
    assert result2[ThrustMode.IDLE] == 11.0
    assert result2[ThrustMode.APPROACH] == 2.0
    assert result2[ThrustMode.CLIMB] == 3.0
    assert result2[ThrustMode.TAKEOFF] == 44.0

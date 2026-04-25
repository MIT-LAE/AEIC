import numpy as np
import pytest

from AEIC.performance.types import ThrustMode, ThrustModeValues


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


def test_thrust_mode_values_comparison():
    assert ThrustModeValues() != 0.0


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


def test_or_thrust_mode_values(tm1, tm3):
    result1 = tm3 | tm1
    assert result1[ThrustMode.IDLE] == 11.0
    assert result1[ThrustMode.TAKEOFF] == 44.0
    result2 = tm1 | tm3
    assert result2[ThrustMode.IDLE] == 11.0
    assert result2[ThrustMode.APPROACH] == 2.0
    assert result2[ThrustMode.CLIMB] == 3.0
    assert result2[ThrustMode.TAKEOFF] == 44.0

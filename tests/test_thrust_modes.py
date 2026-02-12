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


def test_or_thrust_mode_values(tm1, tm3):
    result1 = tm3 | tm1
    assert result1[ThrustMode.IDLE] == 11.0
    assert result1[ThrustMode.TAKEOFF] == 44.0
    result2 = tm1 | tm3
    assert result2[ThrustMode.IDLE] == 11.0
    assert result2[ThrustMode.APPROACH] == 2.0
    assert result2[ThrustMode.CLIMB] == 3.0
    assert result2[ThrustMode.TAKEOFF] == 44.0

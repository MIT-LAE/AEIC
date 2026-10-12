"""Tests for the standard atmosphere helpers."""

import pytest

from AEIC.units import FEET_TO_METERS, KNOTS_TO_MPS
from AEIC.utils.standard_atmosphere import (
    cas_to_tas,
    crossover_altitude,
    speed_of_sound_at_altitude,
)


@pytest.mark.parametrize('cas', [50.0, 100.0, 150.0, 200.0])
def test_cas_to_tas_is_identity_at_sea_level(cas):
    """At sea level the reference and the local state coincide, so TAS == CAS
    within rounding differences.
    """
    assert float(cas_to_tas(cas, 0.0)) == pytest.approx(cas, rel=1e-5)


def test_cas_to_tas_at_altitude():
    """Check a reference point computed with
    https://aerotoolbox.com/airspeed-conversions/

    At 150 m/s CAS at 10 km gives 244.015 m/s within rounding differences due to
    truncated constants.
    """
    assert float(cas_to_tas(150.0, 10_000.0)) == pytest.approx(244.015, rel=1e-4)


def test_cas_to_tas_rejects_altitude_out_of_range():
    with pytest.raises(ValueError, match='Altitude out of range'):
        cas_to_tas(150.0, 25_001.0)


@pytest.mark.parametrize(
    ('cas_kts', 'mach', 'expected_ft'),
    [
        # Airspeed schedules of real PIANO climb and descent exports.
        (275.0, 0.74, 30680.0),
        (275.0, 0.78, 33271.0),
    ],
)
def test_crossover_altitude_matches_piano(cas_kts, mach, expected_ft):
    """PIANO rounds the crossover it reports to the foot."""
    altitude_m = float(crossover_altitude(cas_kts * KNOTS_TO_MPS, mach))
    assert altitude_m / FEET_TO_METERS == pytest.approx(expected_ft, abs=2.0)


@pytest.mark.parametrize(
    ('cas', 'mach'),
    [
        (140.0, 0.75),
        # Above the tropopause.
        (125.0, 0.80),
    ],
)
def test_crossover_altitude_equates_cas_and_mach(cas, mach):
    altitude_m = float(crossover_altitude(cas, mach))
    assert float(cas_to_tas(cas, altitude_m)) == pytest.approx(
        mach * float(speed_of_sound_at_altitude(altitude_m)), rel=1e-6
    )

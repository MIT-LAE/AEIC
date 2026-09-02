"""Tests for the standard atmosphere helpers."""

import pytest

from AEIC.utils.standard_atmosphere import cas_to_tas


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

"""Unit conversion factors for various measurements."""

from AEIC.constants import g0

FEET_TO_METERS = 0.3048
"""Unit conversion factor for feet to meters."""

METERS_TO_FEET = 3.28084
"""Unit conversion factor for meters to feet."""

METERS_TO_FL = METERS_TO_FEET / 100
"""Unit conversion factor for meters to flight level."""

FL_TO_METERS = 100 * FEET_TO_METERS
"""Unit conversion factor for flight level to meters."""

STATUTE_MILES_TO_KM = 1.609344
"""Unit conversion factor for statute miles to kilometers."""

KNOTS_TO_MPS = 0.514444
"""Unit conversion factor for knots to m/s."""

MPS_TO_KNOTS = 1 / KNOTS_TO_MPS
"""Unit conversion factor for m/s to knots."""

NAUTICAL_MILES_TO_METERS = 1852
"""Unit conversion factor for nautical miles to meters."""

MINUTES_TO_SECONDS = 60
"""Unit conversion factor for minutes to seconds."""

FPM_TO_MPS = FEET_TO_METERS / MINUTES_TO_SECONDS
"""Unit conversion factor for feet per minute to meters per second."""

POUNDS_TO_KG = 0.45359237
"""Unit conversion factor for pounds to kilograms."""

POUNDS_FORCE_TO_NEWTONS = POUNDS_TO_KG * g0
"""Unit conversion factor for pounds-force to newtons."""

POUNDS_PER_HOUR_TO_KG_PER_S = POUNDS_TO_KG / 3600
"""Unit conversion factor for pounds per hour to kilograms per second."""

PPM = 1.0e-6
"""Parts per million as a unitless fraction."""

KG_TO_GRAMS = 1000.0
"""Unit conversion factor for kilograms to grams."""

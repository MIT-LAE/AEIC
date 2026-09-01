"""Reader for PIANO aircraft performance text exports.

PIANO writes three files for an aircraft: a cruise table sweeping mass,
altitude and Mach number, plus climb and descent files holding one detail
block per initial mass. This package parses all three into SI tables ready to
be written to a performance model TOML file.
"""

from AEIC.parsers.piano_reader.piano_data import PianoData, PianoOverrides

__all__ = ['PianoData', 'PianoOverrides']

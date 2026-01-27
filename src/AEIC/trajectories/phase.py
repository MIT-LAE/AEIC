"""Trajectories are divided into a sequence of "phases" (e.g., climb, cruise,
descent). Each phase has a corresponding field in the `Trajectory` class that
records the number of points in that phase for each trajectory.

Some phases are expected to be simulated by all performance models (climb,
cruise, descent) while others are optional (taxi, takeoff, approach, idle) and
may only be simulated by more detailed models or may be populated from
time-in-mode values. For uniformity of representation, these optional phases
are included as "normal" points in the trajectory.

Each `Trajectory` records the number of points in each phase using fields with
names prefixed by `n_` (e.g., `n_climb`, `n_cruise`, `n_descent`).
"""

from enum import Enum, auto

import numpy as np

from AEIC.storage.field_sets import Dimension, Dimensions, FieldMetadata

PHASE_FIELD_PREFIX = 'n_'


class FlightPhase(Enum):
    """Flight phases known to AEIC."""

    IDLE_ORIGIN = auto()
    TAXI_ORIGIN = auto()
    TAKEOFF = auto()
    CLIMB = auto()
    CRUISE = auto()
    DESCENT = auto()
    APPROACH = auto()
    TAXI_DESTINATION = auto()
    IDLE_DESTINATION = auto()

    @property
    def field_name(self):
        """Trajectory point count field name for this flight phase."""
        return PHASE_FIELD_PREFIX + self.name.lower()

    @property
    def method_name(self):
        """Method name used in trajectory builders for this flight phase."""
        return 'fly_' + self.name.lower()

    @property
    def field_label(self):
        """Human-readable label for this flight phase."""
        return self.name.lower().replace('_', ' at ')

    @classmethod
    def from_field_name(cls, field_name: str):
        """Parse trajectory point count field name to get flight phase."""
        return cls[field_name[len(PHASE_FIELD_PREFIX) :].upper()]


REQUIRED_PHASES = {
    FlightPhase.CLIMB,
    FlightPhase.CRUISE,
    FlightPhase.DESCENT,
}
"""Flight phases we expect all performance models to simulate."""


FlightPhases = dict[FlightPhase, int]
"""Type alias for trajectory point counts in each flight phase."""


PHASE_FIELDS = {
    phase.field_name: FieldMetadata(
        dimensions=Dimensions(Dimension.TRAJECTORY),
        field_type=np.int32,
        description=f'Number of points in {phase.field_label} phase',
        units='count',
        required=(phase in REQUIRED_PHASES),
        default=0,
    )
    for phase in FlightPhase
}
"""Convenience dictionary of field metadata for all flight phases.

This is used in the "base" field set for trajectory datasets to add the phase
point count metadata variables to every trajectory."""

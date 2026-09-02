"""This module implements a performance model built from PIANO aircraft
performance exports.

PIANO cruise data is swept over Mach number as well as flight level and mass, and
every phase carries thrust, drag and trajectory columns.
"""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from typing import Any, Literal

from pydantic import PositiveFloat, model_validator

from AEIC.performance.types import (
    AircraftState,
    Performance,
    SimpleFlightRules,
    TableInput,
)

from .base import BasePerformanceModel, PerformanceTableInput


class PianoPerformanceModel(BasePerformanceModel[SimpleFlightRules]):
    """Performance model built from PIANO climb, cruise and descent exports."""

    model_type: Literal['piano']
    """Performance model type discriminator."""

    climb_flight_performance: PerformanceTableInput
    """Climb performance table."""

    cruise_flight_performance: PerformanceTableInput
    """Cruise performance table, swept over Mach number."""

    descent_flight_performance: PerformanceTableInput
    """Descent performance table."""

    cruise_reference_mach: TableInput | None = None
    """Reference Mach numbers PIANO labels in the cruise sweep."""

    descent_idle_thrust: TableInput | None = None
    """Altitude below which each descent block flies at idle thrust."""

    operating_empty_mass_kg: PositiveFloat
    """Operating empty mass [kg].

    PIANO exports do not contain one, and unlike BADA the sweep gives no basis
    for deriving it from the lowest tabulated mass, so it must be supplied.
    """

    @model_validator(mode='before')
    @classmethod
    def require_operating_empty_mass(cls, data: Any) -> Any:
        """Report a missing operating empty mass in the terms of the file.

        Pydantic would report the field as required, which reads as an error
        in the exporter rather than a value the user has to look up.
        """
        if isinstance(data, dict) and not any(
            key.lower() == 'operating_empty_mass_kg' for key in data
        ):
            raise ValueError(
                'PIANO exports do not contain an operating empty mass, so '
                '"operating_empty_mass_kg" must be set in the performance '
                'model file.'
            )
        return data

    @property
    def empty_mass(self) -> float:
        """Operating empty mass."""
        return self.operating_empty_mass_kg

    @property
    def maximum_mass(self) -> float:
        """Maximum aircraft mass across the three phase tables."""
        return max(
            max(self.climb_flight_performance.column('mass')),
            max(self.cruise_flight_performance.column('mass')),
            max(self.descent_flight_performance.column('mass')),
        )

    def evaluate_impl(
        self, state: AircraftState, rules: SimpleFlightRules
    ) -> Performance:
        raise NotImplementedError('PIANO performance evaluation not yet implemented.')

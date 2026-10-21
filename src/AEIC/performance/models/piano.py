"""This module implements a performance model built from PIANO aircraft
performance exports.

PIANO cruise data is swept over Mach number as well as flight level and mass, and
every phase carries thrust, drag and trajectory columns.

The model evaluates performance by linear interpolation over ragged tables:
climb and descent in mass and flight level, and cruise in flight level, mass
and Mach number. Queries outside the data raise `ValueError`.
"""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

from pydantic import PositiveFloat, PrivateAttr, model_validator

from AEIC.performance.models.base import (
    BasePerformanceModel,
    PerformanceTableInput,
    ROCDFilter,
)
from AEIC.performance.models.ragged import RaggedPerformanceTable
from AEIC.performance.types import (
    AircraftState,
    CruiseMach,
    Performance,
    SimpleFlightRules,
    TableInput,
)
from AEIC.utils.models import CIBaseModel

PianoFlightRules = SimpleFlightRules | CruiseMach
"""Flight rules accepted by the PIANO performance model."""


class HoldPerformance(CIBaseModel):
    """Hold performance at a reference condition."""

    fuel_flow: PositiveFloat
    """Total aircraft fuel flow in a hold [kg/s]."""


class PianoPerformanceModel(BasePerformanceModel[PianoFlightRules]):
    """Performance model built from PIANO climb, cruise and descent exports.

    Climb and descent use `SimpleFlightRules`. Cruise uses `CruiseMach`,
    because PIANO cruise performance depends on Mach number."""

    FLIGHT_RULES_CLASS: ClassVar[type] = PianoFlightRules  # type: ignore[assignment]

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

    hold_flight_performance: HoldPerformance
    """Hold performance.

    PIANO can simulate hold performance directly, unlike BADA/LegacyPerformanceModel
    which uses cruise performance at FL30. The representative datapoint is the initial
    fuel flow for a hold at 3000 ft with 'normal landing' mass and minimizing drag.
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

    @model_validator(mode='before')
    @classmethod
    def require_hold_flight_performance(cls, data: Any) -> Any:
        """Report a missing hold fuel flow with the option that supplies it.

        Pydantic would report the field as required, which does not say where
        the value comes from.
        """
        if isinstance(data, dict) and not any(
            key.lower() == 'hold_flight_performance' for key in data
        ):
            raise ValueError(
                'PIANO exports do not contain a hold fuel flow, so '
                '"hold_flight_performance.fuel_flow" must be set in the '
                'performance model file. Supply it with --hold-fuel-flow-lbperh.'
            )
        return data

    _climb_table: RaggedPerformanceTable = PrivateAttr()
    _cruise_table: RaggedPerformanceTable = PrivateAttr()
    _descent_table: RaggedPerformanceTable = PrivateAttr()

    @model_validator(mode='after')
    def build_performance_tables(self) -> Self:
        """Build the interpolation tables from the "flight_performance"
        sections."""
        self._climb_table = RaggedPerformanceTable.from_input(
            self.climb_flight_performance, ROCDFilter.POSITIVE, ('mass', 'fl')
        )
        self._cruise_table = RaggedPerformanceTable.from_input(
            self.cruise_flight_performance, ROCDFilter.ZERO, ('fl', 'mass', 'mach')
        )
        self._descent_table = RaggedPerformanceTable.from_input(
            self.descent_flight_performance, ROCDFilter.NEGATIVE, ('mass', 'fl')
        )
        return self

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
        self, state: AircraftState, rules: PianoFlightRules
    ) -> Performance:
        """Interpolate performance in the table for the flight phase.

        Raises:
            ValueError: If the rules are `SimpleFlightRules.CRUISE`, or if the
                state is outside the data.
        """
        match rules:
            case SimpleFlightRules.CLIMB:
                return self._climb_table.interpolate(state)
            case SimpleFlightRules.DESCEND:
                return self._descent_table.interpolate(state)
            case CruiseMach(mach=mach):
                return self._cruise_table.interpolate(state, mach=mach)
            case SimpleFlightRules.CRUISE:
                raise ValueError(
                    'PIANO cruise performance depends on Mach number. '
                    'Use CruiseMach flight rules instead of SimpleFlightRules.CRUISE.'
                )

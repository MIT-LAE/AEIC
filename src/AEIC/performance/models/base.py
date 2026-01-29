from abc import ABC, abstractmethod
from typing import ClassVar, Self

from pydantic import PositiveInt, model_validator

from AEIC.performance import (
    AircraftState,
    LTOPerformance,
    Performance,
    SimpleFlightRules,
    Speeds,
)
from AEIC.performance.utils.apu import APU, find_apu
from AEIC.utils.models import CIBaseModel
from AEIC.utils.types import AircraftClass
from AEIC.utils.units import FEET_TO_METERS


class BasePerformanceModel[RulesT = SimpleFlightRules](CIBaseModel, ABC):
    """Base class for aircraft performance models.

    This a generic class parameterized by the type of flight rules accepted by
    the class's :meth:`evaluate` method. By default, this is
    :class:`SimpleFlightRules <AEIC.performance.types.SimpleFlightRules>`, but
    subclasses can override this to specify more sophisticated flight rule
    types.

    (This class uses the pattern of splitting the :meth:`evaluate` method into
    two steps to allow for type checking of the flight rules input before
    calling the actual implementation defined in subclasses. Similarly, it
    duplicates the flight rules class generic parameter as a class variable to
    allow for both static and runtime checking of the flight rules type in
    subclasses.)

    """

    aircraft_name: str
    """Aircraft name (e.g., "A320")."""

    aircraft_class: AircraftClass
    """Aircraft class (e.g., wide or narrow body)."""

    maximum_altitude_ft: PositiveInt
    """Aircraft maximum altitude in feet."""

    @property
    def maximum_altitude(self) -> float:
        """Aircraft maximum altitude in meters."""
        return self.maximum_altitude_ft * FEET_TO_METERS

    maximum_payload_kg: PositiveInt
    """Aircraft maximum payload in kilograms."""

    @property
    def maximum_payload(self) -> float:
        """Aircraft maximum payload in kg."""
        return self.maximum_payload_kg

    number_of_engines: PositiveInt
    """Number of engines."""

    apu_name: str | None = None
    """Optional APU name. This is used to look up APU data from the APU
    database."""

    speeds: Speeds | None
    """Optional speed data."""

    lto_performance: LTOPerformance | None
    """Optional LTO performance data."""

    FLIGHT_RULES_CLASS: ClassVar[type] = SimpleFlightRules
    """The class type for flight rules accepted by this performance model."""

    @model_validator(mode='after')
    def load_apu_data(self) -> Self:
        """Load APU data from APU database after successful instance creation."""
        self._apu: APU | None = None
        if self.apu_name is not None:
            self._apu = find_apu(self.apu_name)
        return self

    @property
    def apu(self) -> APU | None:
        """APU data associated with the performance model.

        This is loaded from the APU database based on the ``apu_name`` field
        using the ``AEIC.performance.utils.apu.find_apu`` function."""
        return self._apu

    def evaluate(self, state: AircraftState, rules: RulesT) -> Performance:
        """Evaluate aircraft performance based on the given state and flight
        rules.

        The implementation of this method is split into two steps
        (``_evaluate_checked``, defined here, and ``evaluate_impl``, defined in
        subclasses) to ensure that the flight rules type is checked before
        evaluation. The type declaration on ``evaluate`` uses the generic type
        variable ``RulesT`` to allow subclasses to specify more specific flight
        rules types.
        """
        return self._evaluate_checked(state, rules)

    def _evaluate_checked(self, state: AircraftState, rules) -> Performance:
        """Check flight rules type and evaluate performance."""
        if not isinstance(rules, self.FLIGHT_RULES_CLASS):
            raise TypeError(
                f'Expected flight rules of type {self.FLIGHT_RULES_CLASS}, '
                f'got {type(rules)}'
            )
        return self.evaluate_impl(state, rules)

    @abstractmethod
    def evaluate_impl(self, state: AircraftState, rules: RulesT) -> Performance:
        """Actual performance evaluation function, to be implemented in
        subclasses."""
        ...

    @property
    @abstractmethod
    def empty_mass(self) -> float:
        """Aircraft empty mass, to be implemented by subclasses."""
        ...

    @property
    @abstractmethod
    def maximum_mass(self) -> float:
        """Aircraft maximum mass, to be implemented by subclasses."""
        ...

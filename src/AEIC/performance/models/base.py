from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar, Self

from pydantic import ConfigDict, PositiveInt, model_validator

from AEIC.config import config
from AEIC.performance.utils.apu import APU, find_apu
from AEIC.performance.utils.edb import EDBEntry
from AEIC.types import (
    AircraftClass,
    AircraftState,
    LTOPerformance,
    ModeValues,
    Performance,
    SimpleFlightRules,
    Speeds,
    ThrustMode,
)
from AEIC.units import FEET_TO_METERS
from AEIC.utils.models import CIBaseModel


class LTOModeDataInput(CIBaseModel):
    """LTO data for a given flight phase."""

    # TODO: Better docstrings.
    thrust_frac: float
    fuel_kgs: float
    EI_NOx: float
    EI_HC: float
    EI_CO: float


class LTOPerformanceInput(CIBaseModel):
    """LTO performance data as represented in configuration file."""

    # TODO: Make source an enum? EDB or in-file?
    # TODO: Docstrings.
    source: str
    ICAO_UID: str
    rated_thrust: float
    mode_data: dict[ThrustMode, LTOModeDataInput]

    def convert(self) -> LTOPerformance:
        """Create internal representation of LTO performance data from input
        data."""

        def extract(attr_name: str, scale: float = 1.0) -> ModeValues:
            return ModeValues(
                {m: getattr(self.mode_data[m], attr_name) * scale for m in ThrustMode}
            )

        return LTOPerformance(
            source=self.source,
            ICAO_UID=self.ICAO_UID,
            rated_thrust=self.rated_thrust,
            fuel_flow=extract('fuel_kgs'),
            EI_NOx=extract('EI_NOx'),
            EI_HC=extract('EI_HC'),
            EI_CO=extract('EI_CO'),
            thrust_pct=extract('thrust_frac', scale=100.0),
        )

    @classmethod
    def from_internal(cls, lto: LTOPerformance) -> Self:
        """Create LTOPerformanceInput from internal LTOPerformance data."""

        mode_data = {
            m: LTOModeDataInput(
                thrust_frac=lto.thrust_pct[m] / 100.0,
                fuel_kgs=lto.fuel_flow[m],
                EI_NOx=lto.EI_NOx[m],
                EI_HC=lto.EI_HC[m],
                EI_CO=lto.EI_CO[m],
            )
            for m in ThrustMode
        }

        return cls(
            source=lto.source,
            ICAO_UID=lto.ICAO_UID,
            rated_thrust=lto.rated_thrust,
            mode_data=mode_data,
        )


class BasePerformanceModel[RulesT = SimpleFlightRules](CIBaseModel, ABC):
    """Base class for aircraft performance models.

    This a generic class parameterized by the type of flight rules accepted by
    the class's :meth:`evaluate` method. By default, this is
    :class:`SimpleFlightRules <AEIC.types.SimpleFlightRules>`, but
    subclasses can override this to specify more sophisticated flight rule
    types.

    (This class uses the pattern of splitting the :meth:`evaluate` method into
    two steps to allow for type checking of the flight rules input before
    calling the actual implementation defined in subclasses. Similarly, it
    duplicates the flight rules class generic parameter as a class variable to
    allow for both static and runtime checking of the flight rules type in
    subclasses.)

    """

    model_config = ConfigDict(frozen=True)
    """Configuration is frozen after creation."""

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

    lto_performance: LTOPerformanceInput | None
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

    @cached_property
    def edb(self) -> EDBEntry:
        if self.lto_performance is None:
            raise ValueError(
                'LTO performance data not available, so no engine ID for EDB lookup.'
            )
        return EDBEntry.get_engine(config.engine_file, self.lto_performance.ICAO_UID)

    @property
    def apu(self) -> APU | None:
        """APU data associated with the performance model.

        This is loaded from the APU database based on the ``apu_name`` field
        using the ``AEIC.performance.utils.apu.find_apu`` function."""
        return self._apu

    @cached_property
    def lto(self) -> LTOPerformance:
        """LTO performance data associated with the performance model.

        Raises:
            ValueError: If LTO performance data is not available.
        """
        if self.lto_performance is None:
            raise ValueError('LTO performance data not available.')
        return self.lto_performance.convert()

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

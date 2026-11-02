"""This module implements a performance model built from PIANO aircraft
performance exports.

PIANO cruise data is swept over Mach number as well as flight level and mass, and
every phase carries thrust, drag and trajectory columns.

The model evaluates performance by linear interpolation over ragged tables:
climb and descent in mass and flight level, and cruise in flight level, mass
and Mach number.

The climb and descent mass axis is the initial mass of each PIANO climb or descent
because PIANO reports a climb/descent profile and not isolated performance points.
The performance interpolation is in (initial_mass, fl) space.
Before it queries these tables, the model converts the current mass of the query to
the corresponding initial mass that would have led to the current mass of the aircraft
via `PianoClimbDescentMassAxis`. Queries outside the data raise `ValueError`.

Cruise also interpolates the rate of climb at maximum climb (MCL) thrust, and raises
`ValueError` if it is below `PianoConstantMachCruise.min_rocd_mcl_mach`.
"""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby, pairwise
from operator import itemgetter
from typing import Any, ClassVar, Literal, Self

import numpy as np
from pydantic import PositiveFloat, PrivateAttr, model_validator

from AEIC.performance.models.base import (
    BasePerformanceModel,
    PerformanceTableInput,
    ROCDFilter,
)
from AEIC.performance.models.ragged import (
    RaggedPerformanceTable,
    query_fl,
)
from AEIC.performance.types import (
    AircraftState,
    Performance,
    PianoConstantMachCruise,
    SimpleFlightRules,
    TableInput,
)
from AEIC.storage import FlightPhase
from AEIC.units import METERS_TO_FL
from AEIC.utils.models import CIBaseModel

PianoFlightRules = SimpleFlightRules | PianoConstantMachCruise
"""Flight rules accepted by the PIANO performance model."""


class HoldPerformance(CIBaseModel):
    """Hold performance at a reference condition."""

    fuel_flow: PositiveFloat
    """Total aircraft fuel flow in a hold [kg/s]."""


@dataclass(frozen=True)
class _PianoClimbDescentBurn:
    """Cumulative burn of one PIANO climb or descent, as a function of flight
    level."""

    initial_mass: float
    """Initial mass of the climb or descent [kg]."""

    fl: np.ndarray
    """Flight levels of the rows, sorted."""

    cumulative_burn: np.ndarray
    """Cumulative burn at each row [kg]."""

    def covers(self, fl: float) -> bool:
        """Whether `fl` is inside the flight levels of the rows."""
        return bool(self.fl[0] <= fl <= self.fl[-1])

    def current_mass(self, fl: float) -> float:
        """Mass at `fl`. The caller must check `covers` first."""
        return self.initial_mass - float(np.interp(fl, self.fl, self.cumulative_burn))


class PianoClimbDescentMassAxis:
    """Find the corresponding initial mass (climb or descent) for a profile
    that would have led to the current mass at the requested FL in a PIANO
    climb or descent table.

    PIANO flies one climb or descent from each initial mass in the table, so a
    row's values are for the current mass `initial_mass - burn`. At the query
    flight level, the conversion finds the two neighbouring initial masses
    whose current masses are on each side of the query mass. It then interpolates
    the initial mass between them."""

    def __init__(self, burns: list[_PianoClimbDescentBurn], phase: FlightPhase):
        """Make the mass axis for one climb or descent table.

        Args:
            burns: Cumulative burn of each climb or descent, sorted by
                initial mass from light to heavy.
            phase: Flight phase of the table. Used in error messages.

        Raises:
            ValueError: If at some row FL a heavier initial mass does not give
                a heavier current mass than the lighter one next to it.
        """
        self._burns = burns
        self._phase = phase.name.lower()
        self._check_current_mass_order()

    def _check_current_mass_order(self) -> None:
        """Check that a heavier initial mass stays heavier than its neighbour.

        Otherwise two initial masses can give the same current mass, and the
        initial mass of a query is not unique. Each burn is linear between its
        rows, so it is sufficient to check at the row FLs of the two burns."""
        for low, high in pairwise(self._burns):
            for fl in sorted({*low.fl, *high.fl}):
                if not (low.covers(fl) and high.covers(fl)):
                    continue
                current_mass_low = low.current_mass(fl)
                current_mass_high = high.current_mass(fl)
                if current_mass_high <= current_mass_low:
                    raise ValueError(
                        f'{self._phase} table: at fl={fl}, the {self._phase} from '
                        f'initial mass {high.initial_mass} has a current mass of '
                        f'{current_mass_high}, which is not above the '
                        f'{current_mass_low} of the {self._phase} from initial '
                        f'mass {low.initial_mass}'
                    )

    @classmethod
    def from_input(cls, ptin: PerformanceTableInput, phase: FlightPhase) -> Self:
        """Build from the `fl`, `mass` (initial mass) and `burn` columns of a
        table input.

        Raises:
            ValueError: If a column is missing.
        """
        rows = sorted(zip(ptin.column('mass'), ptin.column('fl'), ptin.column('burn')))
        burns = []
        for mass, rows_of_mass in groupby(rows, key=itemgetter(0)):
            _, fl, burn = zip(*rows_of_mass)
            burns.append(_PianoClimbDescentBurn(mass, np.array(fl), np.array(burn)))
        return cls(burns, phase)

    def find_corresponding_initial_state(self, state: AircraftState) -> AircraftState:
        """Copy of `state` with its state.aircraft_mass converted to the
        initial climb/descent mass that would have led to the current aircraft mass at
        the current FL.

        Used to query the climb/descent performance tables which are functions of
        (initial_mass, fl).

        Passes 'min' and 'max' through unchanged. The table resolves them to
        the lightest or heaviest initial mass that is valid at the flight
        level."""
        if state.aircraft_mass == 'min' or state.aircraft_mass == 'max':
            return state

        initial_mass = self.get_corresponding_initial_mass(
            state.aircraft_mass, query_fl(state.altitude)
        )
        return replace(state, aircraft_mass=initial_mass)

    def get_corresponding_initial_mass(self, mass_at_fl: float, fl: float) -> float:
        """Initial climb/descent mass that would have led to the climb/descent
        that matches the mass_at_fl, fl combination.

        Used to query the climb/descent performance table which are functions of
        (initial_mass, fl).

        Raises:
            ValueError: If no two neighbouring initial masses cover `mass_at_fl` at
                `fl`.
        """
        # For the FL the aircraft is at, get the (initial_mass, mass_at_fl)
        # pairs that are tabulated in the performance table
        table_masses_at_fl = self._get_table_masses_at_fl(fl)

        # If one of the table_mass_at_fl is exactly the current mass
        # then we can just return the corresponding initial mass that led to
        # this climb/descent.
        for table_initial_mass, table_mass_at_fl in table_masses_at_fl:
            if table_mass_at_fl == mass_at_fl:
                return table_initial_mass

        # Interpolate initial_mass in mass_at_fl space: we want to find the
        # corresponding initial_mass that would have led to the mass_at_fl.
        # This is done by linearly interpolating the tabulated initial masses
        # weighted by the tabulated mass_at_fl values
        for point_low, point_high in pairwise(table_masses_at_fl):
            table_initial_mass_low, table_mass_at_fl_low = point_low
            table_initial_mass_high, table_mass_at_fl_high = point_high

            low_covers_fl = table_mass_at_fl_low is not None
            high_covers_fl = table_mass_at_fl_high is not None

            if (
                low_covers_fl
                and high_covers_fl
                and table_mass_at_fl_low <= mass_at_fl <= table_mass_at_fl_high
            ):
                weight = (mass_at_fl - table_mass_at_fl_low) / (
                    table_mass_at_fl_high - table_mass_at_fl_low
                )
                return table_initial_mass_low + weight * (
                    table_initial_mass_high - table_initial_mass_low
                )

        # The requested mass_at_fl is not in the bounds of the tabulated mass_at_fl
        if all(table_mass_at_fl is None for _, table_mass_at_fl in table_masses_at_fl):
            raise ValueError(
                f'{self._phase} table: no tabulated initial masses reaches fl={fl}'
            )

        # List the tabulated mass_at_fl of each initial mass
        tabulated = ', '.join(
            f'{"none" if table_mass_at_fl is None else table_mass_at_fl} '
            f'(from initial mass = {table_initial_mass})'
            for table_initial_mass, table_mass_at_fl in table_masses_at_fl
        )
        raise ValueError(
            f'{self._phase} table: current mass {mass_at_fl} at fl={fl} is not '
            f'in range of the tabulated data. '
            f'Tabulated masses available at this fl: {tabulated}'
        )

    def _get_table_masses_at_fl(self, fl: float) -> list[tuple[float, float | None]]:
        """`(initial mass, mass_at_fl)` of each burn at `fl`, with None for
        a burn that does not cover `fl`."""
        return [
            (burn.initial_mass, burn.current_mass(fl) if burn.covers(fl) else None)
            for burn in self._burns
        ]


class PianoPerformanceModel(BasePerformanceModel[PianoFlightRules]):
    """Performance model built from PIANO climb, cruise and descent exports.

    Climb and descent use `SimpleFlightRules`. Cruise uses `PianoConstantMachCruise`,
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

    PIANO can simulate a hold directly. `LegacyPerformanceModel` instead uses
    the cruise fuel flow at the lowest tabulated cruise flight level and the
    minimum mass. The representative datapoint is the initial fuel flow for a
    hold at 3000 ft with 'normal landing' mass and minimizing drag.

    PIANO exports do not contain this value, so it must be supplied.
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
    """Climb table. The mass axis is the initial mass of each PIANO climb, so
    query it through `_climb_mass`."""

    _climb_mass: PianoClimbDescentMassAxis = PrivateAttr()
    """Converts the current mass of a climb query to an initial mass."""

    _cruise_table: RaggedPerformanceTable = PrivateAttr()
    """Cruise table."""

    _descent_table: RaggedPerformanceTable = PrivateAttr()
    """Descent table. The mass axis is the initial mass of each PIANO descent, so
    query it through `_descent_mass`."""

    _descent_mass: PianoClimbDescentMassAxis = PrivateAttr()
    """Converts the current mass of a descent query to an initial mass."""

    @model_validator(mode='after')
    def build_performance_tables(self) -> Self:
        """Build the interpolation tables from the "flight_performance"
        sections."""
        self._climb_table = RaggedPerformanceTable.from_input(
            self.climb_flight_performance,
            ROCDFilter.POSITIVE,
            ('mass', 'fl'),
        )
        self._cruise_table = RaggedPerformanceTable.from_input(
            self.cruise_flight_performance,
            ROCDFilter.ZERO,
            ('fl', 'mass', 'mach'),
            extra_outputs=('rocd_mcl_fix_mach',),
        )
        self._descent_table = RaggedPerformanceTable.from_input(
            self.descent_flight_performance,
            ROCDFilter.NEGATIVE,
            ('mass', 'fl'),
        )
        self._climb_mass = PianoClimbDescentMassAxis.from_input(
            self.climb_flight_performance, FlightPhase.CLIMB
        )
        self._descent_mass = PianoClimbDescentMassAxis.from_input(
            self.descent_flight_performance, FlightPhase.DESCENT
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

    @property
    def hold_fuel_flow(self) -> float:
        """Hold fuel flow [kg/s], from `hold_flight_performance`.

        PIANO estimates this value directly."""
        return self.hold_flight_performance.fuel_flow

    @property
    def maximum_rocd(self) -> float:
        return max(
            max(self._climb_table.df.rocd),
            max(self._cruise_table.df.rocd),
            max(self._descent_table.df.rocd),
        )

    @property
    def minimum_tas(self) -> float:
        return min(
            min(self._climb_table.df.tas),
            min(self._cruise_table.df.tas),
            min(self._descent_table.df.tas),
        )

    def evaluate_impl(
        self, state: AircraftState, rules: PianoFlightRules
    ) -> Performance:
        """Interpolate performance in the table for the flight phase.

        Raises:
            ValueError: If the rules are `SimpleFlightRules.CRUISE`, if the
                state is outside the data, or if the cruise climb potential is
                below `PianoConstantMachCruise.min_rocd_mcl_mach`.
        """
        match rules:
            case SimpleFlightRules.CLIMB:
                # Interpolate in (initial_mass, fl) space
                return self._climb_table.interpolate(
                    self._climb_mass.find_corresponding_initial_state(state)
                )
            case SimpleFlightRules.DESCEND:
                # Interpolate in (initial_mass, fl) space
                return self._descent_table.interpolate(
                    self._descent_mass.find_corresponding_initial_state(state)
                )
            case PianoConstantMachCruise():
                return self._evaluate_cruise(state, rules)
            case SimpleFlightRules.CRUISE:
                raise ValueError(
                    'PIANO cruise performance depends on Mach number. '
                    'Use PianoConstantMachCruise flight rules instead of '
                    'SimpleFlightRules.CRUISE.'
                )

    def _evaluate_cruise(
        self, state: AircraftState, rules: PianoConstantMachCruise
    ) -> Performance:
        """Interpolate cruise performance and check the climb potential."""
        values = self._cruise_table.interpolate_values(state, mach=rules.mach)

        # A negative climb potential means that the drag is higher than the
        # MCL thrust, so the aircraft cannot hold level flight at this Mach.
        rocd_mcl = values['rocd_mcl_fix_mach']
        minimum = rules.min_rocd_mcl_mach
        if rocd_mcl < minimum:
            raise ValueError(
                f'cruise at FL{state.altitude * METERS_TO_FL:.0f}, '
                f'mass {state.aircraft_mass} kg, Mach {rules.mach}: rate of '
                f'climb at MCL thrust {rocd_mcl:.3f} m/s is below the minimum '
                f'{minimum} m/s'
            )

        return Performance(
            true_airspeed=values['tas'],
            rate_of_climb=values['rocd'],
            fuel_flow=values['fuel_flow'],
        )

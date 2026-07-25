"""This module implements a performance model for PIANO-derived performance
data.

The table shape is the same as the legacy (BADA-derived) performance model
(see :mod:`AEIC.performance.models.legacy`): performance data is split into
climb, cruise and descent sections, and evaluation is done by bilinear
interpolation in flight level and aircraft mass. This is a separate model
type rather than reusing the legacy model because PIANO can offer richer
data than BADA PTF files provide:

- BADA PTF files always report exactly 2 or 3 masses (low/nominal/high) for
  climb/cruise and exactly 1 (nominal) for descent; PIANO can report more.
- BADA PTF files only ever report a single nominal-mass fuel_flow/ROCD value
  for climb/descent; PIANO can report genuine per-mass variation (a heavier
  aircraft burns more fuel climbing to the same altitude).

The legacy model's validation is intentionally strict about both of these,
matching what BADA PTF data actually looks like. This model relaxes both,
since a PIANO-derived table can legitimately have more resolution."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Self

import pandas as pd
from pydantic import PrivateAttr, model_validator

from AEIC.performance.models.legacy import (
    Interpolator,
    PerformanceTableInput,
    ROCDFilter,
)
from AEIC.performance.types import AircraftState, Performance, SimpleFlightRules
from AEIC.units import METERS_TO_FL

from .base import BasePerformanceModel


@dataclass
class PianoPerformanceTable:
    """Aircraft performance data table for a PIANO-derived performance
    model.

    Structurally identical to :class:`AEIC.performance.models.legacy.
    PerformanceTable` (climb/cruise/descent split, bilinear interpolation in
    FL and mass via the shared ``Interpolator``), but without the mass-count
    and FL-only-fuel_flow/ROCD restrictions that only hold for BADA PTF
    data. TAS is still required to be FL-only in every phase: it comes from
    a CAS/Mach speed schedule that doesn't depend on aircraft mass,
    regardless of data source.
    """

    df: pd.DataFrame
    """Performance table data."""

    fl: list[float]
    """Sorted list of unique flight levels in the table."""

    tas: list[float]
    """Sorted list of unique airspeed values in the table."""

    rocd: list[float]
    """Sorted list of unique ROCD values in the table."""

    mass: list[float]
    """Sorted list of unique mass values in the table."""

    rocd_filter: ROCDFilter
    """ROCD filter for this performance table segment."""

    _interpolator: Interpolator | None = field(default=None, repr=False, compare=False)
    """Interpolator for single flight phase table segment."""

    ZERO_ROCD_TOL: ClassVar[float] = 1.0e-6
    """Tolerance for zero rate of climb/descent comparisons."""

    def __post_init__(self):
        match self.rocd_filter:
            case ROCDFilter.NEGATIVE:
                phase = 'descent'
                if not all(v < -self.ZERO_ROCD_TOL for v in self.rocd):
                    raise ValueError(
                        'ROCD values in descent performance table are not all negative'
                    )
            case ROCDFilter.ZERO:
                phase = 'cruise'
                if not all(abs(v) <= self.ZERO_ROCD_TOL for v in self.rocd):
                    raise ValueError(
                        'ROCD values in cruise performance table are not all zero'
                    )
            case ROCDFilter.POSITIVE:
                phase = 'climb'
                # Condition is different here because climb ROCD values can be
                # zero near the operating ceiling of an aircraft.
                if not all(v >= 0.0 for v in self.rocd):
                    raise ValueError(
                        'some ROCD values in climb performance table are negative'
                    )

        # At least one mass for descent (bilinear interpolation degrades to
        # FL-only there), at least two for climb/cruise. No upper bound --
        # unlike the legacy (BADA) model, a PIANO-derived table isn't capped
        # at 2-3 masses.
        min_masses = 1 if self.rocd_filter == ROCDFilter.NEGATIVE else 2
        if len(self.mass) < min_masses:
            raise ValueError(
                f'Piano performance table ({phase}) has too few mass values '
                f'({len(self.mass)}, need at least {min_masses})'
            )

        # For each of positive, zero, and negative ROCD, it should be the case
        # that the input data is dense in (FL, mass) values, in the sense that
        # #rows = #FL × #mass.
        if len(self.df.fl.unique()) * len(self.df.mass.unique()) != len(self.df):
            raise ValueError(
                f'Performance data for {phase} does not have full coverage'
            )

        def check_fl_only(var: str):
            if len(self.df.drop_duplicates(subset=['fl', var])) != len(
                self.df.fl.unique()
            ):
                raise ValueError(
                    f'{var} for {phase} phase depends on variables other than FL'
                )

        # Only TAS is required to be FL-only. fuel_flow (climb, descent) and
        # ROCD (descent) may vary by mass, since PIANO can report genuine
        # per-mass variation there.
        check_fl_only('tas')

    @classmethod
    def from_input(cls, ptin: PerformanceTableInput, rocd_type: ROCDFilter) -> Self:
        """Convert performance table data from input format.

        This class holds performance table data in the form needed for
        trajectory and emissions calculations. The constructor converts from
        the input format from the performance model TOML file."""

        # Convert to Pandas DataFrame for easier handling.
        df = pd.DataFrame(
            [row[: len(ptin.cols)] for row in ptin.data], columns=ptin.cols
        )

        # Extract column unique values for searching.
        fl = sorted(df.fl.unique().tolist())
        tas = sorted(df.tas.unique().tolist())
        rocd = sorted(df.rocd.unique().tolist())
        mass = sorted(df.mass.unique().tolist())

        return cls(df=df, fl=fl, tas=tas, rocd=rocd, mass=mass, rocd_filter=rocd_type)

    def __len__(self) -> int:
        return len(self.df)

    def interpolate(self, state: AircraftState) -> Performance:
        """Perform bilinear interpolation in flight level and aircraft mass."""

        fl = state.altitude * METERS_TO_FL
        mass = state.aircraft_mass
        if mass == 'min':
            mass = min(self.mass)
        elif mass == 'max':
            mass = max(self.mass)

        # Lazily create interpolator for flight phase segment.
        if self._interpolator is None:
            self._interpolator = Interpolator(self.df)

        return self._interpolator(fl, mass)


class PianoPerformanceModel(BasePerformanceModel[SimpleFlightRules]):
    """PIANO-derived table-based performance model."""

    model_type: Literal['piano']
    """Model type identifier for TOML input files."""

    climb_flight_performance: PerformanceTableInput
    """Input data for flight performance table in climb phase."""

    cruise_flight_performance: PerformanceTableInput
    """Input data for flight performance table in cruise phase."""

    descent_flight_performance: PerformanceTableInput
    """Input data for flight performance table in descent phase."""

    _climb_performance_table: PianoPerformanceTable = PrivateAttr()
    _cruise_performance_table: PianoPerformanceTable = PrivateAttr()
    _descent_performance_table: PianoPerformanceTable = PrivateAttr()

    @model_validator(mode='after')
    def validate_pm(self, info):
        """Validate performance model after creation."""

        self._climb_performance_table = PianoPerformanceTable.from_input(
            self.climb_flight_performance, ROCDFilter.POSITIVE
        )
        self._cruise_performance_table = PianoPerformanceTable.from_input(
            self.cruise_flight_performance, ROCDFilter.ZERO
        )
        self._descent_performance_table = PianoPerformanceTable.from_input(
            self.descent_flight_performance, ROCDFilter.NEGATIVE
        )

        return self

    @property
    def empty_mass(self) -> float:
        """Empty aircraft mass.

        Empty mass per BADA-3 is lowest mass in performance table / 1.2."""
        return (
            min(
                min(self._climb_performance_table.mass),
                min(self._cruise_performance_table.mass),
                min(self._descent_performance_table.mass),
            )
            / 1.2
        )

    @property
    def maximum_mass(self) -> float:
        """Maximum aircraft mass from performance table."""
        return max(
            max(self._climb_performance_table.mass),
            max(self._cruise_performance_table.mass),
            max(self._descent_performance_table.mass),
        )

    def evaluate_impl(
        self, state: AircraftState, rules: SimpleFlightRules
    ) -> Performance:
        """Implementation of performance evaluation for the PIANO-derived
        performance model.

        The performance table is separated into climb, cruise and descent
        segments. The performance evaluation implementation uses bilinear
        interpolation in flight level and aircraft mass in the relevant
        segment of the performance table (selected by the flight rule) to
        get performance values."""

        match rules:
            case SimpleFlightRules.CLIMB:
                return self._climb_performance_table.interpolate(state)
            case SimpleFlightRules.CRUISE:
                return self._cruise_performance_table.interpolate(state)
            case SimpleFlightRules.DESCEND:
                return self._descent_performance_table.interpolate(state)

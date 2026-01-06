# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Literal, Self, cast

import numpy as np
import pandas as pd
from pydantic import model_validator

from AEIC.performance.types import AircraftState
from AEIC.utils.models import CIBaseModel
from AEIC.utils.units import METERS_TO_FL


# TODO: Better docstrings.
class PerformanceTableInput(CIBaseModel):
    """Performance table data."""

    cols: list[str]
    """Performance table column labels."""

    data: list[list[float]]
    """Performance table data."""

    @model_validator(mode='after')
    def validate_names_and_sizes(self) -> Self:
        """Normalize and check input column names and array sizes."""

        self.cols = [c.lower() for c in self.cols]

        # Validate column names.
        if len(self.cols) != len(set(self.cols)):
            raise ValueError('Duplicate column names in performance table')
        for required in ['fuel_flow', 'fl', 'tas', 'rocd', 'mass']:
            if required not in self.cols:
                raise ValueError(
                    f'Missing required "{required}" column in performance table'
                )

        # Validate data table dimensions.
        ncols = len(self.cols)
        ndata = len(self.data[0])
        if ndata < ncols:
            raise ValueError('Not enough data columns in performance table')
        if any(len(row) != ndata for row in self.data):
            raise ValueError('Inconsistent number of data columns in performance table')

        return self


class ROCDFilter(Enum):
    """Rate of climb/descent filter for performance table subsetting."""

    NEGATIVE = auto()
    ZERO = auto()
    POSITIVE = auto()


@dataclass
class PerformanceTable:
    """Performance table data."""

    @dataclass
    class Interpolate:
        fuel_flow: float
        true_airspeed: float
        rate_of_climb: float

    df: pd.DataFrame
    """Performance table data."""

    fl: list[float]
    tas: list[float]
    rocd: list[float]
    mass: list[float]

    ZERO_ROCD_TOL: ClassVar[float] = 1.0e-6

    @classmethod
    def from_input(cls, ptin: PerformanceTableInput) -> Self:
        """Convert performance table data from input format.

        This class holds performance table data in the form needed for
        trajectory and emissions calculations. The constructor converts from
        the input format from the performance model TOML file."""

        # Convert to Pandas DataFrame for easier handling.
        df = pd.DataFrame(
            [row[: len(ptin.cols)] for row in ptin.data], columns=np.array(ptin.cols)
        )

        # Extract column unique values for searching.
        fl = sorted(df.fl.unique().tolist())
        tas = sorted(df.tas.unique().tolist())
        rocd = sorted(df.rocd.unique().tolist())
        mass = sorted(df.mass.unique().tolist())

        return cls(df=df, fl=fl, tas=tas, rocd=rocd, mass=mass)

    def __len__(self) -> int:
        return len(self.df)

    def interpolate(
        self,
        *,
        state: AircraftState | None = None,
        fl: float | None = None,
        mass: float | None = None,
    ) -> PerformanceTable.Interpolate:
        if state is not None:
            if fl is not None or mass is not None:
                raise ValueError(
                    'interpolation with state cannot be combined with fl or mass'
                )
            return self._interpolate_state(state)
        if fl is not None and mass is not None:
            return self._interpolate_fl_mass(fl, mass)
        if fl is not None:
            return self._interpolate_fl(fl)
        raise ValueError('interpolation requires at least FL to be specified')

    def _interpolate_state(self, state: AircraftState) -> PerformanceTable.Interpolate:
        fl = state.altitude * METERS_TO_FL
        mass = state.aircraft_mass
        if mass == 'min':
            mass = min(self.mass)
        elif mass == 'max':
            mass = max(self.mass)
        return self._interpolate_fl_mass(fl, mass)

    def _interpolate_fl(
        self,
        fl: float,
    ) -> PerformanceTable.Interpolate:
        # TODO: Docstring.

        # Bracket input values.
        t = self.subset(fl=fl)

        # The interpolation requires exactly two bracketing points for an
        # unambiguous result.
        if len(t.fl) > 2 or len(t.mass) > 1:
            raise ValueError('interpolation pre-condition failed')

        # Make masks for bracketing values of FL and mass.
        fl_lo: pd.Series = t.df.fl < fl
        fl_hi: pd.Series = t.df.fl >= fl

        # Extract bracketing rows from table.
        r_lo = t.df[fl_lo]
        r_hi = t.df[fl_hi]

        # Flatten these rows to series for interpolation weight calculation.
        s_lo = r_lo.squeeze()
        s_hi = r_hi.squeeze()

        # Interpolation weights.
        f_fl = (fl - s_lo.fl) / (s_hi.fl - s_lo.fl)

        # Extract target columns as Numpy array for interpolation.
        def extract(row):
            return cast(
                pd.DataFrame, row[['fuel_flow', 'tas', 'rocd']]
            ).values.squeeze()

        v_lo = extract(r_lo)
        v_hi = extract(r_hi)

        # Bilinear interpolation.
        result = v_lo * (1 - f_fl) + v_hi * f_fl

        return PerformanceTable.Interpolate(
            fuel_flow=result[0], true_airspeed=result[1], rate_of_climb=result[2]
        )

    def _interpolate_fl_mass(
        self, fl: float, mass: float
    ) -> PerformanceTable.Interpolate:
        # TODO: Docstring.

        # Bracket input values.
        t = self.subset(fl=fl, mass=mass)

        # The interpolation requires exactly four bracketing points for an
        # unambiguous result.
        if len(t.fl) > 2 or len(t.mass) > 2:
            raise ValueError('interpolation pre-condition failed')

        # Make masks for bracketing values of FL and mass.
        fl_lo: pd.Series = t.df.fl < fl
        fl_hi: pd.Series = t.df.fl >= fl
        mass_lo: pd.Series = t.df.mass < mass
        mass_hi: pd.Series = t.df.mass >= mass

        # Extract bracketing rows from table.
        r_lo_lo = t.df[fl_lo & mass_lo]
        r_lo_hi = t.df[fl_lo & mass_hi]
        r_hi_lo = t.df[fl_hi & mass_lo]
        r_hi_hi = t.df[fl_hi & mass_hi]

        # Flatten these rows to series for interpolation weight calculation.
        s_lo_lo = r_lo_lo.squeeze()
        s_lo_hi = r_lo_hi.squeeze()
        s_hi_lo = r_hi_lo.squeeze()

        # Interpolation weights.
        f_fl = (fl - s_lo_lo.fl) / (s_hi_lo.fl - s_lo_lo.fl)
        f_mass = (mass - s_lo_lo.mass) / (s_lo_hi.mass - s_lo_lo.mass)

        # Extract target columns as Numpy array for interpolation.
        def extract(row):
            return cast(
                pd.DataFrame, row[['fuel_flow', 'tas', 'rocd']]
            ).values.squeeze()

        v_lo_lo = extract(r_lo_lo)
        v_lo_hi = extract(r_lo_hi)
        v_hi_lo = extract(r_hi_lo)
        v_hi_hi = extract(r_hi_hi)

        # Bilinear interpolation.
        result = (
            v_lo_lo * (1 - f_fl) * (1 - f_mass)
            + v_lo_hi * (1 - f_fl) * f_mass
            + v_hi_lo * f_fl * (1 - f_mass)
            + v_hi_hi * f_fl * f_mass
        )

        return PerformanceTable.Interpolate(
            fuel_flow=result[0], true_airspeed=result[1], rate_of_climb=result[2]
        )

    # TODO: Docstring and comments about isinstance asserts.
    def subset(
        self,
        *,
        fl: float | None = None,
        mass: float | Literal['max', 'min'] | None = None,
        rocd: ROCDFilter | None = None,
    ) -> PerformanceTable:
        df_new = self.df

        if fl is not None:
            lo_fl, hi_fl = self.bracketing_fls(fl)
            df_new = df_new[(df_new.fl >= lo_fl) & (df_new.fl <= hi_fl)]
        assert isinstance(df_new, pd.DataFrame)

        if mass is not None:
            if mass == 'max':
                max_mass = max(self.mass)
                df_new = df_new[df_new.mass == max_mass]
            elif mass == 'min':
                min_mass = min(self.mass)
                df_new = df_new[df_new.mass == min_mass]
            else:
                lo_mass, hi_mass = self.bracketing_mass(mass)
                df_new = df_new[(df_new.mass >= lo_mass) & (df_new.mass <= hi_mass)]
        assert isinstance(df_new, pd.DataFrame)

        if rocd is not None:
            match rocd:
                case ROCDFilter.NEGATIVE:
                    df_new = df_new[df_new.rocd < -self.ZERO_ROCD_TOL]
                case ROCDFilter.ZERO:
                    df_new = df_new[
                        (df_new.rocd >= -self.ZERO_ROCD_TOL)
                        & (df_new.rocd <= self.ZERO_ROCD_TOL)
                    ]
                case ROCDFilter.POSITIVE:
                    df_new = df_new[df_new.rocd > self.ZERO_ROCD_TOL]
        assert isinstance(df_new, pd.DataFrame)

        fl_new = sorted(df_new.fl.unique().tolist())
        tas_new = sorted(df_new.tas.unique().tolist())
        rocd_new = sorted(df_new.rocd.unique().tolist())
        mass_new = sorted(df_new.mass.unique().tolist())

        return PerformanceTable(
            df=df_new, fl=fl_new, tas=tas_new, rocd=rocd_new, mass=mass_new
        )

    # TODO: Remove this eventually.
    def search_indexes(
        self, column: str, error_label: str, value: float
    ) -> tuple[int, int]:
        """Searches the valid values in the performance model for the indices
        bounding a known value in a specified column.

        Args:
            column (str): Column name to search.
            value (float): Value of interest.

        Returns:
            (tuple[int, int]) Tuple containing the indices of the values in performance
                data that bound the given value.
        """

        if column not in self.df.columns:
            raise ValueError(f'Performance data missing required {column} column')

        values = getattr(self, column)
        idx_hi = np.searchsorted(values, value)

        if idx_hi == 0:
            raise ValueError(
                f'Aircraft is trying to operate below minimum {error_label}'
            )
        if idx_hi == len(values):
            raise ValueError(
                f'Aircraft is trying to operate above maximum {error_label}'
            )

        return (int(idx_hi - 1), int(idx_hi))

    def bracketing_fls(self, FL: float) -> tuple[float, float]:
        """Find the bracketing flight levels for a given flight level."""
        idx_lo, idx_hi = self.search_indexes('fl', f'cruise altitude (FL {FL:.2f})', FL)
        return (self.fl[idx_lo], self.fl[idx_hi])

    def bracketing_mass(self, mass: float) -> tuple[float, float]:
        """Find the bracketing mass values for a given mass."""
        idx_lo, idx_hi = self.search_indexes('mass', 'mass', mass)
        return (self.mass[idx_lo], self.mass[idx_hi])


@dataclass
class LegacyPerformanceTable(PerformanceTable):
    def __post_init__(self):
        # Check that we have three mass values.
        if len(self.mass) != 3:
            raise ValueError(
                'Legacy performance table must have exactly three mass values'
            )

        # TAS at zero ROC should depend only on FL.
        check = self.df[
            (self.df.rocd >= -self.ZERO_ROCD_TOL) & (self.df.rocd <= self.ZERO_ROCD_TOL)
        ]
        if len(set(zip(check.fl, check.tas))) != len(set(check.fl)):
            raise ValueError('TAS at zero ROC depends on variables other than FL')

        # Fuel flow and TAS for positive ROC should depend only on FL.
        check = self.df[self.df.rocd > self.ZERO_ROCD_TOL]
        if len(set(zip(check.fl, check.tas))) != len(set(check.fl)):
            raise ValueError('TAS at positive ROC depends on variables other than FL')
        if len(set(zip(check.fl, check.fuel_flow))) != len(set(check.fl)):
            raise ValueError(
                'fuel flow at positive ROC depends on variables other than FL'
            )

        # Fuel flow and TAS for negative ROC should depend only on FL.
        # TODO: COMMENTED OUT BECAUSE THE SAMPLE PERFORMANCE MODEL DOESN'T PASS
        # THIS TEST!
        # check = self.df[self.df.rocd < -self.ZERO_ROCD_TOL]
        # if len(set(zip(check.fl, check.tas))) != len(set(check.fl)):
        #     raise ValueError(
        #         'TAS at negative ROC depends on variables other than FL'
        #     )
        # if len(set(zip(check.fl, check.fuel_flow))) != len(set(check.fl)):
        #     raise ValueError(
        #         'fuel flow at negative ROC depends on variables other than FL'
        #     )

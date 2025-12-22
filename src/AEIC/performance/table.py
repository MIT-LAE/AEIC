from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import PrivateAttr, model_validator

from AEIC.utils.models import CIBaseModel

from .base import BasePerformanceModel

# ------------------------------------------------------------------------------
#
#  Main performance table class
#


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


class PerformanceTable:
    """Performance table data."""

    df: pd.DataFrame
    """Proof-of-concept representation of performance table as Pandas data
    frame. Intended to replace Numpy array representation eventually."""

    # These are all private attributes because they are derived from the input
    # table data during model validation. These variables support the existing
    # Numpy array-based representation of the performance table. They will
    # eventually be replaced with a representation using a Pandas data frame
    # with a simplified subsetting and interpolation API.
    # TODO: Remove these eventually.
    _old_table: np.ndarray

    def __init__(self, ptin: PerformanceTableInput) -> None:
        """Convert performance table data from input format.

        This class holds performance table data in the form needed for
        trajectory and emissions calculations. The constructor converts from
        the input format from the performance model TOML file."""

        # Convert to Pandas DataFrame for easier handling.
        self.df = pd.DataFrame(
            [row[: len(ptin.cols)] for row in ptin.data], columns=np.array(ptin.cols)
        )

        # Extract column unique values for searching.
        self.fl = sorted(self.df.fl.unique())
        self.tas = sorted(self.df.tas.unique())
        self.rocd = sorted(self.df.rocd.unique())
        self.mass = sorted(self.df.mass.unique())

        self._old_setup(ptin)

    def _old_setup(self, ptin: PerformanceTableInput) -> None:
        cols = ptin.cols
        data = ptin.data

        # Identify output column (we assume it's the first column or
        # explicitly labeled as fuel flow)
        try:
            output_col_idx = cols.index("fuel_flow")  # Output is fuel flow
        except ValueError:
            raise ValueError("FUEL_FLOW column not found in performance data.")

        input_col_names = [c for i, c in enumerate(cols) if i != output_col_idx]

        # Extract and sort unique values for each input dimension
        input_values = {
            col: sorted(set(row[cols.index(col)] for row in data))
            for col in input_col_names
        }
        input_indices = {
            col: {val: idx for idx, val in enumerate(input_values[col])}
            for col in input_col_names
        }

        # Prepare multidimensional shape and index arrays
        shape = tuple(len(input_values[col]) for col in input_col_names)
        fuel_flow_array = np.zeros(shape)

        # Get index arrays for each input variable
        index_arrays = [
            np.array([input_indices[col][row[cols.index(col)]] for row in data])
            for col in input_col_names
        ]
        index_arrays = tuple(index_arrays)

        # Extract output (fuel flow) values
        fuel_flow = np.array([row[output_col_idx] for row in data])

        # Assign to multidimensional array using advanced indexing
        fuel_flow_array[index_arrays] = fuel_flow

        # Save results
        self._old_table = fuel_flow_array

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
        ind_high = np.searchsorted(values, value)

        if ind_high == 0:
            raise ValueError(
                f'Aircraft is trying to operate below minimum {error_label}'
            )
        if ind_high == len(values):
            raise ValueError(
                f'Aircraft is trying to operate above maximum {error_label}'
            )

        return (int(ind_high - 1), int(ind_high))

    def search_mass_ind(self, mass: float) -> tuple[int, int]:
        """Search flight levels for the indices bounding a given mass value."""
        return self.search_indexes('mass', 'mass', mass)

    def search_flight_levels_ind(self, FL: float) -> tuple[int, int]:
        """Search flight levels for the indices bounding a given FL value."""
        return self.search_indexes('fl', f'cruise altitude (FL {FL:.2f})', FL)

    def bracketing_fls(self, FL: float) -> tuple[float, float]:
        """Find the bracketing flight levels for a given flight level."""
        ind_low, ind_high = self.search_flight_levels_ind(FL)
        return (self.fl[ind_low], self.fl[ind_high])

    def bracketing_mass(self, mass: float) -> tuple[float, float]:
        """Find the bracketing mass values for a given mass."""
        ind_low, ind_high = self.search_mass_ind(mass)
        return (self.mass[ind_low], self.mass[ind_high])


# ------------------------------------------------------------------------------
#
#  Performance model class
#


class TablePerformanceModel(BasePerformanceModel):
    """Table-based performance model."""

    # TODO: Better docstrings.

    model_type: Literal['table']
    """Model type identifier for TOML input files."""

    flight_performance: PerformanceTableInput
    """Main flight performance table."""

    _performance_table: PerformanceTable = PrivateAttr()

    @model_validator(mode='after')
    def validate_pm(self, info):
        """Validate performance model after creation."""

        # TODO: Check that everything is filled in, pulling values from the EDB
        # or elsewhere as required.
        self._performance_table = PerformanceTable(self.flight_performance)
        return self

    @property
    def performance_table(self) -> PerformanceTable:
        return self._performance_table

    def fuel_flow(
        self, altitude: float, mass: float, rocd: float, speed: float
    ) -> float:
        """*Fuel flow API not yet implemented.*"""
        raise NotImplementedError(
            'TablePerformanceModel.fuel_flow not implemented yet.'
        )

    def search_mass_ind(self, mass: float) -> tuple[int, int]:
        """Search flight levels for the indices bounding a given mass value.

        (Forwards to performance table.)"""
        return self.performance_table.search_indexes('mass', 'mass', mass)

    def search_flight_levels_ind(self, FL: float) -> tuple[int, int]:
        """Search flight levels for the indices bounding a given FL value.

        (Forwards to performance table.)"""
        return self.performance_table.search_indexes(
            'fl', f'cruise altitude (FL {FL:.2f})', FL
        )

    def bracketing_fls(self, FL: float) -> tuple[float, float]:
        """Find the bracketing flight levels for a given flight level.

        (Forwards to performance table.)"""
        return self.performance_table.bracketing_fls(FL)

    def bracketing_mass(self, mass: float) -> tuple[float, float]:
        """Find the bracketing mass values for a given mass.

        (Forwards to performance table.)"""
        return self.performance_table.bracketing_mass(mass)

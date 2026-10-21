"""Linear interpolation over performance tables with ragged axes.

In a ragged table, an inner axis can have different points under each node of
the outer axes. For example, each mass in a PIANO climb table has its own
flight levels and its own ceiling. Interpolation is nested: it interpolates
along the innermost axis first, then across each outer axis in turn.

Queries outside the data raise `ValueError`."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from itertools import groupby
from typing import ClassVar, Self

import numpy as np
import pandas as pd

from AEIC.performance.models.base import PerformanceTableInput
from AEIC.performance.models.legacy import ROCDFilter
from AEIC.performance.types import AircraftState, Performance
from AEIC.units import METERS_TO_FL


@dataclass(frozen=True, slots=True)
class _NodeBase:
    """Fields common to all nodes of the interpolation tree."""

    axis_values: list[float]
    """Sorted axis values under this node."""

    outer_coords: tuple[float, ...]
    """Fixed coordinates of this node on the outer axes."""


@dataclass(frozen=True, slots=True)
class _InnerNode(_NodeBase):
    """Node for an outer axis. Holds one child node for each axis value."""

    children: list[_Node]
    """Child node for each axis value."""


@dataclass(frozen=True, slots=True)
class _LeafNode(_NodeBase):
    """Node for the last axis. Holds the values of the interpolation variables
    for each axis value."""

    interp_data: list[tuple[float, ...]]
    """Values of interpolation variables for each point in `axis_values`.

    In the order of `interp_vars` passed to `RaggedInterpolator`."""


_Node = _InnerNode | _LeafNode
"""Node of the interpolation tree for one axis."""


class RaggedInterpolator:
    """Linear interpolation over nested axes.

    Axes are listed outer to inner. An inner axis can have different points
    under each node of the outer axes."""

    def __init__(
        self,
        df: pd.DataFrame,
        axes: tuple[str, ...],
        interp_vars: tuple[str, ...],
        name: str,
    ):
        self.axes = axes
        self.interp_vars = interp_vars
        self.name = name
        self._last_depth = len(axes) - 1

        # Rows store axis values then interpolation variables
        df = df.sort_values(list(axes), kind='stable')
        rows = df[list(axes) + list(interp_vars)].to_numpy(dtype=float).tolist()
        # Build the interpolation tree from the root node which has no fixed coordinates
        self._root = self._build(rows=rows, depth=0, outer_coords=())

    def _build(
        self, rows: list[list[float]], depth: int, outer_coords: tuple[float, ...]
    ) -> _Node:
        """Build the tree node for rows sorted by axes and sharing
        `outer_coords`."""

        # Innermost interpolation axis is where we store the values of the
        # interpolation variables, so we create a _LeafNode
        if depth == self._last_depth:
            n_axes = len(self.axes)
            axis_values = [row[depth] for row in rows]
            # Store the values of the interpolation variables
            interp_data = [tuple(row[n_axes:]) for row in rows]
            return _LeafNode(
                axis_values=axis_values,
                outer_coords=outer_coords,
                interp_data=interp_data,
            )

        def value_on_this_axis(row: list[float]) -> float:
            return row[depth]

        axis_values = []
        children = []

        # The rows are sorted, so the rows of each axis value are contiguous.
        # Group by value on the current interpolation axis: we are not on the
        # last interpolation axis so each of these children represents one
        # additional level of interpolation
        for axis_value, rows_with_value in groupby(rows, key=value_on_this_axis):
            axis_values.append(axis_value)
            # Append the current axis value to the outer coords
            new_coords = (*outer_coords, axis_value)
            # Create a new node that is one interpolation axis deeper
            child_node = self._build(list(rows_with_value), depth + 1, new_coords)
            children.append(child_node)

        # We are not at the last interpolation axis so we create a new node with
        # a child per interpolation axis value
        return _InnerNode(
            axis_values=axis_values, outer_coords=outer_coords, children=children
        )

    def __call__(self, *coords: float) -> tuple[float, ...]:
        """Interpolate the interpolation variables at the given coordinates.

        Raises:
            ValueError: If a coordinate is outside the data.
        """

        if len(coords) != len(self.axes):
            raise TypeError(
                f'{self.name} table: expected {len(self.axes)} coordinates '
                f'{self.axes}, got {len(coords)}'
            )
        if not self._root.axis_values:
            raise ValueError(f'{self.name} table: no data')

        # Convert everything to Python floats for consistency
        coords = tuple([float(x) for x in coords])
        return self._evaluate(self._root, coords, 0)

    def _evaluate(
        self, node: _Node, coords: tuple[float, ...], depth: int
    ) -> tuple[float, ...]:
        axis_values = node.axis_values
        coord = coords[depth]

        # Bounds are inclusive. A NaN coordinate also fails this check.
        if not axis_values[0] <= coord <= axis_values[-1]:
            raise ValueError(self._range_message(depth, coord, node))

        # Find the two axis values around `coord`, and how far `coord` is from
        # the lower one, as a fraction of the interval. With one axis value,
        # the range check accepts only that exact value.
        n_values = len(axis_values)
        if n_values == 1:
            # idx_high is never used in this branch but to be
            # explicit set it to 0 anyway
            idx_low, idx_high = 0, 0
            interp_weight = 0.0
        else:
            # Bisect right returns the idx of the first value in
            # axis_values (sorted) > coord
            # Then the lower point for interpolation is idx - 1
            idx_low = bisect_right(axis_values, coord) - 1
            # Clamp idx_low so that idx_low + 1 is always valid
            idx_low = min(idx_low, n_values - 2)
            idx_high = idx_low + 1

            axis_value_low = axis_values[idx_low]
            axis_value_high = axis_values[idx_high]

            interp_weight = (coord - axis_value_low) / (
                axis_value_high - axis_value_low
            )

            # If interp_weight == 1, it means coord is exactly
            # axis_values[idx_high]. Setting idx_low = idx_high
            # and interp_weight = 0.0 allows us to treat the
            # cases:
            #      coord == axis_values[idx_high]
            #      coord == axis_values[idx_low]
            # the same way afterwards
            if interp_weight == 1.0:
                idx_low, interp_weight = idx_high, 0.0

        if isinstance(node, _LeafNode):
            # data_low and data_high are tuples with one value for each
            # variable in interp_vars
            data_low = node.interp_data[idx_low]

            # On an exact axis value (idx_low or idx_high thanks to
            # earlier normalization) we return directly
            if interp_weight == 0.0:
                return data_low

            data_high = node.interp_data[idx_high]

        else:
            children = node.children
            data_low = self._evaluate(children[idx_low], coords, depth + 1)

            # On an exact axis value (idx_low or idx_high thanks to
            # earlier normalization) we return directly but here this is because
            # coords[depth + 1] could be out of range in children[idx_high].axis_values
            # as there are no guarantees that two nodes have the same range of axis
            # values (that's the point of the ragged interpolator):
            # data_high is not needed (we are exactly at idx_low so interp_weight = 0)
            # but the code could still throw in the subsequent call to
            #   self._evaluate(children[idx_high], coords, depth + 1)
            # because it could be that (true for any node down that subtree):
            #   children[idx_high].axis_values[0] > coords[depth + 1]
            #               AND / OR
            #   children[idx_high].axis_values[-1] < coords[depth + 1]
            # so we skip evaluating it entirely and return
            if interp_weight == 0.0:
                return data_low

            data_high = self._evaluate(children[idx_high], coords, depth + 1)

        # Apply interpolation weight to each interpolation variable
        return tuple(
            [
                (1.0 - interp_weight) * d_low + interp_weight * d_high
                for d_low, d_high in zip(data_low, data_high)
            ]
        )

    def _range_message(self, depth: int, coord: float, node: _Node) -> str:
        message = (
            f'{self.name} table: {self.axes[depth]}={coord} '
            f'outside [{node.axis_values[0]}, {node.axis_values[-1]}]'
        )
        if node.outer_coords:
            where = ', '.join(
                f'{axis}={value}' for axis, value in zip(self.axes, node.outer_coords)
            )
            message += f' for {where}'
        return message


@dataclass
class RaggedPerformanceTable:
    """Performance table for one flight phase, with ragged axes.

    On construction, the class checks that every axis and output column is
    present, that no key is duplicated over the axes, that no output value is
    NaN, and that the sign of the ROCD values matches the flight phase. It does
    not check that the table is dense."""

    df: pd.DataFrame
    """Performance table data."""

    axes: tuple[str, ...]
    """Interpolation axes, outer to inner."""

    rocd_filter: ROCDFilter
    """ROCD filter for this flight phase."""

    _interpolator: RaggedInterpolator = field(init=False, repr=False)
    """Interpolator over the table data."""

    OUTPUTS: ClassVar[tuple[str, ...]] = ('tas', 'rocd', 'fuel_flow')
    """Columns returned in `Performance`."""

    ZERO_ROCD_TOL: ClassVar[float] = 1.0e-6
    """Tolerance for zero rate of climb/descent comparisons."""

    def __post_init__(self):
        phase = self.phase

        for col in (*self.axes, *self.OUTPUTS):
            if col not in self.df.columns:
                raise ValueError(f'Missing "{col}" column in {phase} performance table')

        if self.df.duplicated(subset=list(self.axes)).any():
            raise ValueError(
                f'{phase} performance table has duplicate {self.axes} keys'
            )

        if self.df[list(self.OUTPUTS)].isna().any().any():
            raise ValueError(f'{phase} performance table has NaN output values')

        # Check the sign of the ROCD values in the table matches the ROCD
        # filter for this flight phase.
        rocd = self.df.rocd
        match self.rocd_filter:
            case ROCDFilter.NEGATIVE:
                if not (rocd < -self.ZERO_ROCD_TOL).all():
                    raise ValueError(
                        'ROCD values in descent performance table are not all negative'
                    )
            case ROCDFilter.ZERO:
                if not (rocd.abs() <= self.ZERO_ROCD_TOL).all():
                    raise ValueError(
                        'ROCD values in cruise performance table are not all zero'
                    )
            case ROCDFilter.POSITIVE:
                # Climb ROCD values can be zero near the operating ceiling of
                # an aircraft.
                if not (rocd >= 0.0).all():
                    raise ValueError(
                        'some ROCD values in climb performance table are negative'
                    )

        self.df = self.df.sort_values(list(self.axes), kind='stable', ignore_index=True)
        self._interpolator = RaggedInterpolator(self.df, self.axes, self.OUTPUTS, phase)

    @classmethod
    def from_input(
        cls,
        ptin: PerformanceTableInput,
        rocd_filter: ROCDFilter,
        axes: tuple[str, ...],
    ) -> Self:
        """Convert performance table data from the TOML input format."""

        # Extra data columns are allowed, so cut each row to the labelled
        # columns.
        df = pd.DataFrame(
            [row[: len(ptin.cols)] for row in ptin.data], columns=np.array(ptin.cols)
        )
        return cls(df=df, axes=axes, rocd_filter=rocd_filter)

    @property
    def phase(self) -> str:
        """Name of the flight phase for this table."""
        match self.rocd_filter:
            case ROCDFilter.POSITIVE:
                return 'climb'
            case ROCDFilter.ZERO:
                return 'cruise'
            case ROCDFilter.NEGATIVE:
                return 'descent'

    def interpolate(self, state: AircraftState, **extra_coords: float) -> Performance:
        """Interpolate performance at an aircraft state.

        Flight level and mass come from `state`. Give a value for each other
        axis of the table as a keyword argument, for example `mach=0.78`.

        Raises:
            ValueError: If the keyword arguments do not match the axes of the
                table other than `fl` and `mass`. Also if the state is outside
                the data.
        """

        # Validate coords
        expected = set(self.axes) - {'fl', 'mass'}
        if set(extra_coords) != expected:
            raise ValueError(
                f'{self.phase} performance needs coordinates {sorted(expected)}, '
                f'got {sorted(extra_coords)}'
            )

        mass = state.aircraft_mass
        if mass == 'min':
            mass = self.df.mass.min()
        elif mass == 'max':
            mass = self.df.mass.max()

        # Round FL to 1e-9 because the metric altitude for an FL
        # does not convert exactly back to the FL which can cause
        # out of bounds issues in the interpolator e.g.:
        # FL400 => Meters => FL 400 + 1e-15
        # And if interpolator bound is at FL400, this point
        # would be considered out of bounds even though is it not really
        # TODO 1e-9 works because FL in performance tables (BADA/PIANO)
        # have at most 2-3 decimals, if they had 10 this rounding would fail
        # to snap the points correctly to the interpolation grid
        fl = round(state.altitude * METERS_TO_FL, 9)

        # Unpack all interpolation coordinates passed by the caller
        coords = {'fl': fl, 'mass': mass, **extra_coords}
        # Reorder them to be consistent with what the interpolator expects
        ordered_coords = tuple(coords[a] for a in self.axes)
        tas, rocd, fuel_flow = self._interpolator(*ordered_coords)
        return Performance(true_airspeed=tas, rate_of_climb=rocd, fuel_flow=fuel_flow)

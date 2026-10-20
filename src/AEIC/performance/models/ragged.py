"""Linear interpolation over performance tables with ragged axes.

In a ragged table, an inner axis can have different points under each node of
the outer axes. For example, each mass in a PIANO climb table has its own
flight levels and its own ceiling. Interpolation is nested: it interpolates
along the innermost axis first, then across each outer axis in turn.

Queries outside the data raise `ValueError`."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from itertools import groupby

import pandas as pd


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
        # Build the interpolation tree from the root node which has no fixed axes
        self._root = self._build(rows, 0, ())

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
            children.append(
                # Create a new node that is one interpolation axis deeper
                self._build(
                    list(rows_with_value), depth + 1, (*outer_coords, axis_value)
                )
            )
        # We are not at the last interpolation axis so we create a new node with
        # a child per next interpolation axis value
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
            idx_low, interp_weight = 0, 0.0
        else:
            idx_low = bisect_right(axis_values, coord) - 1
            if idx_low > n_values - 2:
                idx_low = n_values - 2
            axis_value_low = axis_values[idx_low]
            axis_value_high = axis_values[idx_low + 1]
            interp_weight = (coord - axis_value_low) / (
                axis_value_high - axis_value_low
            )

            # On the upper axis value, use only that value. Then a fraction of
            # zero always means "exactly on the axis value at `lower_index`".
            if interp_weight == 1.0:
                idx_low, interp_weight = idx_low + 1, 0.0

        if isinstance(node, _LeafNode):
            # data_low and data_high are tuples with one value for each variable in
            # interp_vars
            data_low = node.interp_data[idx_low]

            # On an exact axis value, return directly that value as idx_low + 1 could be
            # out of range
            if interp_weight == 0.0:
                return data_low

            data_high = node.interp_data[idx_low + 1]

        else:
            children = node.children
            data_low = self._evaluate(children[idx_low], coords, depth + 1)

            # On an exact axis value, return directly that value as idx_low + 1 could be
            # out of range
            if interp_weight == 0.0:
                return data_low

            data_high = self._evaluate(children[idx_low + 1], coords, depth + 1)

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

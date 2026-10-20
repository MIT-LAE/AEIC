"""Linear interpolation over performance tables with ragged axes.

In a ragged table, an inner axis can have different points under each node of
the outer axes. For example, each mass in a PIANO climb table has its own
flight levels and its own ceiling. Interpolation is nested: it interpolates
along the innermost axis first, then across each outer axis in turn. On a
rectilinear grid, this gives exactly the multilinear result.

Queries outside the data raise `ValueError`. There is no extrapolation."""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from bisect import bisect_right
from itertools import groupby
from operator import itemgetter
from typing import Any, NamedTuple

import pandas as pd


class _Node(NamedTuple):
    """Node of the interpolation tree for one axis."""

    xs: list[float]
    """Sorted unique axis values under this node."""

    children: list[Any]
    """One child for each axis value. A child is a `_Node` for an inner axis
    and an output tuple for the last axis."""

    outer: tuple[float, ...]
    """Coordinates of this node on the outer axes. Used in error messages."""


class RaggedInterpolator:
    """Linear interpolation over nested axes.

    Axes are listed outer to inner. An inner axis can have different points
    under each node of the outer axes."""

    def __init__(
        self,
        df: pd.DataFrame,
        axes: tuple[str, ...],
        outputs: tuple[str, ...],
        name: str,
    ):
        self.axes = axes
        self.outputs = outputs
        self.name = name
        self._last = len(axes) - 1

        # Rows as plain Python floats: axis values first, then outputs.
        df = df.sort_values(list(axes), kind='stable')
        rows = df[list(axes) + list(outputs)].to_numpy(dtype=float).tolist()
        self._root = self._build(rows, 0, ())

    def _build(self, rows: list[list[float]], depth: int, outer: tuple) -> _Node:
        """Build the tree node for rows sorted by axes and sharing `outer`."""

        if depth == self._last:
            ncoords = len(self.axes)
            xs = [row[depth] for row in rows]
            children = [tuple(row[ncoords:]) for row in rows]
            return _Node(xs, children, outer)

        xs = []
        children = []
        for x, group in groupby(rows, key=itemgetter(depth)):
            xs.append(x)
            children.append(self._build(list(group), depth + 1, (*outer, x)))
        return _Node(xs, children, outer)

    def __call__(self, *coords: float) -> tuple[float, ...]:
        """Interpolate the outputs at the given coordinates, in axis order.

        Raises:
            ValueError: If a coordinate is outside the data.
        """

        if len(coords) != len(self.axes):
            raise TypeError(
                f'{self.name} table: expected {len(self.axes)} coordinates '
                f'{self.axes}, got {len(coords)}'
            )
        if not self._root.xs:
            raise ValueError(f'{self.name} table: no data')
        return self._evaluate(self._root, tuple(map(float, coords)), 0)

    def _evaluate(
        self, node: _Node, coords: tuple[float, ...], depth: int
    ) -> tuple[float, ...]:
        # This is the hot path. Avoid helper calls and generator expressions.
        xs, children, _ = node
        x = coords[depth]

        # Bounds are inclusive. A NaN coordinate also fails this check.
        if not xs[0] <= x <= xs[-1]:
            raise ValueError(self._range_message(depth, x, node))

        # One node: the range check accepts only that exact value. On an exact
        # node, do not use the data of the neighbouring node. It can be out of
        # range there.
        n = len(xs)
        if n == 1:
            k, t = 0, 0.0
        else:
            k = bisect_right(xs, x) - 1
            if k > n - 2:
                k = n - 2
            x0 = xs[k]
            t = (x - x0) / (xs[k + 1] - x0)
            if t == 1.0:
                k, t = k + 1, 0.0

        if depth == self._last:
            a = children[k]
            if t == 0.0:
                return a
            b = children[k + 1]
        else:
            a = self._evaluate(children[k], coords, depth + 1)
            if t == 0.0:
                return a
            b = self._evaluate(children[k + 1], coords, depth + 1)

        s = 1.0 - t
        return tuple([s * va + t * vb for va, vb in zip(a, b)])

    def _range_message(self, depth: int, x: float, node: _Node) -> str:
        message = (
            f'{self.name} table: {self.axes[depth]}={x} '
            f'outside [{node.xs[0]}, {node.xs[-1]}]'
        )
        if node.outer:
            where = ', '.join(
                f'{axis}={value}' for axis, value in zip(self.axes, node.outer)
            )
            message += f' for {where}'
        return message

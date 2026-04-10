"""
Tests for the voxel tracing kernels in AEIC.gridding.kernels.

Both traverse_segment_nonuniform_z (core 3-D Amanatides-Woo traversal) and
process_segments_nonuniform_z (batched dispatcher with fast/slow path) are
exercised.  The tests are self-contained: they build small grids and segment
arrays directly rather than going through the Grid / Trajectory objects, so
they run without a trajectory store or mission database.
"""

import numpy as np
import pytest

from AEIC.gridding.kernels import (
    process_segments_nonuniform_z,
    traverse_segment_nonuniform_z,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NI, NJ, NK, NSPEC = 4, 4, 5, 2

# Left-edge boundaries for 5 uniform vertical cells: [0-100), [100-200), ...,
# [400-∞).  The last cell has no upper boundary in the kernel (checked via
# k+1 >= z_edges.shape[0]).
Z_EDGES = np.array([0.0, 100.0, 200.0, 300.0, 400.0])


def _fresh_grid(ni=NI, nj=NJ, nk=NK, nspec=NSPEC):
    return np.zeros((ni, nj, nk, nspec), dtype=np.float64)


def _weights(a=10.0, b=5.0):
    return np.array([a, b], dtype=np.float64)


# ---------------------------------------------------------------------------
# traverse_segment_nonuniform_z
# ---------------------------------------------------------------------------


class TestTraverseSegment:
    """Unit tests for the core Amanatides-Woo segment traversal."""

    # --- weight conservation ---

    def test_weight_conservation_diagonal(self):
        """Total accumulated weight equals input weight for a diagonal segment."""
        grid = _fresh_grid()
        w = _weights()
        traverse_segment_nonuniform_z(0.2, 0.2, 50.0, 3.8, 3.8, 350.0, Z_EDGES, grid, w)
        assert np.isclose(grid[..., 0].sum(), w[0])
        assert np.isclose(grid[..., 1].sum(), w[1])

    @pytest.mark.parametrize(
        'endpoints',
        [
            (0.5, 1.5, 150.0, 3.5, 1.5, 150.0),  # i-direction only
            (1.5, 0.5, 150.0, 1.5, 3.5, 150.0),  # j-direction only
            (1.5, 1.5, 50.0, 1.5, 1.5, 350.0),  # z-direction only
        ],
    )
    def test_weight_conservation_axis_aligned(self, endpoints):
        """Conservation holds for each axis-aligned direction."""
        grid = _fresh_grid()
        w = _weights()
        traverse_segment_nonuniform_z(*endpoints, Z_EDGES, grid, w)
        assert np.isclose(grid[..., 0].sum(), w[0])
        assert np.isclose(grid[..., 1].sum(), w[1])

    # --- horizontal splitting ---

    def test_horizontal_i_boundary_equal_split(self):
        """Segment crossing one i-boundary at its midpoint gives a 50/50 split."""
        grid = _fresh_grid()
        w = _weights()
        # i: 0.3 → 1.7.  Boundary at i=1 is crossed at t=0.7/1.4=0.5.
        # j and z are constant, so the segment stays in (j=1, k=1).
        traverse_segment_nonuniform_z(
            0.3, 1.5, 150.0, 1.7, 1.5, 150.0, Z_EDGES, grid, w
        )

        assert np.allclose(grid[0, 1, 1, :], 0.5 * w)
        assert np.allclose(grid[1, 1, 1, :], 0.5 * w)
        # No other cells should receive weight.
        assert np.isclose(grid.sum(), w.sum())

    def test_horizontal_i_boundary_asymmetric_split(self):
        """Non-centred crossing gives the correct proportional fractions."""
        grid = _fresh_grid()
        w = _weights()
        # i: 0.1 → 1.1.  Boundary at i=1 is crossed at t=0.9/1.0=0.9.
        traverse_segment_nonuniform_z(
            0.1, 1.5, 150.0, 1.1, 1.5, 150.0, Z_EDGES, grid, w
        )

        assert np.allclose(grid[0, 1, 1, :], 0.9 * w)
        assert np.allclose(grid[1, 1, 1, :], 0.1 * w)
        assert np.isclose(grid.sum(), w.sum())

    # --- vertical splitting ---

    def test_vertical_ascending_proportional_split(self):
        """Ascending segment distributes weight by fraction of z-range in each cell.

        z: 50 → 250 (total Δz = 200).
          cell k=0 [0,100):   25–100  → 75 / 200 ... wait, 50–100 = 50 → 0.25
          cell k=1 [100,200): 100–200 →         → 0.50
          cell k=2 [200,300): 200–250 →         → 0.25
        """
        grid = _fresh_grid()
        w = _weights()
        traverse_segment_nonuniform_z(1.5, 1.5, 50.0, 1.5, 1.5, 250.0, Z_EDGES, grid, w)

        assert np.allclose(grid[1, 1, 0, :], 0.25 * w)
        assert np.allclose(grid[1, 1, 1, :], 0.50 * w)
        assert np.allclose(grid[1, 1, 2, :], 0.25 * w)
        assert np.isclose(grid.sum(), w.sum())

    def test_vertical_descending_same_distribution(self):
        """Descending and ascending segments produce identical cell distributions."""
        w = _weights()
        grid_asc = _fresh_grid()
        grid_desc = _fresh_grid()
        traverse_segment_nonuniform_z(
            1.5, 1.5, 50.0, 1.5, 1.5, 250.0, Z_EDGES, grid_asc, w
        )
        traverse_segment_nonuniform_z(
            1.5, 1.5, 250.0, 1.5, 1.5, 50.0, Z_EDGES, grid_desc, w
        )

        assert np.allclose(grid_asc, grid_desc)

    # --- boundary clamping ---

    def test_segment_clamped_to_last_cell(self):
        """Segment above the highest z_edge is clamped to the last vertical cell."""
        grid = _fresh_grid()
        w = _weights()
        # z_edges[-1] = 400; z0=450, z1=600 → both beyond the array, clamped to k=4.
        traverse_segment_nonuniform_z(
            1.5, 1.5, 450.0, 1.5, 1.5, 600.0, Z_EDGES, grid, w
        )

        assert np.allclose(grid[1, 1, NK - 1, :], w)
        assert np.isclose(grid.sum(), w.sum())

    def test_segment_clamped_to_first_cell(self):
        """Segment below z_edges[0] is clamped to k=0."""
        grid = _fresh_grid()
        w = _weights()
        # Both z values negative → below the grid bottom (z_edges[0]=0).
        traverse_segment_nonuniform_z(
            1.5, 1.5, -50.0, 1.5, 1.5, -10.0, Z_EDGES, grid, w
        )

        assert np.allclose(grid[1, 1, 0, :], w)
        assert np.isclose(grid.sum(), w.sum())

    def test_horizontal_out_of_bounds_weight_dropped(self):
        """Segment entirely outside the horizontal grid contributes nothing."""
        grid = _fresh_grid()
        w = _weights()
        # i starts well beyond ni.
        traverse_segment_nonuniform_z(
            float(NI + 1), 1.5, 150.0, float(NI + 2), 1.5, 150.0, Z_EDGES, grid, w
        )
        assert grid.sum() == 0.0

    # --- non-uniform vertical grid ---

    def test_nonuniform_z_edges(self):
        """Non-uniform cell boundaries give proportional (not equal) fractions.

        Cells: [0, 50), [50, 200), [200, 500) with widths 50, 150, 300.
        Segment z: 25 → 375 (Δz = 350).
          k=0: z 25–50  →  25/350
          k=1: z 50–200 → 150/350
          k=2: z 200–375 → 175/350
        """
        z_edges_nu = np.array([0.0, 50.0, 200.0, 500.0])
        grid = np.zeros((NI, NJ, 3, NSPEC), dtype=np.float64)
        w = _weights()
        traverse_segment_nonuniform_z(
            1.5, 1.5, 25.0, 1.5, 1.5, 375.0, z_edges_nu, grid, w
        )

        assert np.allclose(grid[1, 1, 0, :], (25 / 350) * w, rtol=1e-10)
        assert np.allclose(grid[1, 1, 1, :], (150 / 350) * w, rtol=1e-10)
        assert np.allclose(grid[1, 1, 2, :], (175 / 350) * w, rtol=1e-10)
        assert np.isclose(grid.sum(), w.sum())

    def test_single_cell_segment_all_weight_in_one_voxel(self):
        """A segment entirely within one voxel deposits all weight there."""
        grid = _fresh_grid()
        w = _weights()
        # Segment from (0.1, 0.1, 10) to (0.9, 0.9, 90) — stays in (i=0, j=0, k=0).
        traverse_segment_nonuniform_z(0.1, 0.1, 10.0, 0.9, 0.9, 90.0, Z_EDGES, grid, w)

        assert np.allclose(grid[0, 0, 0, :], w)
        assert np.isclose(grid.sum(), w.sum())

    def test_negative_i_direction(self):
        """Segment travelling in the negative i-direction distributes correctly."""
        grid = _fresh_grid()
        w = _weights()
        # i: 1.7 → 0.3 (negative step), j and z fixed.
        # Boundary at i=1 crossed at t=0.7/1.4=0.5.
        traverse_segment_nonuniform_z(
            1.7, 1.5, 150.0, 0.3, 1.5, 150.0, Z_EDGES, grid, w
        )

        assert np.allclose(grid[1, 1, 1, :], 0.5 * w)
        assert np.allclose(grid[0, 1, 1, :], 0.5 * w)
        assert np.isclose(grid.sum(), w.sum())


# ---------------------------------------------------------------------------
# process_segments_nonuniform_z
# ---------------------------------------------------------------------------

# Small grid with explicit origin used for all process_segments tests.
_LAT_MIN = 0.0
_LON_MIN = 0.0
_DLAT = 1.0
_DLON = 1.0


def _run_process(lat0, lon0, z0, lat1, lon1, z1, weights_2d, grid=None):
    """Convenience wrapper around process_segments_nonuniform_z."""
    if grid is None:
        grid = _fresh_grid()
    process_segments_nonuniform_z(
        np.asarray(lat0, dtype=np.float64),
        np.asarray(lon0, dtype=np.float64),
        np.asarray(z0, dtype=np.float64),
        np.asarray(lat1, dtype=np.float64),
        np.asarray(lon1, dtype=np.float64),
        np.asarray(z1, dtype=np.float64),
        np.asarray(weights_2d, dtype=np.float64),
        grid,
        Z_EDGES,
        _LAT_MIN,
        _DLAT,
        _LON_MIN,
        _DLON,
    )
    return grid


class TestProcessSegments:
    """Tests for the batched segment dispatcher (fast path + slow path)."""

    # --- fast path ---

    def test_fast_path_single_voxel(self):
        """A segment within one voxel uses the fast path and accumulates directly."""
        w2d = np.array([[10.0, 5.0]])
        grid = _run_process([0.5], [0.5], [50.0], [0.5], [0.5], [50.0], w2d)

        # lat=0.5, lon=0.5 → i=0, j=0; z=50 → k=0
        assert np.allclose(grid[0, 0, 0, :], [10.0, 5.0])
        assert np.isclose(grid.sum(), 15.0)

    def test_fast_path_out_of_bounds_ignored(self):
        """Out-of-bounds fast-path segment is silently dropped."""
        w2d = np.array([[10.0, 5.0]])
        # lat=100 → i = 100/DLAT = 100 >> NI
        grid = _run_process([100.0], [100.0], [50.0], [100.0], [100.0], [50.0], w2d)
        assert grid.sum() == 0.0

    # --- slow path ---

    def test_slow_path_lat_boundary_equal_split(self):
        """Slow-path segment crossing a latitude boundary splits weight equally."""
        w2d = np.array([[10.0, 5.0]])
        # lat: 0.3 → 1.7, lon and z fixed → crosses lat cell boundary at t=0.5
        grid = _run_process([0.3], [0.5], [50.0], [1.7], [0.5], [50.0], w2d)

        assert np.allclose(grid[0, 0, 0, :], [5.0, 2.5])
        assert np.allclose(grid[1, 0, 0, :], [5.0, 2.5])
        assert np.isclose(grid.sum(), 15.0)

    def test_slow_path_matches_traverse_directly(self):
        """The slow path gives the same result as calling traverse_segment directly.

        This checks that the lat/lon-to-index coordinate transform in
        process_segments is consistent with the index-space inputs expected by
        traverse_segment.
        """
        lat0_v, lon0_v, z0_v = 0.3, 0.3, 50.0
        lat1_v, lon1_v, z1_v = 1.7, 1.7, 250.0
        w = _weights()

        # traverse_segment expects index-space coordinates.
        grid_direct = _fresh_grid()
        i0f = (lat0_v - _LAT_MIN) / _DLAT
        j0f = (lon0_v - _LON_MIN) / _DLON
        i1f = (lat1_v - _LAT_MIN) / _DLAT
        j1f = (lon1_v - _LON_MIN) / _DLON
        traverse_segment_nonuniform_z(
            i0f, j0f, z0_v, i1f, j1f, z1_v, Z_EDGES, grid_direct, w
        )

        # process_segments expects geographic coordinates.
        grid_batch = _run_process(
            [lat0_v], [lon0_v], [z0_v], [lat1_v], [lon1_v], [z1_v], w[np.newaxis, :]
        )

        assert np.allclose(grid_direct, grid_batch)

    # --- batch accumulation ---

    def test_multiple_segments_accumulate_independently(self):
        """Two fast-path segments in separate voxels accumulate without interference."""
        w2d = np.array(
            [
                [1.0, 2.0],  # → cell (0, 0, 0)
                [3.0, 4.0],  # → cell (2, 2, 2)
            ]
        )
        grid = _run_process(
            [0.5, 2.5],
            [0.5, 2.5],
            [50.0, 250.0],
            [0.5, 2.5],
            [0.5, 2.5],
            [50.0, 250.0],
            w2d,
        )
        assert np.allclose(grid[0, 0, 0, :], [1.0, 2.0])
        assert np.allclose(grid[2, 2, 2, :], [3.0, 4.0])
        assert np.isclose(grid.sum(), 10.0)

    def test_weight_conservation_random_batch(self):
        """Total grid sum equals total input weight when all segments stay in bounds."""
        rng = np.random.default_rng(0)
        nseg = 100
        # Start points well inside the grid so small deltas can't escape.
        lat0 = rng.uniform(0.5, NI - 0.5, nseg)
        lon0 = rng.uniform(0.5, NJ - 0.5, nseg)
        z0 = rng.uniform(50.0, 350.0, nseg)
        lat1 = lat0 + rng.uniform(-0.2, 0.2, nseg)
        lon1 = lon0 + rng.uniform(-0.2, 0.2, nseg)
        z1 = z0 + rng.uniform(-40.0, 40.0, nseg)
        w2d = rng.uniform(0.0, 1.0, (nseg, NSPEC))

        grid = _run_process(lat0, lon0, z0, lat1, lon1, z1, w2d)
        assert np.isclose(grid.sum(), w2d.sum(), rtol=1e-6)

    def test_species_weights_are_independent(self):
        """Each species column is accumulated independently."""
        # Two identical segments but with different per-species weights.
        # Species 0 weight is 3× species 1; the ratio should be preserved in the grid.
        w2d = np.array([[9.0, 3.0]])
        grid = _run_process([0.3], [0.3], [50.0], [1.7], [1.7], [250.0], w2d)

        ratio = grid[..., 0].sum() / grid[..., 1].sum()
        assert np.isclose(ratio, 3.0)

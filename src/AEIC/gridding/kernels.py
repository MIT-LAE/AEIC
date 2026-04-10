import numba as nb
import numpy as np


@nb.njit
def traverse_segment_nonuniform_z(
    i0f,
    j0f,
    z0,
    i1f,
    j1f,
    z1,
    z_edges,
    grid,
    weights,
):
    """
    Traverse a single trajectory segment through a 3D grid with:
      - uniform spacing in horizontal (lat/lon)
      - non-uniform spacing in vertical (z_edges)

    and accumulate weighted contributions into grid cells.

    Parameters
    ----------
    i0f, j0f : float
        Start point in *horizontal index space* (continuous).
        i.e. i = (lat - lat_min) / dlat, j = (lon - lon_min) / dlon

    i1f, j1f : float
        End point in horizontal index space.

    z0, z1 : float
        Vertical coordinate at start/end of segment.
        Must be in the same units as z_edges (e.g. pressure or altitude).

    z_edges : array (nlev + 1,)
        Monotonic array of vertical cell boundaries.

    grid : array (ni, nj, nk, nspecies)
        Accumulation grid. Updated in-place.

    weights : array (nspecies,)
        Total emission for this segment for each species.

    Notes
    -----
    - This implements a 3D voxel traversal (Amanatides & Woo style).
    - Horizontal axes are uniform; vertical axis uses explicit boundaries.
    - The segment is parameterized as P(t), t in [0, 1].
    - Each voxel receives a fraction `dt` of the total segment weight.
    - Sum of contributions over all voxels = total segment weight.

    Performance considerations
    --------------------------
    - This is the "slow path" used only when a segment crosses multiple cells.
    - Designed to be branch-light but still readable.
    - No Python calls inside (Numba nopython-compatible).

    Assumptions / limitations
    ------------------------
    - Lat/lon treated as linear coordinates (not true great circles).
    - Vertical coordinate is linear in z (consistent with ERA5 levels).
    - No longitude wrapping handled here (must be preprocessed).
    """

    ni, nj, nk, nspec = grid.shape

    # Direction vector in parametric space
    di = i1f - i0f
    dj = j1f - j0f
    dz = z1 - z0

    # Current voxel indices (integer)
    i = int(np.floor(i0f))
    j = int(np.floor(j0f))
    k = np.searchsorted(z_edges, z0) - 1

    # Clamp vertical index to valid range
    if k < 0:
        k = 0
    elif k >= nk:
        k = nk - 1

    # Step direction along each axis
    step_i = 1 if di > 0 else -1
    step_j = 1 if dj > 0 else -1
    step_k = 1 if dz > 0 else -1

    # Horizontal step size in parameter t
    t_delta_i = abs(1.0 / di) if di != 0.0 else 1e30
    t_delta_j = abs(1.0 / dj) if dj != 0.0 else 1e30

    # Distance to first horizontal boundary
    if di > 0:
        t_max_i = (np.floor(i0f) + 1.0 - i0f) / di
    elif di < 0:
        t_max_i = (i0f - np.floor(i0f)) / -di
    else:
        t_max_i = 1e30

    if dj > 0:
        t_max_j = (np.floor(j0f) + 1.0 - j0f) / dj
    elif dj < 0:
        t_max_j = (j0f - np.floor(j0f)) / -dj
    else:
        t_max_j = 1e30

    # Vertical: first boundary crossing (non-uniform spacing).
    # Mirror the same bounds guards used in the k-advancement section below.
    if dz != 0.0:
        if dz > 0:
            if k + 1 >= z_edges.shape[0]:
                t_max_k = 1e30
            else:
                t_max_k = (z_edges[k + 1] - z0) / dz
        else:
            t_max_k = (z_edges[k] - z0) / dz
    else:
        t_max_k = 1e30

    t = 0.0

    while True:
        # Select next boundary crossing (min of the three)
        axis = 0
        t_next = t_max_i

        if t_max_j < t_next:
            axis = 1
            t_next = t_max_j

        if t_max_k < t_next:
            axis = 2
            t_next = t_max_k

        # If next crossing is beyond segment end → final step
        if t_next >= 1.0:
            dt = 1.0 - t

            if 0 <= i < ni and 0 <= j < nj and 0 <= k < nk:
                for s in range(nspec):
                    grid[i, j, k, s] += weights[s] * dt
            break

        # Accumulate contribution in current voxel
        dt = t_next - t
        if 0 <= i < ni and 0 <= j < nj and 0 <= k < nk:
            for s in range(nspec):
                grid[i, j, k, s] += weights[s] * dt

        t = t_next

        # Advance to next voxel
        if axis == 0:
            i += step_i
            t_max_i += t_delta_i

        elif axis == 1:
            j += step_j
            t_max_j += t_delta_j

        else:
            k += step_k

            # Recompute vertical crossing (non-uniform spacing)
            if dz != 0.0:
                z_curr = z0 + t * dz

                if dz > 0:
                    if k + 1 >= z_edges.shape[0]:
                        t_max_k = 1e30
                    else:
                        z_next = z_edges[k + 1]
                        t_max_k = t + (z_next - z_curr) / dz
                else:
                    if k < 0:
                        t_max_k = 1e30
                    else:
                        z_next = z_edges[k]
                        t_max_k = t + (z_next - z_curr) / dz
            else:
                t_max_k = 1e30


@nb.njit
def process_segments_nonuniform_z(
    lat0, lon0, z0, lat1, lon1, z1, weights, grid, z_edges, lat_min, dlat, lon_min, dlon
):
    """
    Process a batch of trajectory segments and accumulate emissions onto a grid.

    This function implements a two-path strategy:

    1. Fast path:
       Segments entirely within a single voxel → direct accumulation.

    2. Slow path:
       Segments crossing multiple voxels → full voxel traversal.

    Parameters
    ----------
    lat0, lon0, z0 : arrays (nseg,)
        Start points of segments.

    lat1, lon1, z1 : arrays (nseg,)
        End points of segments.

    weights : array (nseg, nspecies)
        Total emissions per segment and species.

    grid : array (ni, nj, nk, nspecies)
        Accumulation grid (updated in-place).

    z_edges : array (nlev + 1,)
        Vertical grid boundaries.

    lat_min, dlat, lon_min, dlon : float
        Horizontal grid definition.

    Notes
    -----
    - This function is designed for chunked processing (e.g. in parallel jobs).
    - No internal parallelism (to avoid race conditions on grid writes).
    - Assumes longitude wrapping has already been handled.
    """

    nseg = lat0.shape[0]
    nspec = weights.shape[1]

    for s in range(nseg):
        # Map endpoints to horizontal index space
        i0f = (lat0[s] - lat_min) / dlat
        j0f = (lon0[s] - lon_min) / dlon
        i1f = (lat1[s] - lat_min) / dlat
        j1f = (lon1[s] - lon_min) / dlon

        # Integer indices (for fast-path check)
        i0 = int(np.floor(i0f))
        j0 = int(np.floor(j0f))
        i1 = int(np.floor(i1f))
        j1 = int(np.floor(j1f))

        k0 = np.searchsorted(z_edges, z0[s]) - 1
        k1 = np.searchsorted(z_edges, z1[s]) - 1

        # Clamp vertical indices
        if k0 < 0:
            k0 = 0
        elif k0 >= grid.shape[2]:
            k0 = grid.shape[2] - 1

        if k1 < 0:
            k1 = 0
        elif k1 >= grid.shape[2]:
            k1 = grid.shape[2] - 1

        # --- Fast path: segment lies entirely within one voxel ---
        if i0 == i1 and j0 == j1 and k0 == k1:
            if 0 <= i0 < grid.shape[0] and 0 <= j0 < grid.shape[1]:
                for sp in range(nspec):
                    grid[i0, j0, k0, sp] += weights[s, sp]
            continue

        # --- Slow path: voxel traversal ---
        traverse_segment_nonuniform_z(
            i0f,
            j0f,
            z0[s],
            i1f,
            j1f,
            z1[s],
            z_edges,
            grid,
            weights[s],
        )

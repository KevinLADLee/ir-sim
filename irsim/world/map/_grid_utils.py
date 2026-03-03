"""Vectorized grid-based collision detection utilities.

All grid collision logic lives here. Both :class:`ObstacleMap` and the
module-level helper ``_grid_collision_geometry`` delegate to
:func:`grid_collision` to avoid duplicated code.

The implementation uses NumPy vectorisation and Shapely 2.x ufuncs
(backed by GEOS C) instead of Python-level loops -- typically 10-50x
faster on grids larger than ~50x50.
"""

from __future__ import annotations

import numpy as np
import shapely

# Grid-based collision detection constants (canonical definitions)
OCCUPANCY_THRESHOLD: float = 50
"""Grid values strictly above this are considered occupied (0-100 scale)."""

CELL_CENTER_OFFSET: float = 0.5
"""Offset to convert a grid index to the cell centre (0.5 = middle)."""

COLLISION_RADIUS_FACTOR: float = 0.5
"""Collision radius as a fraction of the larger cell dimension."""


def grid_collision(
    grid: np.ndarray,
    grid_reso: tuple[float, float],
    geometry: shapely.Geometry,
    world_offset: tuple[float, float] = (0.0, 0.0),
) -> bool:
    """Check whether *geometry* collides with any occupied cell in *grid*.

    A cell is occupied when its value exceeds :data:`OCCUPANCY_THRESHOLD`.
    Collision is defined as ``distance(geometry, cell_centre) <=
    collision_radius`` where ``collision_radius = max(x_reso, y_reso) *
    COLLISION_RADIUS_FACTOR``.  This matches the original per-cell semantic
    exactly, but uses vectorised NumPy + Shapely 2 ufuncs.

    Args:
        grid: Occupancy grid, shape ``(nx, ny)``, values in 0-100.
        grid_reso: Cell size ``(x_reso, y_reso)`` in world metres.
        geometry: Shapely geometry to test.
        world_offset: World-coordinate origin ``(ox, oy)`` of the grid.

    Returns:
        ``True`` if a collision is detected.
    """
    x_reso, y_reso = grid_reso
    offset_x, offset_y = world_offset

    # --- Bounding-box → grid-index range ---
    minx, miny, maxx, maxy = geometry.bounds
    i_min = max(0, int((minx - offset_x) / x_reso))
    i_max = min(grid.shape[0] - 1, int((maxx - offset_x) / x_reso))
    j_min = max(0, int((miny - offset_y) / y_reso))
    j_max = min(grid.shape[1] - 1, int((maxy - offset_y) / y_reso))

    if i_min > i_max or j_min > j_max:
        return False

    # --- Extract occupied cell indices (vectorised) ---
    subgrid = grid[i_min : i_max + 1, j_min : j_max + 1]
    local_ij = np.argwhere(subgrid > OCCUPANCY_THRESHOLD)
    if local_ij.size == 0:
        return False

    # --- Compute cell centres in world coordinates ---
    cx = offset_x + (local_ij[:, 0] + i_min + CELL_CENTER_OFFSET) * x_reso
    cy = offset_y + (local_ij[:, 1] + j_min + CELL_CENTER_OFFSET) * y_reso

    # --- Batch distance check via Shapely 2.x ufunc (GEOS C) ---
    points = shapely.points(cx, cy)
    distances = shapely.distance(geometry, points)
    collision_radius = max(x_reso, y_reso) * COLLISION_RADIUS_FACTOR
    return bool(np.any(distances <= collision_radius))

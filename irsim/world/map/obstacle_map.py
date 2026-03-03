from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiLineString
from shapely.strtree import STRtree

from irsim.world.object_base import ObjectBase

from ._grid_utils import (
    CELL_CENTER_OFFSET,
    COLLISION_RADIUS_FACTOR,
    OCCUPANCY_THRESHOLD,
    grid_collision,
)

# Re-export constants so existing code can import from this module.
__all__ = [
    "CELL_CENTER_OFFSET",
    "COLLISION_RADIUS_FACTOR",
    "OCCUPANCY_THRESHOLD",
    "ObstacleMap",
]


class ObstacleMap(ObjectBase):
    def __init__(
        self,
        shape: dict | None = None,
        color: str = "k",
        static: bool = True,
        grid_map: np.ndarray | None = None,
        grid_reso: np.ndarray | None = None,
        world_offset: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an obstacle map object from a set of line segments.

        Args:
            shape (dict | None): Map shape configuration with keys like
                ``{"name": "map", "reso": float, "points": array}``.
            color (str): Display color. Default "k".
            static (bool): Whether the object is static. Default True.
            grid_map (np.ndarray | None): Grid map array for fast collision detection.
            grid_reso (np.ndarray | None): Resolution [x_reso, y_reso] of the grid.
            world_offset (list | None): World offset [x, y].
            **kwargs: Forwarded to ``ObjectBase`` constructor.
        """
        if shape is None:
            shape = {"name": "map", "reso": "0.1", "points": None}
        super().__init__(
            shape=shape,
            role="obstacle",
            color=color,
            static=static,
            **kwargs,
        )

        self.linestrings = list(self.geometry.geoms)
        self.geometry_tree = STRtree(self.linestrings)

        # Grid-based collision detection data
        self.grid_map = grid_map
        self.grid_reso = (
            grid_reso if grid_reso is not None else np.array([[1.0], [1.0]])
        )
        self.world_offset = world_offset if world_offset is not None else [0.0, 0.0]

    def check_grid_collision(self, geometry) -> bool:
        """Check collision using vectorised grid lookup.

        Delegates to :func:`._grid_utils.grid_collision` which uses NumPy +
        Shapely 2.x ufuncs instead of per-cell Python loops.

        Args:
            geometry: Shapely geometry object to check collision for.

        Returns:
            ``True`` if collision detected.
        """
        if self.grid_map is None:
            return False

        return grid_collision(
            self.grid_map,
            (self.grid_reso[0, 0], self.grid_reso[1, 0]),
            geometry,
            world_offset=(self.world_offset[0], self.world_offset[1]),
        )

    def is_collision(self, geometry) -> bool:
        """Check collision against grid (if present) and map geometry."""
        if self.grid_map is not None and self.check_grid_collision(geometry):
            return True

        candidate_indices = self.geometry_tree.query(geometry)
        filtered_lines = [self.linestrings[i] for i in candidate_indices]
        if not filtered_lines:
            return False
        filtered_multi_line = MultiLineString(filtered_lines)
        return shapely.intersects(geometry, filtered_multi_line)

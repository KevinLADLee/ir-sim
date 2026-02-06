"""

Jump Point Search (JPS) grid planning.

An optimization of A* for uniform-cost grids that prunes symmetric paths
and expands "jump points" only, preserving optimality while reducing nodes expanded.

author: D. Harabor, A. Grastien (original JPS)
adapted by: Reinis Cimurs (grid/collision integration)

See: https://en.wikipedia.org/wiki/Jump_point_search

"""

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import shapely

from irsim.lib.handler.geometry_handler import GeometryFactory
from irsim.world.map import Map


# 8 directions: (dx, dy, cost)
_DIRECTIONS = [
    (1, 0, 1.0),
    (0, 1, 1.0),
    (-1, 0, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
]


class JPSPlanner:
    def __init__(self, env_map: Map, resolution: float) -> None:
        """
        Initialize JPS planner.

        Args:
            env_map (Map): Environment map where planning takes place.
            resolution (float): Grid resolution in meters.
        """
        self.resolution = resolution
        self.obstacle_list = env_map.obstacle_list[:]
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = (
            env_map.height,
            env_map.width,
        )
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

    class Node:
        """Node class (same as A* for path reconstruction)."""

        def __init__(self, x: int, y: int, cost: float, parent_index: int) -> None:
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self) -> str:
            return f"{self.x},{self.y},{self.cost},{self.parent_index}"

    def planning(
        self,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        show_animation: bool = True,
    ) -> np.ndarray:
        """
        JPS path search.

        Args:
            start_pose (np.ndarray): start pose [x, y]
            goal_pose (np.ndarray): goal pose [x, y]
            show_animation (bool): If true, shows the animation of planning process

        Returns:
            np.ndarray: shape (2, N) array [rx, ry] of the final path
        """
        start_node = self.Node(
            self.calc_xy_index(start_pose[0].item(), self.min_x),
            self.calc_xy_index(start_pose[1].item(), self.min_y),
            0.0,
            -1,
        )
        goal_node = self.Node(
            self.calc_xy_index(goal_pose[0].item(), self.min_x),
            self.calc_xy_index(goal_pose[1].item(), self.min_y),
            0.0,
            -1,
        )

        open_set: dict[int, "JPSPlanner.Node"] = {}
        closed_set: dict[int, "JPSPlanner.Node"] = {}
        open_set[self.calc_grid_index_from_xy(start_node.x, start_node.y)] = start_node

        while open_set:
            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost
                + self._heuristic(goal_node.x, goal_node.y, open_set[o].x, open_set[o].y),
            )
            current = open_set[c_id]

            if show_animation:  # pragma: no cover
                plt.plot(
                    self.calc_grid_position(current.x, self.min_x),
                    self.calc_grid_position(current.y, self.min_y),
                    "xc",
                )
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )
                if len(closed_set) % 10 == 0:
                    plt.pause(0.01)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            parent_dx, parent_dy = self._parent_direction(current, closed_set)
            for (dx, dy, step_cost) in self._get_pruned_directions(
                current.x, current.y, parent_dx, parent_dy
            ):
                jump_point = self._jump(
                    current.x, current.y, dx, dy, goal_node.x, goal_node.y
                )
                if jump_point is None:
                    continue
                jx, jy = jump_point
                move_cost = self._move_cost(current.x, current.y, jx, jy, step_cost)
                node = self.Node(
                    jx,
                    jy,
                    current.cost + move_cost,
                    c_id,
                )
                n_id = self.calc_grid_index_from_xy(jx, jy)

                if n_id in closed_set:
                    continue
                if n_id not in open_set or open_set[n_id].cost > node.cost:
                    open_set[n_id] = node

        rx, ry = self._calc_final_path(goal_node, closed_set)
        return np.array([rx, ry])

    def _parent_direction(
        self,
        node: "JPSPlanner.Node",
        closed_set: dict[int, "JPSPlanner.Node"],
    ) -> tuple[Optional[int], Optional[int]]:
        """Return (dx, dy) from parent to node, or (None, None) for start."""
        if node.parent_index == -1:
            return (None, None)
        parent = closed_set[node.parent_index]
        dx = node.x - parent.x
        dy = node.y - parent.y
        # normalize to -1, 0, 1
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        return (dx, dy)

    def _get_pruned_directions(
        self,
        x: int,
        y: int,
        parent_dx: Optional[int],
        parent_dy: Optional[int],
    ) -> list[tuple[int, int, float]]:
        """Return list of (dx, dy, cost) to try from (x,y) given parent direction."""
        if parent_dx is None and parent_dy is None:
            return _DIRECTIONS

        out: list[tuple[int, int, float]] = []
        is_diag = parent_dx != 0 and parent_dy != 0
        if is_diag:
            out.append((parent_dx, parent_dy, math.sqrt(2)))
            out.append((parent_dx, 0, 1.0))
            out.append((0, parent_dy, 1.0))
            return out

        # Cardinal: natural direction + forced diagonals only
        out.append((parent_dx or 0, parent_dy or 0, 1.0))
        if (parent_dx, parent_dy) == (1, 0):
            if not self._is_walkable(x, y + 1):
                out.append((1, 1, math.sqrt(2)))
            if not self._is_walkable(x, y - 1):
                out.append((1, -1, math.sqrt(2)))
        elif (parent_dx, parent_dy) == (-1, 0):
            if not self._is_walkable(x, y + 1):
                out.append((-1, 1, math.sqrt(2)))
            if not self._is_walkable(x, y - 1):
                out.append((-1, -1, math.sqrt(2)))
        elif (parent_dx, parent_dy) == (0, 1):
            if not self._is_walkable(x + 1, y):
                out.append((1, 1, math.sqrt(2)))
            if not self._is_walkable(x - 1, y):
                out.append((-1, 1, math.sqrt(2)))
        elif (parent_dx, parent_dy) == (0, -1):
            if not self._is_walkable(x + 1, y):
                out.append((1, -1, math.sqrt(2)))
            if not self._is_walkable(x - 1, y):
                out.append((-1, -1, math.sqrt(2)))
        return out

    def _jump(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        gx: int,
        gy: int,
    ) -> Optional[tuple[int, int]]:
        """Jump from (x,y) in direction (dx,dy); return (jx, jy) or None."""
        nx, ny = x + dx, y + dy
        if not self._is_walkable(nx, ny):
            return None
        if nx == gx and ny == gy:
            return (nx, ny)

        is_diag = dx != 0 and dy != 0
        if is_diag:
            if self._has_forced_neighbor_diag(nx, ny, dx, dy):
                return (nx, ny)
            if self._jump(nx, ny, dx, dy, gx, gy) is not None:
                return (nx, ny)
            if self._jump(nx, ny, dx, 0, gx, gy) is not None:
                return (nx, ny)
            if self._jump(nx, ny, 0, dy, gx, gy) is not None:
                return (nx, ny)
            return None

        if self._has_forced_neighbor_cardinal(nx, ny, dx, dy):
            return (nx, ny)
        return self._jump(nx, ny, dx, dy, gx, gy)

    def _has_forced_neighbor_cardinal(self, x: int, y: int, dx: int, dy: int) -> bool:
        """True if (x,y) has a forced neighbor when approached along cardinal (dx,dy)."""
        if dx == 1 and dy == 0:
            return (not self._is_walkable(x, y + 1)) or (not self._is_walkable(x, y - 1))
        if dx == -1 and dy == 0:
            return (not self._is_walkable(x, y + 1)) or (not self._is_walkable(x, y - 1))
        if dx == 0 and dy == 1:
            return (not self._is_walkable(x + 1, y)) or (not self._is_walkable(x - 1, y))
        if dx == 0 and dy == -1:
            return (not self._is_walkable(x + 1, y)) or (not self._is_walkable(x - 1, y))
        return False

    def _has_forced_neighbor_diag(self, x: int, y: int, dx: int, dy: int) -> bool:
        """True if (x,y) has a forced neighbor when approached along diagonal (dx,dy)."""
        if not self._is_walkable(x - dx, y + dy) or not self._is_walkable(x + dx, y - dy):
            return True
        return False

    def _is_walkable(self, ix: int, iy: int) -> bool:
        """True if grid cell (ix, iy) is in bounds and not in collision."""
        px = self.calc_grid_position(ix, self.min_x)
        py = self.calc_grid_position(iy, self.min_y)
        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False
        return not self._check_collision(px, py)

    def _move_cost(self, x1: int, y1: int, x2: int, y2: int, step_cost: float) -> float:
        """Cost of moving from (x1,y1) to (x2,y2) with given step cost."""
        steps = max(abs(x2 - x1), abs(y2 - y1))
        return steps * step_cost

    def _heuristic(self, gx: int, gy: int, x: int, y: int) -> float:
        """Octile heuristic for 8-directional grid."""
        dx = abs(gx - x)
        dy = abs(gy - y)
        return (dx + dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def _calc_final_path(
        self,
        goal_node: "JPSPlanner.Node",
        closed_set: dict[int, "JPSPlanner.Node"],
    ) -> tuple[list[float], list[float]]:
        rx = [self.calc_grid_position(goal_node.x, self.min_x)]
        ry = [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx, ry

    def calc_grid_position(self, index: int, min_position: float) -> float:
        return index * self.resolution + min_position

    def calc_xy_index(self, position: float, min_pos: float) -> int:
        return round((position - min_pos) / self.resolution)

    def calc_grid_index_from_xy(self, x: int, y: int) -> int:
        return (y - self.min_y) * self.x_width + (x - self.min_x)

    def _check_collision(self, x: float, y: float) -> bool:
        """True if world position (x,y) is in collision (reuses A* logic)."""
        node_position = [x, y]
        shape = {
            "name": "rectangle",
            "length": self.resolution,
            "width": self.resolution,
        }
        gf = GeometryFactory.create_geometry(**shape)
        geometry = gf.step(np.c_[node_position])
        return any(
            shapely.intersects(geometry, obj._geometry) for obj in self.obstacle_list
        )

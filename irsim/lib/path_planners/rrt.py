"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

adapted by: Reinis Cimurs

"""

import math
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely import Point as ShapelyPoint
from shapely.affinity import translate as shapely_translate

from irsim.lib.handler.geometry_handler import GeometryFactory
from irsim.world.map import Map


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x: float, y: float) -> None:
            """
            Initialize Node

            Args:
                x (float): x position of the node
                y (float): y position of the node
            """
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:
        """
        Area Bounds
        """

        def __init__(self, env_map: Map) -> None:
            """
            Initialize AreaBounds

            Args:
                env_map (EnvBase): environment where the planning will take place
            """
            self.xmin, self.ymin = 0, 0
            self.xmax, self.ymax = (
                env_map.width,
                env_map.height,
            )

    def __init__(
        self,
        env_map: Map,
        robot_radius: float,
        expand_dis: float = 1.0,
        path_resolution: float = 0.25,
        goal_sample_rate: int = 5,
        max_iter: int = 500,
    ) -> None:
        """
        Initialize RRT planner

        Args:
            env_map (Env): environment map where the planning will take place
            robot_radius (float): robot body modeled as circle with given radius
            expand_dis (float): expansion distance
            path_resolution (float): resolution of the path
            goal_sample_rate (int): goal sample rate
            max_iter (int): max iteration count
        """
        self.obstacle_list = env_map.obstacle_list[:]
        self.max_x, self.max_y = (
            env_map.width,
            env_map.height,
        )
        self.play_area = self.AreaBounds(env_map)
        self.min_rand = 0.0
        self.max_rand = max(self.max_x, self.max_y)
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        self.robot_radius = robot_radius

        # --- collision optimisation ---
        # Grid-based fast path (O(1) per point when grid is available)
        self._grid: Optional[np.ndarray] = getattr(env_map, "grid", None)
        if self._grid is not None:
            self._grid_x_reso = env_map.width / self._grid.shape[0]
            self._grid_y_reso = env_map.height / self._grid.shape[1]
            self._rr_cells_x = max(1, int(np.ceil(robot_radius / self._grid_x_reso)))
            self._rr_cells_y = max(1, int(np.ceil(robot_radius / self._grid_y_reso)))
        else:
            self._grid_x_reso = 0.0
            self._grid_y_reso = 0.0
            self._rr_cells_x = 0
            self._rr_cells_y = 0

        # Cached shapely circle + prepared obstacles (fallback path)
        self._collision_circle = ShapelyPoint(0, 0).buffer(robot_radius)
        for obj in self.obstacle_list:
            shapely.prepare(obj._geometry)

        # --- visualisation state ---
        self._vis_temp: list = []       # transient artists cleared each frame
        self._vis_setup_done: bool = False
        self._tree_line = None          # single Line2D for all tree edges

    def planning(
        self,
        start_pose: list[float],
        goal_pose: list[float],
        show_animation: bool = True,
    ) -> Optional[tuple[list[float], list[float]]]:
        """
        rrt path planning

        Args:
            start_pose (np.array): start pose [x,y]
            goal_pose (np.array): goal pose [x,y]
            show_animation (bool): If true, shows the animation of planning process

        Returns:
            (np.array): xy position array of the final path
        """
        self.start = self.Node(start_pose[0].item(), start_pose[1].item())
        self.end = self.Node(goal_pose[0].item(), goal_pose[1].item())

        self.node_list = [self.start]
        for _i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(
                new_node, self.play_area
            ) and self.check_collision(new_node, self.robot_radius):
                self.node_list.append(new_node)

                if show_animation:
                    self.draw_graph(new_node)

            if (
                self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y)
                <= self.expand_dis
            ):
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    def steer(
        self, from_node: "Node", to_node: "Node", extend_length: float = float("inf")
    ) -> "Node":
        """
        Generate a new node by steering from `from_node` towards `to_node`.

        This method incrementally moves from `from_node` in the direction of `to_node`,
        using a fixed step size (`self.path_resolution`) and not exceeding the
        specified `extend_length`. The result is a new node that approximates a path
        from the start node toward the goal, constrained by resolution and maximum
        step distance.

        If the final position is within one resolution step of `to_node`, it snaps the
        new node exactly to `to_node`.

        Args:
            from_node (Node): The node from which to begin extending.
            to_node (Node): The target node to steer toward.
            extend_length (float, optional): The maximum length to extend. Defaults to infinity.

        Returns:
            (Node): A new node with updated position, path history (path_x, path_y),
        """
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind: int) -> tuple[list[float], list[float]]:
        """
        Generate the final path

        Args:
            goal_ind (int): index of the final goal

        Returns:
            (np.array): xy position array of the final path
        """
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        rx = [node[0] for node in path]
        ry = [node[1] for node in path]
        return np.array([rx, ry])

    def calc_dist_to_goal(self, x: float, y: float) -> float:
        """
        Calculate distance to goal

        Args:
            x (float): x coordinate of the position
            y (float): y coordinate of the position

        Returns:
            (float): distance to the goal
        """
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self) -> "Node":
        """
        Create random node

        Returns:
            (Node): new random node
        """
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
            )
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd: Optional["Node"] = None) -> None:
        """Render the RRT exploration tree on the active matplotlib axes.

        Uses a single ``Line2D`` object (with NaN separators) for the
        entire tree so that repeated calls do not accumulate separate
        artist objects.  The random-sample marker is transient and is
        removed at the start of each subsequent call.

        Axis limits and aspect ratio are **not** modified — these are
        managed by the ir-sim ``EnvPlot``.

        Args:
            rnd (Node | None): Optional node to highlight (e.g. the
                latest random sample or newly added node).
        """
        ax = plt.gca()

        # --- remove transient markers from previous frame ---
        for a in self._vis_temp:
            a.remove()
        self._vis_temp.clear()

        # --- one-time setup (keyboard, start/goal, play-area) ---
        if not self._vis_setup_done:
            ax.figure.canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            ax.plot(self.start.x, self.start.y, "xr", markersize=8, zorder=5)
            ax.plot(self.end.x, self.end.y, "xr", markersize=8, zorder=5)
            if self.play_area is not None:
                ax.plot(
                    [
                        self.play_area.xmin, self.play_area.xmax,
                        self.play_area.xmax, self.play_area.xmin,
                        self.play_area.xmin,
                    ],
                    [
                        self.play_area.ymin, self.play_area.ymin,
                        self.play_area.ymax, self.play_area.ymax,
                        self.play_area.ymin,
                    ],
                    "-k", linewidth=0.6,
                )
            self._vis_setup_done = True

        # --- transient: random-sample marker & collision circle ---
        if rnd is not None:
            (marker,) = ax.plot(rnd.x, rnd.y, "^k")
            self._vis_temp.append(marker)
            if self.robot_radius > 0.0:
                circ = self.plot_circle(rnd.x, rnd.y, self.robot_radius, "-r", ax=ax)
                self._vis_temp.append(circ)

        # --- tree edges (single Line2D, updated in-place) ---
        xs: list[float] = []
        ys: list[float] = []
        for node in self.node_list:
            if node.parent:
                xs.extend(node.path_x)
                ys.extend(node.path_y)
                xs.append(float("nan"))
                ys.append(float("nan"))

        if self._tree_line is None and xs:
            (self._tree_line,) = ax.plot(xs, ys, "-g", linewidth=0.5)
        elif self._tree_line is not None:
            self._tree_line.set_data(xs, ys)

        plt.pause(0.01)

    @staticmethod
    def plot_circle(
        x: float, y: float, size: float, color: str = "-b", ax: Optional[plt.Axes] = None,
    ) -> plt.Line2D:  # pragma: no cover
        """Plot a circle at a given position and return the ``Line2D`` artist.

        Args:
            x (float): Center x coordinate.
            y (float): Center y coordinate.
            size (float): Circle radius.
            color (str): Matplotlib color/style string.
            ax (Axes | None): Target axes; defaults to ``plt.gca()``.

        Returns:
            Line2D: The plotted circle artist.
        """
        if ax is None:
            ax = plt.gca()
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        (line,) = ax.plot(xl, yl, color)
        return line

    @staticmethod
    def get_nearest_node_index(node_list: list["Node"], rnd_node: "Node") -> int:
        """Return the index of the nearest node in the list to a target node.

        Uses numpy vectorised distance computation for speed.

        Args:
            node_list (list[Node]): List of existing nodes.
            rnd_node (Node): Target node to compare distances against.

        Returns:
            int: Index of the nearest node.
        """
        coords = np.array([[n.x, n.y] for n in node_list])
        dists_sq = (coords[:, 0] - rnd_node.x) ** 2 + (coords[:, 1] - rnd_node.y) ** 2
        return int(np.argmin(dists_sq))

    @staticmethod
    def check_if_outside_play_area(node: "Node", play_area: "AreaBounds") -> bool:
        """Check whether the node is inside the defined play area bounds.

        Args:
            node (Node): Node to check.
            play_area (AreaBounds): World bounds.

        Returns:
            bool: True if inside bounds (or no bounds defined); False otherwise.
        """
        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        return not (
            node.x < play_area.xmin
            or node.x > play_area.xmax
            or node.y < play_area.ymin
            or node.y > play_area.ymax
        )

    def check_collision(self, node: "Node", robot_radius: float) -> bool:
        """
        Check if node is acceptable - free of collisions.

        Uses a fast grid-based lookup when a grid map is available,
        otherwise falls back to cached Shapely geometry intersection.

        Args:
            node (Node): node to check
            robot_radius (float): robot radius

        Returns:
            (bool): True if there is no collision. False otherwise
        """
        if node is None:
            return False

        # ---- fast grid path (all points at once) ----
        if self._grid is not None:
            return self._check_collision_grid(node)

        # ---- shapely fallback (cached circle) ----
        for i in range(len(node.path_x)):
            if self._check_node_shapely(node.path_x[i], node.path_y[i]):
                return False  # collision
        return not self._check_node_shapely(node.x, node.y)

    # -- grid-based collision (vectorised numpy) ---------------------------

    def _check_collision_grid(self, node: "Node") -> bool:
        """Grid-based collision check for all path points using slice views.

        For each point, extracts a small subgrid around the robot radius
        and checks for occupied cells. Uses numpy slice views (O(1)
        allocation) instead of fancy indexing.

        Returns True if the path is **collision-free**.
        """
        grid = self._grid
        rows, cols = grid.shape
        inv_xr = 1.0 / self._grid_x_reso
        inv_yr = 1.0 / self._grid_y_reso
        rr_x = self._rr_cells_x
        rr_y = self._rr_cells_y

        for x, y in zip(node.path_x, node.path_y):
            gx = int(x * inv_xr)
            gy = int(y * inv_yr)
            if np.any(grid[max(0, gx - rr_x):min(rows, gx + rr_x + 1),
                           max(0, gy - rr_y):min(cols, gy + rr_y + 1)] > 50):
                return False

        gx = int(node.x * inv_xr)
        gy = int(node.y * inv_yr)
        if np.any(grid[max(0, gx - rr_x):min(rows, gx + rr_x + 1),
                       max(0, gy - rr_y):min(cols, gy + rr_y + 1)] > 50):
            return False

        return True

    # -- shapely fallback (cached geometry) --------------------------------

    def _check_node_shapely(self, x: float, y: float) -> bool:
        """Check a single point for collision using the cached circle.

        Returns True if **collision detected**.
        """
        moved = shapely_translate(self._collision_circle, xoff=x, yoff=y)
        return any(
            shapely.intersects(moved, obj._geometry) for obj in self.obstacle_list
        )

    def check_node(self, x: float, y: float, rr: float) -> bool:
        """Check position for a collision (legacy API, kept for compatibility).

        Args:
            x (float): x value of the position
            y (float): y value of the position
            rr (float): robot radius

        Returns:
            (bool): True if there is a collision. False otherwise
        """
        if self._grid is not None:
            gx = int(x / self._grid_x_reso)
            gy = int(y / self._grid_y_reso)
            rows, cols = self._grid.shape
            rr_cx = max(1, int(np.ceil(rr / self._grid_x_reso)))
            rr_cy = max(1, int(np.ceil(rr / self._grid_y_reso)))
            x_lo = max(0, gx - rr_cx)
            x_hi = min(rows - 1, gx + rr_cx)
            y_lo = max(0, gy - rr_cy)
            y_hi = min(cols - 1, gy + rr_cy)
            return bool(np.any(self._grid[x_lo:x_hi + 1, y_lo:y_hi + 1] > 50))
        return self._check_node_shapely(x, y)

    @staticmethod
    def calc_distance_and_angle(
        from_node: "Node", to_node: "Node"
    ) -> tuple[float, float]:
        """Compute Euclidean distance and heading from one node to another.

        Args:
            from_node (Node): Start node.
            to_node (Node): Target node.

        Returns:
            tuple[float, float]: Distance and angle (radians) toward the target.
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

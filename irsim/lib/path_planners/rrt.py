"""
Rapidly-exploring Random Tree (RRT) path planner.

Collision precedence:
  1. Grid lookup (O(1) per cell) when ``env_map.grid`` is not ``None``.
  2. Shapely geometry intersection when the grid is unavailable.

Reference:
    S. M. LaValle, "Rapidly-Exploring Random Trees: A New Tool for Path
    Planning," 1998.

Implementation reference:
    ZJU-FAST-Lab/sampling-based-path-finding
    https://github.com/ZJU-FAST-Lab/sampling-based-path-finding
    (C++ implementation with kd-tree and efficient tree management.)

Adapted for ir-sim.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely import Point as ShapelyPoint
from shapely.affinity import translate as shapely_translate

from irsim.world.map import EnvGridMap

if TYPE_CHECKING:
    from matplotlib.lines import Line2D

from scipy.spatial import KDTree as _SciKDTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree node (shared by RRT, RRT*, Informed RRT*)
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class TreeNode:
    """A node in the RRT search tree.

    Identity-based equality (``eq=False``) is used so that nodes can be
    safely compared with ``==`` / ``is`` and used as dictionary keys
    without triggering recursive field comparisons.

    Attributes:
        x: X position.
        y: Y position.
        cost: Cumulative cost from the start node (``cost_from_start``).
        cost_from_parent: Cost of the edge from parent to this node.
        parent: Parent node (``None`` for the root).
        children: Direct children in the tree.
        path_x: Discretised X coordinates from parent to this node
                 (used for visualisation and per-point collision checking).
        path_y: Discretised Y coordinates from parent to this node.
    """

    x: float
    y: float
    cost: float = 0.0
    cost_from_parent: float = 0.0
    parent: TreeNode | None = None
    children: list[TreeNode] = field(default_factory=list, repr=False)
    path_x: list[float] = field(default_factory=list, repr=False)
    path_y: list[float] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# RRT planner
# ---------------------------------------------------------------------------


class RRT:
    """Rapidly-exploring Random Tree (RRT) path planner.

    Algorithmic improvements over the naive implementation (inspired by
    ZJU-FAST-Lab/sampling-based-path-finding):

    * Nodes track their children, enabling O(subtree) cost propagation
      via BFS instead of O(n) linear scans.
    * ``scipy.spatial.KDTree`` is used for nearest-neighbour and range
      queries when available, falling back to numpy otherwise.
    * A dedicated *goal node* is kept in the tree; whenever a new node
      can reach the goal at a lower cost the goal's parent is rewired.
    """

    # Expose TreeNode as a class-level alias for backward compatibility
    Node = TreeNode

    class AreaBounds:
        """Rectangular play-area bounds."""

        __slots__ = ("xmin", "ymin", "xmax", "ymax")

        def __init__(self, env_map: EnvGridMap) -> None:
            self.xmin: float = 0.0
            self.ymin: float = 0.0
            self.xmax: float = float(env_map.width)
            self.ymax: float = float(env_map.height)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        env_map: EnvGridMap,
        robot_radius: float,
        expand_dis: float = 1.0,
        path_resolution: float = 0.25,
        goal_sample_rate: int = 5,
        max_iter: int = 500,
    ) -> None:
        """Initialise the RRT planner.

        Args:
            env_map: Environment map (any :class:`~irsim.world.map.EnvGridMap`
                compatible object).
            robot_radius: Robot body modelled as a circle with this radius.
            expand_dis: Maximum extension distance per steer step.
            path_resolution: Discretisation resolution along steered edges.
            goal_sample_rate: Percentage chance of sampling the goal.
            max_iter: Maximum number of iterations.
        """
        self._map = env_map
        self.obstacle_list = env_map.obstacle_list[:]
        self.max_x: float = float(env_map.width)
        self.max_y: float = float(env_map.height)
        self.play_area = self.AreaBounds(env_map)
        self.min_rand: float = 0.0
        self.max_rand: float = max(self.max_x, self.max_y)
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.robot_radius = robot_radius

        # Tree state (reset each planning call)
        self.node_list: list[TreeNode] = []
        self.start: TreeNode | None = None
        self.end: TreeNode | None = None

        self._collision_circle = ShapelyPoint(0, 0).buffer(robot_radius)
        for obj in self.obstacle_list:
            shapely.prepare(obj._geometry)

        # --- KDTree state ---
        self._kd_coords: np.ndarray | None = None
        self._kd_tree: _SciKDTree | None = None
        self._kd_dirty: bool = True

        # --- visualisation state ---
        self._vis_temp: list = []
        self._vis_setup_done: bool = False
        self._tree_line: Line2D | None = None

    # ------------------------------------------------------------------
    # Tree management helpers (reference-style)
    # ------------------------------------------------------------------

    def _add_tree_node(
        self,
        parent: TreeNode,
        x: float,
        y: float,
        cost_from_parent: float,
        path_x: list[float] | None = None,
        path_y: list[float] | None = None,
    ) -> TreeNode:
        """Create a ``TreeNode``, attach it to *parent*, and register it."""
        node = TreeNode(
            x=x,
            y=y,
            cost=parent.cost + cost_from_parent,
            cost_from_parent=cost_from_parent,
            parent=parent,
            children=[],
            path_x=path_x if path_x is not None else [],
            path_y=path_y if path_y is not None else [],
        )
        parent.children.append(node)
        self.node_list.append(node)
        self._kd_dirty = True
        return node

    def _change_node_parent(
        self,
        node: TreeNode,
        new_parent: TreeNode,
        cost_from_parent: float,
    ) -> None:
        """Re-parent *node* and BFS-propagate costs to all descendants.

        Mirrors ``changeNodeParent`` from the C++ reference.
        """
        if node.parent is not None:
            node.parent.children.remove(node)
        node.parent = new_parent
        node.cost_from_parent = cost_from_parent
        node.cost = new_parent.cost + cost_from_parent
        new_parent.children.append(node)

        # BFS cost propagation
        queue: deque[TreeNode] = deque(node.children)
        while queue:
            child = queue.popleft()
            child.cost = child.parent.cost + child.cost_from_parent
            queue.extend(child.children)

    def _fill_path(self, node: TreeNode) -> np.ndarray:
        """Trace parent chain from *node* to root, return ``(2, N)`` array.

        Path order is *node* to root (e.g. goal -> start).
        """
        path_x: list[float] = []
        path_y: list[float] = []
        current: TreeNode | None = node
        while current is not None:
            path_x.append(current.x)
            path_y.append(current.y)
            current = current.parent
        return np.array([path_x, path_y])

    # ------------------------------------------------------------------
    # KDTree helpers
    # ------------------------------------------------------------------

    def _rebuild_kd_tree(self) -> None:
        """Rebuild the spatial index from the current ``node_list``."""
        if not self._kd_dirty or not self.node_list:
            return
        self._kd_coords = np.array([[n.x, n.y] for n in self.node_list])
        self._kd_tree = _SciKDTree(self._kd_coords)
        self._kd_dirty = False

    def _nearest(self, x: float, y: float) -> int:
        """Index of the closest node to ``(x, y)``."""
        self._rebuild_kd_tree()
        _, idx = self._kd_tree.query([x, y])
        return int(idx)

    def _near_in_radius(self, x: float, y: float, radius: float) -> list[int]:
        """Indices of all nodes within *radius* of ``(x, y)``."""
        self._rebuild_kd_tree()
        return self._kd_tree.query_ball_point([x, y], radius)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def planning(
        self,
        start_pose: list[float],
        goal_pose: list[float],
        show_animation: bool = True,
    ) -> np.ndarray | None:
        """Run RRT path planning.

        Args:
            start_pose: Start position ``[x, y]``.
            goal_pose: Goal position ``[x, y]``.
            show_animation: Render the exploration tree during planning.

        Returns:
            ``(2, N)`` numpy array ``[rx, ry]`` or *None*.
        """
        sx, sy = float(start_pose[0]), float(start_pose[1])
        gx, gy = float(goal_pose[0]), float(goal_pose[1])

        self.start = TreeNode(x=sx, y=sy, cost=0.0)
        self.end = TreeNode(x=gx, y=gy, cost=float("inf"))

        # Reset
        self.node_list = [self.start]
        self._kd_dirty = True
        self._vis_setup_done = False
        self._tree_line = None
        self._vis_temp = []

        for _ in range(self.max_iter):
            # 1. Sample
            rnd_node = self.get_random_node()

            # 2. Nearest
            nearest_ind = self._nearest(rnd_node.x, rnd_node.y)
            nearest_node = self.node_list[nearest_ind]

            # 3. Steer
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 4. Bounds + collision
            if not self._check_bounds(new_node.x, new_node.y):
                continue
            if not self.check_collision(new_node, self.robot_radius):
                continue

            # 5. Add to tree
            cost_fp = math.hypot(
                new_node.x - nearest_node.x, new_node.y - nearest_node.y,
            )
            added = self._add_tree_node(
                nearest_node,
                new_node.x,
                new_node.y,
                cost_fp,
                path_x=new_node.path_x,
                path_y=new_node.path_y,
            )

            if show_animation:
                self.draw_graph(added)

            # 6. Try to connect to goal
            dist_to_goal = math.hypot(
                added.x - self.end.x, added.y - self.end.y,
            )
            if dist_to_goal <= self.expand_dis:
                goal_edge = self.steer(added, self.end, self.expand_dis)
                if self.check_collision(goal_edge, self.robot_radius):
                    # For basic RRT, return on the first feasible connection
                    self.end.parent = added
                    self.end.cost_from_parent = dist_to_goal
                    self.end.cost = added.cost + dist_to_goal
                    self.end.path_x = goal_edge.path_x
                    self.end.path_y = goal_edge.path_y
                    return self._fill_path(self.end)

        return None  # cannot find path

    # ------------------------------------------------------------------
    # Steer
    # ------------------------------------------------------------------

    def steer(
        self,
        from_node: TreeNode,
        to_node: TreeNode,
        extend_length: float = float("inf"),
    ) -> TreeNode:
        """Steer from *from_node* towards *to_node*.

        Returns a **temporary** ``TreeNode`` (not yet registered in the
        tree) with ``path_x``/``path_y`` populated and ``parent`` set to
        *from_node*.
        """
        d, theta = self.calc_distance_and_angle(from_node, to_node)
        if extend_length > d:
            extend_length = d

        new_node = TreeNode(x=from_node.x, y=from_node.y)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        n_expand = math.floor(extend_length / self.path_resolution)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * cos_t
            new_node.y += self.path_resolution * sin_t
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d2, _ = self.calc_distance_and_angle(new_node, to_node)
        if d2 <= self.path_resolution:
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node
        return new_node

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_random_node(self) -> TreeNode:
        """Uniform random sample with goal-bias."""
        if random.randint(0, 100) > self.goal_sample_rate:
            return TreeNode(
                x=random.uniform(self.min_rand, self.max_rand),
                y=random.uniform(self.min_rand, self.max_rand),
            )
        return TreeNode(x=self.end.x, y=self.end.y)

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_bounds(self, x: float, y: float) -> bool:
        """Return *True* if ``(x, y)`` lies inside the play area."""
        pa = self.play_area
        return pa.xmin <= x <= pa.xmax and pa.ymin <= y <= pa.ymax

    def check_collision(self, node: TreeNode, robot_radius: float) -> bool:
        """Check whether *node*'s edge is collision-free.

        Delegates to :meth:`Map.grid_occupied` when a grid is present;
        falls back to Shapely geometry for non-grid obstacles.

        Returns:
            *True* if the path is **collision-free**.
        """
        if node is None:
            return False

        rr = robot_radius
        # Check each point along the discretised edge
        for px, py in zip(node.path_x, node.path_y):
            if self._check_point(px, py, rr):
                return False
        # Also check the node endpoint itself
        return not self._check_point(node.x, node.y, rr)

    def _check_point(self, x: float, y: float, rr: float) -> bool:
        """Single-point collision check.

        Returns *True* if a **collision is detected**.
        """
        # Grid path (fast)
        grid_hit = self._map.grid_occupied(x, y, margin_x=rr, margin_y=rr)
        if grid_hit is True:
            return True
        if grid_hit is False and not self.obstacle_list:
            return False
        # Shapely fallback (or combined when obstacle_list exists alongside grid)
        moved = shapely_translate(self._collision_circle, xoff=x, yoff=y)
        return any(
            shapely.intersects(moved, obj._geometry) for obj in self.obstacle_list
        )

    def check_node(self, x: float, y: float, rr: float) -> bool:
        """Point collision check (legacy API).

        Returns *True* if there **is** a collision.
        """
        return self._check_point(x, y, rr)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw_graph(self, rnd: TreeNode | None = None) -> None:
        """Render the RRT tree on the active matplotlib axes."""
        ax = plt.gca()

        # Remove transient markers from previous frame
        for a in self._vis_temp:
            a.remove()
        self._vis_temp.clear()

        # One-time setup
        if not self._vis_setup_done:
            ax.figure.canvas.mpl_connect(
                "key_release_event",
                lambda event: plt.close(event.canvas.figure) if event.key == "escape" else None,
            )
            ax.plot(self.start.x, self.start.y, "xr", markersize=8, zorder=5)
            ax.plot(self.end.x, self.end.y, "xr", markersize=8, zorder=5)
            if self.play_area is not None:
                pa = self.play_area
                ax.plot(
                    [pa.xmin, pa.xmax, pa.xmax, pa.xmin, pa.xmin],
                    [pa.ymin, pa.ymin, pa.ymax, pa.ymax, pa.ymin],
                    "-k",
                    linewidth=0.6,
                )
            self._vis_setup_done = True
            plt.pause(0.05)

        # Transient markers
        if rnd is not None:
            (marker,) = ax.plot(rnd.x, rnd.y, "^k")
            self._vis_temp.append(marker)
            if self.robot_radius > 0.0:
                circ = self._plot_circle(
                    rnd.x, rnd.y, self.robot_radius, "-r", ax=ax
                )
                self._vis_temp.append(circ)

        # Tree edges (single Line2D, updated in-place)
        xs: list[float] = []
        ys: list[float] = []
        for node in self.node_list:
            if node.parent and node.path_x:
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
    def _plot_circle(
        x: float,
        y: float,
        size: float,
        color: str = "-b",
        ax: plt.Axes | None = None,
    ) -> Line2D:
        """Plot a circle and return the ``Line2D`` artist."""
        if ax is None:
            ax = plt.gca()
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        (line,) = ax.plot(xl, yl, color)
        return line

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def calc_dist_to_goal(self, x: float, y: float) -> float:
        """Euclidean distance from ``(x, y)`` to the goal."""
        return math.hypot(x - self.end.x, y - self.end.y)

    @staticmethod
    def calc_distance_and_angle(
        from_node: TreeNode,
        to_node: TreeNode,
    ) -> tuple[float, float]:
        """Euclidean distance and heading between two nodes."""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

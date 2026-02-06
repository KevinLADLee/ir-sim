"""

Path planning with Informed RRT*

Informed RRT* improves upon RRT* by constraining the sampling region to an
ellipsoidal subset of the planning space once an initial solution has been
found.  The ellipse is defined by the start and goal positions as foci, and
the current best path cost determines its size.  As the cost decreases, the
ellipse shrinks, focusing exploration on the region that can actually produce
shorter paths.

Reference:
    J. D. Gammell, S. S. Srinivasa, and T. D. Barfoot,
    "Informed RRT*: Optimal Sampling-based Path Planning Focused via Direct
    Sampling of an Admissible Ellipsoidal Heuristic," in Proc. IEEE/RSJ
    Int. Conf. Intelligent Robots and Systems (IROS), 2014.

adapted for ir-sim

"""

import math
import random
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np

from irsim.lib.path_planners.rrt_star import RRTStar
from irsim.world.map import Map


class InformedRRTStar(RRTStar):
    """
    Informed RRT* path planner.

    After finding an initial feasible path the sampler switches from uniform
    random sampling to sampling inside an informed ellipsoidal set whose
    foci are the start and goal.  This dramatically speeds up convergence
    towards the optimal path.
    """

    def __init__(
        self,
        env_map: Map,
        robot_radius: float,
        expand_dis: float = 1.5,
        path_resolution: float = 0.25,
        goal_sample_rate: int = 10,
        max_iter: int = 500,
        connect_circle_dist: float = 50.0,
        search_until_max_iter: bool = True,
    ) -> None:
        """
        Initialize Informed RRT* planner.

        Args:
            env_map (Map): environment map where the planning will take place.
            robot_radius (float): robot body modeled as circle with given radius.
            expand_dis (float): expansion distance per steer step.
            path_resolution (float): resolution of the path.
            goal_sample_rate (int): percentage chance of sampling the goal
                (used before any feasible path is found).
            max_iter (int): maximum iteration count.
            connect_circle_dist (float): connection / rewiring radius parameter.
            search_until_max_iter (bool): always keep searching until max_iter
                (Informed RRT* always does this).
        """
        super().__init__(
            env_map,
            robot_radius,
            expand_dis,
            path_resolution,
            goal_sample_rate,
            max_iter,
            connect_circle_dist,
            search_until_max_iter=True,
        )

        # -- informed-sampling state --
        self._best_cost: float = float("inf")
        self._best_path: Optional[np.ndarray] = None
        self._c_min: float = 0.0
        self._x_center: np.ndarray = np.zeros(2)
        self._C: np.ndarray = np.eye(2)

        # -- visualisation (persistent artists, updated via set_data) --
        self._tree_line = None          # single Line2D for all tree edges
        self._start_marker = None       # start marker (drawn once)
        self._goal_marker = None        # goal marker (drawn once)
        self._ellipse_fill = None       # Polygon for ellipse fill
        self._ellipse_line = None       # Line2D for ellipse outline
        self._best_path_line = None     # Line2D for best path
        self._vis_setup_done: bool = False
        self._cost_history: list[float] = []

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def planning(
        self,
        start_pose: list[float],
        goal_pose: list[float],
        show_animation: bool = True,
    ) -> Optional[tuple[list[float], list[float]]]:
        """
        Informed RRT* path planning.

        Args:
            start_pose: start pose ``[x, y]``.
            goal_pose:  goal  pose ``[x, y]``.
            show_animation: If *True*, render the tree, ellipse and best path
                during planning.

        Returns:
            2×N ``ndarray`` of (x, y) waypoints, or *None* if no path was found.
        """
        self.start = self.Node(start_pose[0].item(), start_pose[1].item())
        self.end = self.Node(goal_pose[0].item(), goal_pose[1].item())

        # -- pre-compute ellipse invariants ----------------------------
        self._c_min = math.hypot(
            self.end.x - self.start.x, self.end.y - self.start.y
        )
        self._x_center = np.array(
            [(self.start.x + self.end.x) / 2.0,
             (self.start.y + self.end.y) / 2.0]
        )
        theta = math.atan2(
            self.end.y - self.start.y, self.end.x - self.start.x
        )
        self._C = np.array(
            [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta),  math.cos(theta)]]
        )

        self._best_cost = float("inf")
        self._best_path = None
        self._cost_history = []

        # -- reset vis handles --
        self._tree_line = None
        self._start_marker = None
        self._goal_marker = None
        self._ellipse_fill = None
        self._ellipse_line = None
        self._best_path_line = None
        self._vis_setup_done = False

        # -- main loop -------------------------------------------------
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # -- sample ------------------------------------------------
            rnd = self._informed_sample()

            # -- extend tree (same as RRT*) ----------------------------
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(
                self.node_list[nearest_ind], rnd, self.expand_dis
            )
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + math.hypot(
                new_node.x - near_node.x, new_node.y - near_node.y
            )

            if not self.check_collision(new_node, self.robot_radius):
                continue

            near_inds = self.find_near_nodes(new_node)
            node_with_updated_parent = self.choose_parent(new_node, near_inds)
            if node_with_updated_parent:
                self.rewire(node_with_updated_parent, near_inds)
                self.node_list.append(node_with_updated_parent)
            else:
                self.node_list.append(new_node)

            # -- check for path improvement ----------------------------
            last_index = self.search_best_goal_node()
            if last_index is not None:
                path = self.generate_final_course(last_index)
                cost = self._path_cost(path)
                if cost < self._best_cost:
                    old_cost = self._best_cost
                    self._best_cost = cost
                    self._best_path = path
                    self._cost_history.append(cost)
                    if old_cost == float("inf"):
                        print(
                            f"[Informed RRT*] iter {i}: first path found, "
                            f"cost = {cost:.3f}  (c_min = {self._c_min:.3f})"
                        )
                    else:
                        print(
                            f"[Informed RRT*] iter {i}: path improved  "
                            f"{old_cost:.3f} -> {cost:.3f}  "
                            f"(Δ = {old_cost - cost:.3f})"
                        )

            # -- draw (throttled) --------------------------------------
            if show_animation and (i % 10 == 0 or i == self.max_iter - 1):
                self._draw_informed(i)

        # -- summary ---------------------------------------------------
        if self._best_path is not None:
            print(
                f"[Informed RRT*] done — final cost = {self._best_cost:.3f}, "
                f"improvements = {len(self._cost_history)}, "
                f"nodes = {len(self.node_list)}"
            )
            return self._best_path

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        print("[Informed RRT*] no feasible path found")
        return None

    # ------------------------------------------------------------------
    # Informed sampling
    # ------------------------------------------------------------------

    def _informed_sample(self) -> "RRTStar.Node":
        """Return a random sample — from the ellipse if a path exists,
        otherwise uniform with goal-bias."""
        if self._best_cost < float("inf"):
            return self._sample_from_ellipse()
        return self.get_random_node()

    def _sample_from_ellipse(self) -> "RRTStar.Node":
        """Uniform sample from the prolate hyper-spheroid (2-D ellipse)."""
        c_best = self._best_cost
        c_min = self._c_min

        # semi-axes
        a = c_best / 2.0
        b = math.sqrt(max(a * a - (c_min / 2.0) ** 2, 0.0))

        # uniform sample inside the unit disk
        ang = random.uniform(0, 2 * math.pi)
        r = math.sqrt(random.uniform(0, 1))
        x_ball = np.array([r * math.cos(ang), r * math.sin(ang)])

        # scale → rotate → translate
        sample = self._C @ np.array([a * x_ball[0], b * x_ball[1]]) + self._x_center

        # clamp to play area
        sample[0] = max(self.play_area.xmin, min(sample[0], self.play_area.xmax))
        sample[1] = max(self.play_area.ymin, min(sample[1], self.play_area.ymax))

        return self.Node(float(sample[0]), float(sample[1]))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _draw_informed(self, iteration: int) -> None:
        """Draw tree, ellipse and best path without clearing the axes.

        All visual elements use **persistent** matplotlib artists that are
        created once and updated in-place via ``set_data()`` /
        ``set_xy()``.  This avoids the overhead of removing and
        recreating objects every frame, and never calls ``plt.cla()``
        so the ir-sim grid map is preserved.
        """
        ax = plt.gca()

        # --- one-time setup (markers that never change) ---
        if not self._vis_setup_done:
            ax.figure.canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            (self._start_marker,) = ax.plot(
                self.start.x, self.start.y, "o",
                color="tab:green", markersize=8, zorder=5, label="start",
            )
            (self._goal_marker,) = ax.plot(
                self.end.x, self.end.y, "o",
                color="tab:red", markersize=8, zorder=5, label="goal",
            )
            self._vis_setup_done = True

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
            (self._tree_line,) = ax.plot(
                xs, ys, "-", color="tab:green", alpha=0.35, linewidth=0.6,
            )
        elif self._tree_line is not None:
            self._tree_line.set_data(xs, ys)

        # --- informed sampling ellipse (updated in-place) ---
        if self._best_cost < float("inf"):
            self._update_ellipse(ax)
        else:
            # hide ellipse if no path yet
            if self._ellipse_line is not None:
                self._ellipse_line.set_data([], [])
            if self._ellipse_fill is not None:
                self._ellipse_fill.set_xy(np.empty((0, 2)))

        # --- best path (updated in-place) ---
        if self._best_path is not None:
            if self._best_path_line is None:
                (self._best_path_line,) = ax.plot(
                    self._best_path[0], self._best_path[1],
                    "-", color="tab:red", linewidth=2.0, zorder=4,
                    label="best path",
                )
            else:
                self._best_path_line.set_data(
                    self._best_path[0], self._best_path[1],
                )

        # --- title & legend (updated in-place, no remove needed) ---
        title = f"Informed RRT*  —  iter {iteration}/{self.max_iter}"
        if self._best_cost < float("inf"):
            title += f"  |  cost = {self._best_cost:.2f}"
            title += f"  |  c_min = {self._c_min:.2f}"
        else:
            title += "  |  searching …"
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper left", fontsize=7)

        plt.pause(0.001)

    def _update_ellipse(self, ax: plt.Axes) -> None:
        """Create or update the informed sampling ellipse on *ax*."""
        c_best = self._best_cost
        c_min = self._c_min

        a = c_best / 2.0
        b = math.sqrt(max(a * a - (c_min / 2.0) ** 2, 0.0))

        # parametric ellipse
        t = np.linspace(0, 2 * np.pi, 80)
        x_ell = a * np.cos(t)
        y_ell = b * np.sin(t)
        pts = self._C @ np.vstack([x_ell, y_ell]) + self._x_center[:, None]
        xy = pts.T  # (N, 2)

        if self._ellipse_line is None:
            # first time — create both artists
            (self._ellipse_line,) = ax.plot(
                pts[0], pts[1], "--", color="tab:blue",
                linewidth=1.2, label="informed set",
            )
            self._ellipse_fill = MplPolygon(
                xy, closed=True, color="tab:blue", alpha=0.08,
            )
            ax.add_patch(self._ellipse_fill)
        else:
            # update existing artists in-place
            self._ellipse_line.set_data(pts[0], pts[1])
            self._ellipse_fill.set_xy(xy)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _path_cost(path: np.ndarray) -> float:
        """Total Euclidean length of a 2×N path array."""
        diffs = np.diff(path, axis=1)
        return float(np.sum(np.hypot(diffs[0], diffs[1])))

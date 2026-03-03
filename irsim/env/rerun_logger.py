"""Rerun sidecar visualisation logger for IR-SIM.

This module provides a read-only observer that logs simulation state to the
`Rerun Viewer <https://rerun.io>`_ each frame.  It reads **only** existing
public attributes on ``ObjectBase``, ``Lidar2D`` and ``World`` — no
modifications to the simulation core are needed.

Requires the optional ``rerun-sdk`` package::

    pip install 'ir-sim[rerun]'
"""

from __future__ import annotations

import contextlib
from math import cos, pi, sin
from typing import TYPE_CHECKING, Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

if TYPE_CHECKING:
    from irsim.world.object_base import ObjectBase
    from irsim.world.world import World


# ---------------------------------------------------------------------------
# Colour helpers  —  vibrant palette for dark backgrounds
# ---------------------------------------------------------------------------

_MPL_COLOR_MAP: dict[str, list[int]] = {
    "r": [231, 76, 60],
    "g": [46, 204, 113],
    "b": [52, 152, 219],
    "k": [60, 60, 70],
    "w": [236, 240, 241],
    "y": [241, 196, 15],
    "m": [155, 89, 182],
    "c": [26, 188, 156],
    "orange": [230, 126, 34],
    "gray": [149, 165, 166],
    "grey": [149, 165, 166],
    "gold": [255, 215, 0],
    "red": [231, 76, 60],
    "green": [46, 204, 113],
    "blue": [52, 152, 219],
    "black": [60, 60, 70],
    "white": [236, 240, 241],
    "yellow": [241, 196, 15],
    "magenta": [155, 89, 182],
    "cyan": [26, 188, 156],
    "purple": [142, 68, 173],
    "brown": [160, 64, 0],
    "pink": [255, 150, 180],
    "lime": [0, 230, 64],
    "navy": [44, 62, 80],
}


def _color_to_rgba(color: str | list | tuple, alpha: float = 1.0) -> list[int]:
    """Convert a matplotlib-style colour string to an RGBA list."""
    if isinstance(color, list | tuple):
        if len(color) == 3:
            return [int(c) for c in color] + [int(alpha * 255)]
        return [int(c) for c in color[:4]]
    rgb = _MPL_COLOR_MAP.get(color, [149, 165, 166])
    return [*rgb, int(alpha * 255)]


# ---------------------------------------------------------------------------
# Draw-order layers  (higher = on top)
# ---------------------------------------------------------------------------
_Z_GRID = 0.0
_Z_BOUNDARY = 1.0
_Z_OBS_FILL = 10.0
_Z_OBS_OUTLINE = 11.0
_Z_TRAJECTORY = 20.0
_Z_LIDAR_BEAM = 25.0
_Z_LIDAR_HIT = 26.0
_Z_ROBOT_FILL = 40.0
_Z_ROBOT_OUTLINE = 41.0
_Z_HEADING = 42.0
_Z_GOAL = 50.0
_Z_LABEL = 60.0


# ---------------------------------------------------------------------------
# RerunLogger
# ---------------------------------------------------------------------------


class RerunLogger:
    """Log IR-SIM simulation state to Rerun each frame.

    All data is obtained by reading existing public properties on the
    simulation objects — nothing is modified.

    Args:
        world: The ``World`` instance (provides bounds, time, grid_map).
        objects: Initial list of ``ObjectBase`` instances.
    """

    def __init__(self, world: World, objects: list[ObjectBase]) -> None:
        # Ensure the Rerun Viewer binary shipped with rerun-sdk is findable.
        import os
        import sys

        venv_bin = os.path.dirname(sys.executable)
        if venv_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

        rr.init("ir-sim", spawn=True)

        # ---- Blueprint with dark background ----
        x0, x1 = world.x_range
        y0, y1 = world.y_range
        span = max(x1 - x0, y1 - y0)
        pad = span * 0.06

        blueprint = rrb.Blueprint(
            rrb.Spatial2DView(
                name="IR-SIM",
                origin="/world",
                visual_bounds=rrb.VisualBounds2D(
                    x_range=[x0 - pad, x1 + pad],
                    y_range=[y0 - pad, y1 + pad],
                ),
                background=[200, 200, 200],
            ),
        )
        rr.send_blueprint(blueprint)

        # Static elements
        if world.grid_map is not None:
            self._log_grid_map(world)
        self._log_world_boundary(world)
        self._log_static_objects(objects)

        # Log initial state for ALL objects (including robots) so the
        # viewer shows something even when step() is never called
        # (e.g. path-planning scripts that plan then exit).
        self._log_initial_objects(objects)

    # ------------------------------------------------------------------
    # Static / one-time logging
    # ------------------------------------------------------------------

    def _log_world_boundary(self, world: World) -> None:
        x0, x1 = world.x_range
        y0, y1 = world.y_range
        rr.log(
            "world/boundary",
            rr.LineStrips2D(
                [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
                colors=[[100, 100, 130, 100]],
                radii=[0.015],
                draw_order=_Z_BOUNDARY,
            ),
            static=True,
        )

    def _log_grid_map(self, world: World) -> None:
        """Render the occupancy grid as filled cells using Points2D.

        Occupied cells (value > 0.5) are drawn as dark squares, matching
        matplotlib's ``cmap='Greys'`` convention (0 = white/free, 1 = black/
        obstacle).  Each occupied cell becomes a point with radius = half the
        cell size so adjacent cells visually merge into solid regions.
        """
        grid = np.asarray(world.grid_map, dtype=np.float32)  # (nx, ny)
        x0, x1 = world.x_range
        y0, y1 = world.y_range
        nx, ny = grid.shape
        x_reso = (x1 - x0) / nx
        y_reso = (y1 - y0) / ny

        # Indices of occupied cells
        occ_i, occ_j = np.where(grid > 0.5)
        if len(occ_i) == 0:
            return

        # Cell centres in world coordinates
        cx = x0 + (occ_i + 0.5) * x_reso
        cy = y0 + (occ_j + 0.5) * y_reso
        pts = np.column_stack([cx, cy])

        # Draw as filled squares (radius ≈ half-cell diagonal so they tile)
        cell_radius = max(x_reso, y_reso) * 0.55
        rr.log(
            "world/grid_map",
            rr.Points2D(
                pts,
                radii=[cell_radius],
                colors=[[40, 40, 48, 255]],
                draw_order=_Z_GRID,
            ),
            static=True,
        )

    def _log_static_objects(self, objects: list[ObjectBase]) -> None:
        for obj in objects:
            if not obj.static:
                continue
            self._log_object_shape(obj, static=True)

    def _log_initial_objects(self, objects: list[ObjectBase]) -> None:
        """Log the initial state of every non-static object.

        This ensures the viewer shows robots, goals, etc. even when the
        caller never invokes :meth:`step` (e.g. path-planning demos).
        """
        for obj in objects:
            if obj.static:
                continue
            self._log_object_shape(obj, static=True)
            self._log_object_goal(obj)
            self._log_object_heading(obj)

    # ------------------------------------------------------------------
    # Draw API  (called from EnvBase.draw_trajectory / draw_points)
    # ------------------------------------------------------------------

    _traj_counter: int = 0

    def draw_trajectory(
        self,
        traj: Any,
        traj_type: str = "g-",
        label: str = "trajectory",
    ) -> None:
        """Log an externally-computed trajectory (e.g. from a planner)."""
        arr = np.asarray(traj, dtype=float)
        # Accept (N,2), (2,N), or list-of-2x1
        if arr.ndim == 2:
            if arr.shape[1] == 2:
                pts = arr.tolist()
            elif arr.shape[0] == 2:
                pts = arr.T.tolist()
            else:
                pts = arr[:, :2].tolist()
        else:
            return
        if len(pts) < 2:
            return

        # Parse colour from matplotlib format string
        color_char = traj_type.replace("-", "").replace("--", "").replace(".", "")
        color_char = color_char.strip() or "g"
        rgba = _color_to_rgba(color_char, alpha=0.9)

        RerunLogger._traj_counter += 1
        rr.log(
            f"world/planned_path/{label}_{RerunLogger._traj_counter}",
            rr.LineStrips2D(
                [pts],
                colors=[rgba],
                radii=[0.04],
                draw_order=_Z_TRAJECTORY,
            ),
            static=True,
        )

    def draw_points(
        self,
        points: Any,
        s: int = 30,
        c: str = "b",
    ) -> None:
        """Log externally-provided scatter points."""
        arr = np.asarray(points, dtype=float)
        if arr.ndim == 2:
            if arr.shape[0] == 2 and arr.shape[1] != 2:
                arr = arr.T
            pts = arr[:, :2]
        else:
            return
        rgba = _color_to_rgba(c, alpha=0.9)
        radius = max(0.02, s / 800.0)
        rr.log(
            "world/user_points",
            rr.Points2D(pts, radii=[radius], colors=[rgba], draw_order=_Z_LABEL),
        )

    # ------------------------------------------------------------------
    # Per-frame
    # ------------------------------------------------------------------

    def step(self, objects: list[ObjectBase], world: Any) -> None:
        rr.set_time("step", sequence=world.count)
        rr.set_time("sim_time", duration=world.time)

        for obj in objects:
            if obj.static:
                continue
            self._log_object_shape(obj)
            self._log_object_goal(obj)
            self._log_object_trajectory(obj)
            self._log_object_heading(obj)
            self._log_object_sensors(obj)

    # ------------------------------------------------------------------
    # Shape  (fill via Points2D + outline via LineStrips2D)
    # ------------------------------------------------------------------

    def _log_object_shape(self, obj: ObjectBase, static: bool = False) -> None:
        entity = f"world/{obj.role}/{obj.name}"
        is_robot = obj.role == "robot"

        obj_rgba = _color_to_rgba(obj.color)
        # Fill colour: semi-transparent
        fill_c = list(obj_rgba)
        fill_c[3] = 180 if is_robot else 100
        # Outline colour: fully opaque, slightly brighter
        out_c = list(obj_rgba)
        out_c[3] = 255

        z_fill = _Z_ROBOT_FILL if is_robot else _Z_OBS_FILL
        z_out = _Z_ROBOT_OUTLINE if is_robot else _Z_OBS_OUTLINE
        out_r = 0.035 if is_robot else 0.025

        shape = obj.shape

        if shape == "circle":
            cx, cy = float(obj.state[0, 0]), float(obj.state[1, 0])
            r = float(obj.radius)

            # Filled disc
            rr.log(
                f"{entity}/fill",
                rr.Points2D(
                    [[cx, cy]],
                    radii=[r],
                    colors=[fill_c],
                    draw_order=z_fill,
                ),
                static=static,
            )
            # Crisp outline ring
            n = 48
            angles = np.linspace(0, 2 * pi, n + 1)
            pts = [[cx + r * cos(a), cy + r * sin(a)] for a in angles]
            rr.log(
                f"{entity}/outline",
                rr.LineStrips2D(
                    [pts],
                    colors=[out_c],
                    radii=[out_r],
                    labels=[obj.name],
                    show_labels=True,
                    draw_order=z_out,
                ),
                static=static,
            )

        elif shape in ("rectangle", "polygon"):
            verts = obj.vertices  # (2, N)
            pts = verts.T.tolist()
            pts_closed = [*pts, pts[0]]
            cx = float(np.mean(verts[0]))
            cy = float(np.mean(verts[1]))

            # Filled approximation — a point at centroid with large radius
            # won't match a rotated rect, so just use thick outline
            rr.log(
                f"{entity}/outline",
                rr.LineStrips2D(
                    [pts_closed],
                    colors=[out_c],
                    radii=[out_r],
                    labels=[obj.name],
                    show_labels=True,
                    draw_order=z_out,
                ),
                static=static,
            )

        elif shape in ("line", "linestring"):
            verts = obj.vertices
            pts = verts.T.tolist()
            rr.log(
                f"{entity}/outline",
                rr.LineStrips2D(
                    [pts],
                    colors=[out_c],
                    radii=[out_r * 1.2],
                    labels=[obj.name],
                    show_labels=False,
                    draw_order=z_out,
                ),
                static=static,
            )
        else:
            try:
                verts = obj.vertices
                if verts is not None and verts.size > 0:
                    pts = verts.T.tolist()
                    pts.append(pts[0])
                    rr.log(
                        f"{entity}/outline",
                        rr.LineStrips2D(
                            [pts],
                            colors=[out_c],
                            radii=[out_r],
                            draw_order=z_out,
                        ),
                        static=static,
                    )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Heading arrow (robots only)
    # ------------------------------------------------------------------

    def _log_object_heading(self, obj: ObjectBase) -> None:
        if obj.role != "robot":
            return
        cx, cy = float(obj.state[0, 0]), float(obj.state[1, 0])
        theta = float(obj.state[2, 0]) if obj.state.shape[0] > 2 else 0.0
        r = float(obj.radius) if obj.shape == "circle" else 0.3
        arrow_len = r * 1.4
        rr.log(
            f"world/{obj.role}/{obj.name}/heading",
            rr.Arrows2D(
                origins=[[cx, cy]],
                vectors=[[arrow_len * cos(theta), arrow_len * sin(theta)]],
                colors=[[255, 255, 255, 220]],
                radii=[0.04],
                draw_order=_Z_HEADING,
            ),
        )

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    def _log_object_goal(self, obj: ObjectBase) -> None:
        if obj.role != "robot":
            return
        goal = obj.goal
        if goal is None:
            return
        gx, gy = float(goal[0, 0]), float(goal[1, 0])
        # Outer ring + inner dot
        rr.log(
            f"world/{obj.role}/{obj.name}/goal/dot",
            rr.Points2D(
                [[gx, gy]],
                radii=[0.12],
                colors=[[255, 215, 0, 200]],
                labels=["goal"],
                show_labels=True,
                draw_order=_Z_GOAL,
            ),
        )
        n = 24
        angles = np.linspace(0, 2 * pi, n + 1)
        ring_r = 0.2
        ring_pts = [[gx + ring_r * cos(a), gy + ring_r * sin(a)] for a in angles]
        rr.log(
            f"world/{obj.role}/{obj.name}/goal/ring",
            rr.LineStrips2D(
                [ring_pts],
                colors=[[255, 215, 0, 150]],
                radii=[0.02],
                draw_order=_Z_GOAL,
            ),
        )

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def _log_object_trajectory(self, obj: ObjectBase) -> None:
        traj = obj.trajectory
        if not traj or len(traj) < 2:
            return
        pts = [[float(s[0, 0]), float(s[1, 0])] for s in traj]
        color = _color_to_rgba(obj.color, alpha=0.7)
        rr.log(
            f"world/{obj.role}/{obj.name}/trajectory",
            rr.LineStrips2D(
                [pts],
                colors=[color],
                radii=[0.018],
                draw_order=_Z_TRAJECTORY,
            ),
        )

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------

    def _log_object_sensors(self, obj: ObjectBase) -> None:
        if not hasattr(obj, "sensors"):
            return
        for sensor in obj.sensors:
            if getattr(sensor, "sensor_type", None) != "lidar2d":
                continue
            self._log_lidar(obj, sensor)

    def _log_lidar(self, obj: ObjectBase, lidar: Any) -> None:
        origin = lidar.lidar_origin  # (3,1)
        ox = float(origin[0, 0])
        oy = float(origin[1, 0])
        theta = float(origin[2, 0]) if origin.shape[0] > 2 else 0.0

        range_data = np.asarray(lidar.range_data).ravel()
        angle_list = np.asarray(lidar.angle_list).ravel()

        # Vectorised end-point calculation
        angles = angle_list + theta
        ex = ox + range_data * np.cos(angles)
        ey = oy + range_data * np.sin(angles)

        # --- Hit points only (where beam hit something) ---
        hit_mask = range_data < (lidar.range_max - 0.05)
        hit_x = ex[hit_mask]
        hit_y = ey[hit_mask]

        entity_base = f"world/{obj.role}/{obj.name}/lidar"

        # Beams: only draw a sparse subset to reduce clutter
        step = max(1, lidar.number // 30)  # ~30 beams shown
        idx = np.arange(0, lidar.number, step)
        sparse_segments = [[[ox, oy], [float(ex[i]), float(ey[i])]] for i in idx]
        rr.log(
            f"{entity_base}/beams",
            rr.LineStrips2D(
                sparse_segments,
                colors=[[0, 200, 230, 40]],
                radii=[0.005],
                draw_order=_Z_LIDAR_BEAM,
            ),
        )

        # Hit points — bright, clearly visible
        if len(hit_x) > 0:
            hit_pts = np.column_stack([hit_x, hit_y])
            rr.log(
                f"{entity_base}/hits",
                rr.Points2D(
                    hit_pts,
                    radii=[0.035],
                    colors=[[231, 76, 60, 240]],
                    draw_order=_Z_LIDAR_HIT,
                ),
            )

    # ------------------------------------------------------------------
    # Planner visualisation callback
    # ------------------------------------------------------------------

    def make_planner_callback(self) -> Any:
        """Return a ``callback(planner, iteration)`` for live planning viz.

        The returned callable logs the RRT/RRT*/Informed-RRT* exploration
        tree, the informed sampling ellipse, and the current best path
        to the Rerun viewer.
        """

        def _callback(planner: Any, iteration: int) -> None:
            rr.set_time("plan_iter", sequence=iteration)

            # --- tree edges ---
            segments: list[list[list[float]]] = []
            for node in planner.node_list:
                if node.parent is not None and node.path_x:
                    seg = [
                        [float(x), float(y)]
                        for x, y in zip(node.path_x, node.path_y, strict=False)
                    ]
                    if len(seg) >= 2:
                        segments.append(seg)
            if segments:
                rr.log(
                    "world/planner/tree",
                    rr.LineStrips2D(
                        segments,
                        colors=[[46, 204, 113, 60]],
                        radii=[0.012],
                        draw_order=_Z_TRAJECTORY - 2,
                    ),
                )

            # --- informed sampling ellipse ---
            best_cost = getattr(planner, "_best_cost", float("inf"))
            c_min = getattr(planner, "_c_min", 0.0)
            x_center = getattr(planner, "_x_center", None)
            rot = getattr(planner, "_rotation_matrix", None)
            if (
                best_cost < float("inf")
                and c_min > 0
                and x_center is not None
                and rot is not None
            ):
                import math as _m

                a = best_cost / 2.0
                b = _m.sqrt(max(a * a - (c_min / 2.0) ** 2, 0.0))
                t = np.linspace(0, 2 * pi, 80)
                pts = (
                    rot @ np.vstack([a * np.cos(t), b * np.sin(t)]) + x_center[:, None]
                )
                ell_pts = pts.T.tolist()
                ell_pts.append(ell_pts[0])  # close
                rr.log(
                    "world/planner/ellipse",
                    rr.LineStrips2D(
                        [ell_pts],
                        colors=[[52, 152, 219, 120]],
                        radii=[0.02],
                        draw_order=_Z_TRAJECTORY - 1,
                    ),
                )
            else:
                # Clear ellipse when not yet found
                rr.log(
                    "world/planner/ellipse",
                    rr.LineStrips2D([], draw_order=_Z_TRAJECTORY - 1),
                )

            # --- best path so far ---
            best_path = getattr(planner, "_best_path", None)
            if best_path is not None:
                bp = np.asarray(best_path)
                path_pts = bp.T.tolist() if bp.shape[0] == 2 else bp[:, :2].tolist()
                if len(path_pts) >= 2:
                    rr.log(
                        "world/planner/best_path",
                        rr.LineStrips2D(
                            [path_pts],
                            colors=[[231, 76, 60, 220]],
                            radii=[0.04],
                            draw_order=_Z_TRAJECTORY + 1,
                        ),
                    )

        return _callback

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        with contextlib.suppress(Exception):
            rr.disconnect()

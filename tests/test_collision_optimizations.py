"""Tests for Phase 1 collision detection optimizations.

Covers:
  T1 — vectorised grid collision (``_grid_utils.grid_collision``)
  T2 — ``_grid_collision_geometry`` / ``check_grid_collision`` consistency
  T3 — planner ``check_node`` / ``is_collision`` with grid_occupied shortcut
  T4 — ``Map._obstacle_tree`` STRtree for obstacle_list
  T5 — centralised collision deduplication (``_check_all_collisions``)
  T6 — performance benchmarks (``@pytest.mark.benchmark``)
"""

from __future__ import annotations

from unittest.mock import patch as mock_patch

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, box

from irsim.world.map._grid_utils import (
    CELL_CENTER_OFFSET,
    COLLISION_RADIUS_FACTOR,
    OCCUPANCY_THRESHOLD,
    grid_collision,
)
from irsim.world.map.obstacle_map import ObstacleMap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obstacle_map(
    grid_map: np.ndarray,
    grid_reso: tuple[float, float] = (1.0, 1.0),
    world_offset: tuple[float, float] = (0.0, 0.0),
) -> ObstacleMap:
    """Create an ObstacleMap with given grid data."""
    nx, ny = grid_map.shape
    rx, ry = grid_reso
    ox, oy = world_offset
    w, h = nx * rx, ny * ry
    points = np.array([[ox, ox + w, ox + w, ox], [oy, oy, oy + h, oy + h]])
    shape = {"name": "map", "reso": 0.1, "points": points}
    return ObstacleMap(
        shape=shape,
        grid_map=grid_map,
        grid_reso=np.array([[rx], [ry]]),
        world_offset=[ox, oy],
    )


def _naive_grid_collision(
    grid: np.ndarray,
    grid_reso: tuple[float, float],
    geometry,
    world_offset: tuple[float, float] = (0.0, 0.0),
) -> bool:
    """Reference (naive) implementation — original per-cell Python loop."""
    x_reso, y_reso = grid_reso
    offset_x, offset_y = world_offset
    minx, miny, maxx, maxy = geometry.bounds
    i_min = max(0, int((minx - offset_x) / x_reso))
    i_max = min(grid.shape[0] - 1, int((maxx - offset_x) / x_reso))
    j_min = max(0, int((miny - offset_y) / y_reso))
    j_max = min(grid.shape[1] - 1, int((maxy - offset_y) / y_reso))
    if i_min > i_max or j_min > j_max:
        return False
    collision_radius = max(x_reso, y_reso) * COLLISION_RADIUS_FACTOR
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            if grid[i, j] > OCCUPANCY_THRESHOLD:
                cell_x = offset_x + (i + CELL_CENTER_OFFSET) * x_reso
                cell_y = offset_y + (j + CELL_CENTER_OFFSET) * y_reso
                cell_center = Point(cell_x, cell_y)
                if geometry.distance(cell_center) <= collision_radius:
                    return True
    return False


# ===================================================================
# T1 — Vectorised grid_collision correctness
# ===================================================================


class TestGridCollisionVectorized:
    """Verify grid_collision matches the naive reference for all shapes."""

    def test_circle_hit(self):
        grid = np.zeros((20, 20))
        grid[10, 10] = 100
        geom = Point(10.5, 10.5).buffer(0.3)
        assert grid_collision(grid, (1.0, 1.0), geom) is True

    def test_circle_miss(self):
        grid = np.zeros((20, 20))
        grid[10, 10] = 100
        geom = Point(0.5, 0.5).buffer(0.3)
        assert grid_collision(grid, (1.0, 1.0), geom) is False

    def test_rectangle_hit(self):
        grid = np.zeros((20, 20))
        grid[5, 5] = 100
        geom = box(5, 5, 6, 6)
        assert grid_collision(grid, (1.0, 1.0), geom) is True

    def test_rectangle_boundary(self):
        """Rectangle exactly touching the collision radius edge."""
        grid = np.zeros((20, 20))
        grid[5, 5] = 100
        # Cell centre at (5.5, 5.5), collision_radius = 0.5
        # Box placed just within radius
        geom = box(5.2, 5.2, 5.8, 5.8)
        assert grid_collision(grid, (1.0, 1.0), geom) is True

    def test_empty_grid(self):
        grid = np.zeros((20, 20))
        geom = box(5, 5, 15, 15)
        assert grid_collision(grid, (1.0, 1.0), geom) is False

    def test_full_grid(self):
        grid = np.full((20, 20), 100.0)
        geom = box(5, 5, 6, 6)
        assert grid_collision(grid, (1.0, 1.0), geom) is True

    def test_out_of_bounds_geometry(self):
        grid = np.full((10, 10), 100.0)
        geom = box(20, 20, 21, 21)
        assert grid_collision(grid, (1.0, 1.0), geom) is False

    def test_threshold_boundary_equal(self):
        """Value exactly at threshold → not occupied (> not >=)."""
        grid = np.full((10, 10), float(OCCUPANCY_THRESHOLD))
        geom = box(5, 5, 6, 6)
        assert grid_collision(grid, (1.0, 1.0), geom) is False

    def test_threshold_boundary_above(self):
        grid = np.full((10, 10), OCCUPANCY_THRESHOLD + 1.0)
        geom = box(5, 5, 6, 6)
        assert grid_collision(grid, (1.0, 1.0), geom) is True

    def test_with_offset(self):
        grid = np.zeros((10, 10))
        grid[5, 5] = 100
        # Offset (10, 10) → cell centre at (15.5, 15.5)
        geom = box(15, 15, 16, 16)
        assert grid_collision(grid, (1.0, 1.0), geom, world_offset=(10.0, 10.0)) is True

    def test_negative_offset(self):
        grid = np.zeros((10, 10))
        grid[5, 5] = 100
        # Offset (-10, -10) → cell centre at (-4.5, -4.5)
        geom = box(-5, -5, -4, -4)
        assert (
            grid_collision(grid, (1.0, 1.0), geom, world_offset=(-10.0, -10.0)) is True
        )

    def test_non_square_resolution(self):
        grid = np.zeros((10, 20))
        grid[5, 10] = 100
        # x_reso=2, y_reso=1 → cell at (11, 10.5), collision_radius = 1.0
        geom = box(10.5, 10, 11.5, 11)
        assert grid_collision(grid, (2.0, 1.0), geom) is True


class TestGridCollisionLargeMap:
    """Vectorised implementation matches naive on a large random grid."""

    def test_consistency_with_naive(self):
        rng = np.random.default_rng(42)
        grid = rng.uniform(0, 100, size=(200, 200))
        reso = (0.5, 0.5)
        mismatches = 0
        n_queries = 200
        for _ in range(n_queries):
            x = rng.uniform(0, 100)
            y = rng.uniform(0, 100)
            geom = box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
            vec = grid_collision(grid, reso, geom)
            naive = _naive_grid_collision(grid, reso, geom)
            if vec != naive:
                mismatches += 1
        assert mismatches == 0, f"{mismatches}/{n_queries} mismatches"


# ===================================================================
# T2 — check_grid_collision / _grid_collision_geometry consistency
# ===================================================================


class TestGridUtilsConsistency:
    """ObstacleMap.check_grid_collision and Map._grid_collision_geometry
    must produce identical results since both delegate to grid_collision."""

    def test_obstacle_map_delegates(self):
        from irsim.world.map import _grid_collision_geometry

        grid = np.zeros((20, 20))
        grid[10, 10] = 100
        reso = (1.0, 1.0)
        offset = (0.0, 0.0)
        omap = _make_obstacle_map(grid)

        geom_hit = box(10, 10, 11, 11)
        geom_miss = box(0, 0, 1, 1)

        assert omap.check_grid_collision(geom_hit) is True
        assert (
            _grid_collision_geometry(grid, reso, geom_hit, world_offset=offset) is True
        )

        assert omap.check_grid_collision(geom_miss) is False
        assert (
            _grid_collision_geometry(grid, reso, geom_miss, world_offset=offset)
            is False
        )


# ===================================================================
# T3 — Planner check_node / is_collision with grid_occupied shortcut
# ===================================================================


class TestPlannerGridShortcut:
    """Planners should use grid_occupied for O(1) checks when grid exists."""

    def test_astar_grid_only(self, env_factory):
        """A* produces a valid path on a grid-only map."""
        from irsim.lib.path_planners.a_star import AStarPlanner

        env = env_factory("test_grid_map.yaml")
        planner_map = env.get_map(resolution=0.5)
        planner = AStarPlanner(planner_map)
        path = planner.planning(
            start_pose=np.array([[2], [2], [0]]),
            goal_pose=np.array([[45], [45], [0]]),
            show_animation=False,
        )
        assert path is not None
        assert path.shape[0] == 2  # (2, N)
        assert path.shape[1] > 2  # non-trivial path

    def test_jps_grid_only(self, env_factory):
        """JPS produces a valid path on a grid-only map."""
        from irsim.lib.path_planners.jps import JPSPlanner

        env = env_factory("test_grid_map.yaml")
        planner_map = env.get_map(resolution=0.5)
        planner = JPSPlanner(planner_map)
        path = planner.planning(
            start_pose=np.array([[2], [2], [0]]),
            goal_pose=np.array([[45], [45], [0]]),
            show_animation=False,
        )
        assert path is not None
        assert path.shape[0] == 2

    def test_planner_no_grid_fallback(self):
        """When grid is None, planner falls back to Shapely geometry path."""
        from irsim.world.map import Map

        m = Map(width=10, height=10, resolution=1.0, grid=None, obstacle_list=[])
        result = m.grid_occupied(5, 5)
        # grid_occupied returns None when no grid → planners fall back
        assert result is None


# ===================================================================
# T4 — Map._obstacle_tree STRtree
# ===================================================================


class TestMapObstacleTree:
    """Map should use STRtree for obstacle_list queries."""

    def test_obstacle_tree_built(self):
        from irsim.world.map import Map

        class FakeObs:
            def __init__(self, geom):
                self._geometry = geom

        obs = [FakeObs(box(i, i, i + 1, i + 1)) for i in range(50)]
        m = Map(width=100, height=100, resolution=1.0, obstacle_list=obs)
        assert m._obstacle_tree is not None

    def test_obstacle_tree_empty(self):
        from irsim.world.map import Map

        m = Map(width=10, height=10, resolution=1.0, obstacle_list=[])
        assert m._obstacle_tree is None

    def test_obstacle_tree_collision_detect(self):
        from irsim.world.map import Map

        class FakeObs:
            def __init__(self, geom):
                self._geometry = geom

        obs = [FakeObs(box(5, 5, 6, 6))]
        m = Map(width=10, height=10, resolution=1.0, obstacle_list=obs)
        # Overlapping geometry → collision
        assert m.is_collision(box(5.5, 5.5, 6.5, 6.5)) is True
        # Non-overlapping → no collision
        assert m.is_collision(box(0, 0, 1, 1)) is False


# ===================================================================
# T5 — Centralised collision deduplication
# ===================================================================


class TestCollisionSymmetry:
    """_check_all_collisions should detect collisions symmetrically
    with at most N*(N-1)/2 narrowphase calls for non-map pairs."""

    def test_symmetric_collision_flags(self, env_factory):
        """When A overlaps B, both A.collision_obj and B.collision_obj are set."""
        env = env_factory("test_collision_world.yaml")
        # Run one step to trigger collision detection
        env.step()

        # Check symmetry: if A collides with B, B must collide with A
        for obj in env.objects:
            for coll in obj.collision_obj:
                if coll.shape == "map":
                    continue  # map collisions are single-direction
                assert obj in coll.collision_obj, (
                    f"{obj.name} has {coll.name} in collision_obj, "
                    f"but {coll.name} does not have {obj.name}"
                )

    def test_narrowphase_call_count(self, env_factory):
        """Non-map pair checks should be at most N*(N-1)/2."""
        env = env_factory("test_collision_world.yaml")

        call_count = 0
        original_intersects = shapely.intersects

        def counting_intersects(a, b):
            nonlocal call_count
            call_count += 1
            return original_intersects(a, b)

        with mock_patch("irsim.env.env_base.shapely.intersects", counting_intersects):
            env._check_all_collisions()

        non_map_non_unobstructed = [
            obj for obj in env.objects if not obj.unobstructed and obj.shape != "map"
        ]
        n = len(non_map_non_unobstructed)
        max_pairs = n * (n - 1) // 2
        assert call_count <= max_pairs, (
            f"narrowphase called {call_count} times, "
            f"but max for {n} objects is {max_pairs}"
        )

    def test_map_collision_not_deduplicated(self, env_factory):
        """Map objects should still be checked against each non-map object."""
        env = env_factory("test_grid_map.yaml")
        env.step()
        # Map collisions should work — no crash, no missing checks
        maps = [obj for obj in env.objects if obj.shape == "map"]
        assert len(maps) >= 1

    def test_collision_mode_stop(self, env_factory):
        """collision_mode=stop should still set stop_flag correctly."""
        env = env_factory("test_collision_world.yaml")
        # Run several steps to trigger collisions
        for _ in range(5):
            env.step()
        # Verify stop_flag is set on colliding robots
        for obj in env.objects:
            if obj.role == "robot" and obj.collision_flag:
                # In "stop" mode, if any collision_obj is not unobstructed,
                # stop_flag should be True
                has_blocking = any(not o.unobstructed for o in obj.collision_obj)
                assert obj.stop_flag == has_blocking


# ===================================================================
# T6 — Performance benchmarks (not CI-blocking)
# ===================================================================


@pytest.mark.benchmark
class TestBenchmarks:
    """Performance benchmarks — run with ``pytest -m benchmark``."""

    def test_bench_grid_collision_vectorized(self):
        """10 000 random queries on a 1000x1000 grid under 5 seconds."""
        import time

        rng = np.random.default_rng(0)
        grid = np.zeros((1000, 1000))
        # Scatter ~10% occupied cells
        occupied = rng.choice(1_000_000, size=100_000, replace=False)
        grid.flat[occupied] = 100
        reso = (0.05, 0.05)

        start = time.perf_counter()
        for _ in range(10_000):
            x = rng.uniform(0, 50)
            y = rng.uniform(0, 50)
            geom = box(x - 0.25, y - 0.25, x + 0.25, y + 0.25)
            grid_collision(grid, reso, geom)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"10000 queries took {elapsed:.2f}s (limit 5s)"

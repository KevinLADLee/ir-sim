"""
Maze-based 2D grid map generator for robot navigation.

Generates orthogonal, Micromouse-style occupancy grids using a perfect maze
(randomized depth-first search / recursive backtracker). The default layout is
an escape task: a free room at the center and an exit carved at the outer
boundary. World size and obstacle_map.resolution determine the final grid
shape; maze parameters only control the logical topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid_map_generator_base import GridMapGenerator

_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_OPPOSITE = (2, 3, 0, 1)
_ENTRANCE_CELLS = {
    "bottom_left": (0, 0),
    "bottom_right": (-1, 0),
    "top_left": (0, -1),
    "top_right": (-1, -1),
}
_ENTRANCE_SIDES = {
    "bottom_left": "south",
    "bottom_right": "south",
    "top_left": "north",
    "top_right": "north",
}


@dataclass(frozen=True)
class _GoalRegion:
    x0: int
    y0: int
    x1: int
    y1: int


def _validate_positive_int(name: str, value: int) -> int:
    """Validate that a parameter is a positive integer."""
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer (got {value!r})")
    return int(value)


def _resolve_corner_cell(cols: int, rows: int, entrance: str) -> tuple[int, int]:
    """Resolve an entrance name to a logical cell coordinate."""
    if entrance not in _ENTRANCE_CELLS:
        known = ", ".join(sorted(_ENTRANCE_CELLS))
        raise ValueError(f"entrance must be one of {known} (got {entrance!r})")
    raw_x, raw_y = _ENTRANCE_CELLS[entrance]
    x = cols - 1 if raw_x == -1 else raw_x
    y = rows - 1 if raw_y == -1 else raw_y
    return x, y


def _resolve_goal_region(
    cols: int,
    rows: int,
    goal_area: str,
    goal_cols: int,
    goal_rows: int,
) -> _GoalRegion:
    """Resolve the logical goal room region."""
    if goal_area != "center":
        raise ValueError(f"goal_area must be 'center' (got {goal_area!r})")
    if goal_cols > cols or goal_rows > rows:
        raise ValueError(
            "goal_rows/goal_cols must not exceed maze dimensions "
            f"(got goal={goal_cols}x{goal_rows}, maze={cols}x{rows})"
        )

    x0 = (cols - goal_cols) // 2
    y0 = (rows - goal_rows) // 2
    return _GoalRegion(x0=x0, y0=y0, x1=x0 + goal_cols, y1=y0 + goal_rows)


def _partition_axis(size: int, bands: int) -> list[tuple[int, int]]:
    """Split an axis into positive-width contiguous bands that cover it exactly."""
    size = _validate_positive_int("size", size)
    bands = _validate_positive_int("bands", bands)
    if size < bands:
        raise ValueError(
            f"grid dimension {size} is too small for {bands} maze bands; "
            "increase world size or decrease maze rows/cols"
        )

    base = size // bands
    remainder = size % bands
    out: list[tuple[int, int]] = []
    start = 0
    for idx in range(bands):
        width = base + (1 if idx < remainder else 0)
        end = start + width
        out.append((start, end))
        start = end
    return out


def generate_maze_adjacency(
    cols: int,
    rows: int,
    seed: int | None = None,
    start_cell: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Generate a perfect maze adjacency grid.

    Returns:
        Array of shape ``(cols, rows, 4)`` where the last axis encodes open
        passages in the order east, north, west, south.
    """
    cols = _validate_positive_int("cols", cols)
    rows = _validate_positive_int("rows", rows)
    sx, sy = start_cell
    if not (0 <= sx < cols and 0 <= sy < rows):
        raise ValueError(
            f"start_cell must be inside maze bounds (got {start_cell!r} for {cols}x{rows})"
        )

    rng = np.random.default_rng(seed)
    adjacency = np.zeros((cols, rows, 4), dtype=bool)
    visited = np.zeros((cols, rows), dtype=bool)
    stack: list[tuple[int, int]] = [(sx, sy)]
    visited[sx, sy] = True

    while stack:
        x, y = stack[-1]
        candidates: list[tuple[int, int, int]] = []
        for direction, (dx, dy) in enumerate(_DIRS):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < cols and 0 <= ny < rows and not visited[nx, ny]:
                candidates.append((direction, nx, ny))
        if not candidates:
            stack.pop()
            continue

        direction, nx, ny = candidates[int(rng.integers(len(candidates)))]
        adjacency[x, y, direction] = True
        adjacency[nx, ny, _OPPOSITE[direction]] = True
        visited[nx, ny] = True
        stack.append((nx, ny))

    return adjacency


def rasterize_maze(
    adjacency: np.ndarray,
    width: int,
    height: int,
    entrance: str = "bottom_left",
    exit: str | None = None,
    goal_area: str = "center",
    goal_rows: int = 2,
    goal_cols: int = 2,
) -> np.ndarray:
    """Convert logical maze adjacency to a 0/100 occupancy grid."""
    if adjacency.ndim != 3 or adjacency.shape[2] != 4:
        raise ValueError(
            "adjacency must have shape (cols, rows, 4) with direction axis size 4"
        )

    cols, rows, _ = adjacency.shape
    width = _validate_positive_int("width", width)
    height = _validate_positive_int("height", height)
    goal_rows = _validate_positive_int("goal_rows", goal_rows)
    goal_cols = _validate_positive_int("goal_cols", goal_cols)

    goal = _resolve_goal_region(cols, rows, goal_area, goal_cols, goal_rows)
    exit_location = entrance if exit is None else exit
    exit_cell = _resolve_corner_cell(cols, rows, exit_location)
    exit_side = _ENTRANCE_SIDES[exit_location]

    grid = np.full((width, height), 100.0, dtype=np.float64)

    x_bands = _partition_axis(width, 2 * cols + 1)
    y_bands = _partition_axis(height, 2 * rows + 1)

    def carve_cell(cx: int, cy: int) -> tuple[int, int, int, int]:
        x0, x1 = x_bands[2 * cx + 1]
        y0, y1 = y_bands[2 * cy + 1]
        grid[x0:x1, y0:y1] = 0.0
        return x0, x1, y0, y1

    cell_bounds: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for cx in range(cols):
        for cy in range(rows):
            cell_bounds[(cx, cy)] = carve_cell(cx, cy)

    for cx in range(cols):
        for cy in range(rows):
            x0, x1, y0, y1 = cell_bounds[(cx, cy)]
            if adjacency[cx, cy, 0] and cx + 1 < cols:
                px0, px1 = x_bands[2 * cx + 2]
                grid[px0:px1, y0:y1] = 0.0
            if adjacency[cx, cy, 1] and cy + 1 < rows:
                py0, py1 = y_bands[2 * cy + 2]
                grid[x0:x1, py0:py1] = 0.0

    # Carve the target room so its center is free, including crossing bands.
    gx0, _gx1, gy0, _gy1 = cell_bounds[(goal.x0, goal.y0)]
    _gx0, gx1, _gy0, gy1 = cell_bounds[(goal.x1 - 1, goal.y1 - 1)]
    grid[gx0:gx1, gy0:gy1] = 0.0

    def carve_goal_connection() -> None:
        for gx in range(goal.x0, goal.x1):
            if goal.y0 > 0:
                x0, x1, _y0, _ = cell_bounds[(gx, goal.y0)]
                py0, py1 = y_bands[2 * goal.y0]
                grid[x0:x1, py0:py1] = 0.0
                return
            if goal.y1 < rows:
                x0, x1, _, _y1 = cell_bounds[(gx, goal.y1 - 1)]
                py0, py1 = y_bands[2 * goal.y1]
                grid[x0:x1, py0:py1] = 0.0
                return
        for gy in range(goal.y0, goal.y1):
            if goal.x0 > 0:
                _x0, _, y0, y1 = cell_bounds[(goal.x0, gy)]
                px0, px1 = x_bands[2 * goal.x0]
                grid[px0:px1, y0:y1] = 0.0
                return
            if goal.x1 < cols:
                _, _x1, y0, y1 = cell_bounds[(goal.x1 - 1, gy)]
                px0, px1 = x_bands[2 * goal.x1]
                grid[px0:px1, y0:y1] = 0.0
                return
        raise RuntimeError("Failed to connect goal area to maze")

    carve_goal_connection()

    ex, ey = exit_cell
    x0, x1, _y0, _y1 = cell_bounds[(ex, ey)]
    if exit_side == "south":
        py0, py1 = y_bands[2 * ey]
        grid[x0:x1, py0:py1] = 0.0
    elif exit_side == "north":
        py0, py1 = y_bands[2 * ey + 2]
        grid[x0:x1, py0:py1] = 0.0
    else:
        raise RuntimeError(f"Unsupported exit side: {exit_side}")

    return grid


class MazeGridGenerator(GridMapGenerator):
    """Orthogonal perfect-maze occupancy grid generator."""

    name = "maze"
    yaml_param_names = (
        "rows",
        "cols",
        "entrance",
        "exit",
        "goal_area",
        "goal_rows",
        "goal_cols",
        "seed",
    )

    def __init__(
        self,
        width: int,
        height: int,
        rows: int = 16,
        cols: int = 16,
        entrance: str = "bottom_left",
        exit: str | None = None,
        goal_area: str = "center",
        goal_rows: int = 2,
        goal_cols: int = 2,
        seed: int | None = None,
    ) -> None:
        """Initialize maze parameters."""
        super().__init__()
        self.width = _validate_positive_int("width", width)
        self.height = _validate_positive_int("height", height)
        self.rows = _validate_positive_int("rows", rows)
        self.cols = _validate_positive_int("cols", cols)
        self.exit = entrance if exit is None else exit
        self.entrance = self.exit
        self.goal_area = goal_area
        self.goal_rows = _validate_positive_int("goal_rows", goal_rows)
        self.goal_cols = _validate_positive_int("goal_cols", goal_cols)
        self.seed = seed

        _resolve_corner_cell(self.cols, self.rows, self.exit)
        _resolve_goal_region(
            self.cols,
            self.rows,
            self.goal_area,
            self.goal_cols,
            self.goal_rows,
        )
        min_width = 2 * self.cols + 1
        min_height = 2 * self.rows + 1
        if self.width < min_width or self.height < min_height:
            raise ValueError(
                "Maze grid is too small for the requested logical maze: "
                f"got {self.width}x{self.height}, need at least "
                f"{min_width}x{min_height}. Increase world size or decrease "
                "maze rows/cols."
            )

    def _build_grid(self) -> np.ndarray:
        """Build the occupancy grid from a logical perfect maze."""
        start_cell = _resolve_corner_cell(self.cols, self.rows, self.exit)
        adjacency = generate_maze_adjacency(
            cols=self.cols,
            rows=self.rows,
            seed=self.seed,
            start_cell=start_cell,
        )
        return rasterize_maze(
            adjacency=adjacency,
            width=self.width,
            height=self.height,
            exit=self.exit,
            goal_area=self.goal_area,
            goal_rows=self.goal_rows,
            goal_cols=self.goal_cols,
        )

    def preview(
        self,
        title: str = "Maze Grid Map",
        cmap: str = "gray_r",
    ) -> None:
        """Preview the maze grid with matplotlib."""
        super().preview(title=title, cmap=cmap)

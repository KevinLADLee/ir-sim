from typing import Any, Optional

import numpy as np

from .grid_map_generator_base import GridMapGenerator
from .image_map_generator import ImageGridGenerator
from .obstacle_map import ObstacleMap
from .perlin_map_generator import PerlinGridGenerator


def resolve_obstacle_map(
    obstacle_map: "str | np.ndarray | dict[str, Any] | None" = None,
    world_width: Optional[float] = None,
    world_height: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Resolve obstacle_map to None or a float64 occupancy grid ndarray.

    Generator specs (dict with ``name``) require ``world_width`` and
    ``world_height``; grid size is computed from world size and ``resolution``.

    Returns:
        None, or ndarray (0-100 grid, dtype float64).
    """
    if obstacle_map is None:
        return None
    if isinstance(obstacle_map, np.ndarray):
        return np.asarray(obstacle_map, dtype=np.float64)
    if isinstance(obstacle_map, str):
        gen = ImageGridGenerator(path=obstacle_map).generate()
        return np.asarray(gen.grid, dtype=np.float64)
    if isinstance(obstacle_map, dict) and obstacle_map.get("name"):
        if world_width is None or world_height is None:
            raise ValueError(
                "obstacle_map generator spec requires world_width and "
                "world_height (passed by World.gen_grid_map)."
            )
        return build_grid_from_generator(
            obstacle_map,
            world_width=world_width,
            world_height=world_height,
        )
    raise TypeError(
        "obstacle_map must be None, a path string, an ndarray, or a generator "
        "spec dict with 'name' and 'resolution'."
    )


def build_grid_from_generator(
    spec: dict[str, Any],
    world_width: float,
    world_height: float,
) -> np.ndarray:
    """Build a grid map from a YAML grid_generator spec (name + resolution + params).

    Grid size is always computed from world size and ``resolution`` (meters per
    cell): (world_width / resolution, world_height / resolution) cells.

    Args:
        spec: Dict from YAML, e.g. ``{name: perlin, resolution: 0.1, ...}``.
        world_width: World width in meters.
        world_height: World height in meters.

    Returns:
        Occupancy grid (0-100) as float64 ndarray.
    """
    name = spec.get("name")
    if not name or name not in GridMapGenerator.registry:
        known = ", ".join(GridMapGenerator.registry)
        raise ValueError(
            f"Unknown or missing grid_generator name: {name!r}. Known: {known}"
        )
    resolution = spec.get("resolution")
    if resolution is None:
        raise ValueError(
            "obstacle_map generator spec must include 'resolution' (meters per cell)."
        )
    grid_width = max(1, round(float(world_width) / float(resolution)))
    grid_height = max(1, round(float(world_height) / float(resolution)))

    cls = GridMapGenerator.registry[name]
    params = {
        k: v
        for k, v in spec.items()
        if k not in ("name", "resolution") and k in cls.yaml_param_names
    }
    params["width"] = grid_width
    params["height"] = grid_height

    return np.asarray(cls(**params).generate().grid, dtype=np.float64)


class Map:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        resolution: float = 0.1,
        obstacle_list: Optional[list] = None,
        grid: Optional[np.ndarray] = None,
    ):
        """
        Map class for storing map data and navigation information

        Args:
            width (int): width of the map
            height (int): height of the map
            resolution (float): resolution of the map
            obstacle_list (list): list of obstacle objects for collision detection
            grid (np.ndarray): grid map data for collision detection.
        """

        if obstacle_list is None:
            obstacle_list = []
        self.width = width
        self.height = height
        self.resolution = resolution
        self.obstacle_list = obstacle_list
        self.grid = grid


__all__ = [
    "GridMapGenerator",
    "ImageGridGenerator",
    "Map",
    "ObstacleMap",
    "PerlinGridGenerator",
    "build_grid_from_generator",
    "resolve_obstacle_map",
]

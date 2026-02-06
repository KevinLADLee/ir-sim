from typing import Any, Optional, Union

import numpy as np

from .grid_map_generator_base import GridMapGenerator
from .image_map_generator import ImageGridGenerator
from .obstacle_map import ObstacleMap
from .perlin_map_generator import Perlin2dMap, PerlinGridGenerator


def resolve_obstacle_map(
    obstacle_map: Optional[Union[str, np.ndarray, Any]],
) -> Optional[Union[np.ndarray, Any]]:
    """Resolve obstacle_map to None, an ndarray, or an object with .grid.

    All type branching (path string, generator spec dict, etc.) lives here.
    World and EnvConfig can pass the result through; only this module
    decides how to interpret strings and dicts.

    Returns:
        None, or ndarray (0-100 grid), or object with .grid attribute.
    """
    if obstacle_map is None:
        return None
    if isinstance(obstacle_map, np.ndarray):
        return obstacle_map
    if isinstance(obstacle_map, str):
        return ImageGridGenerator(path=obstacle_map).generate()
    if isinstance(obstacle_map, dict) and obstacle_map.get("name"):
        return build_grid_from_generator(obstacle_map)
    if hasattr(obstacle_map, "grid"):
        return obstacle_map
    return None


def build_grid_from_generator(spec: dict[str, Any]) -> Any:
    """Build a grid map from a YAML grid_generator spec (name + variable params).

    The spec must have ``name`` (e.g. ``perlin``) to select the generator;
    all other keys are passed as constructor params (filtered by the
    generator's ``yaml_param_names``). The returned object has a ``.grid``
    attribute and can be passed as ``obstacle_map`` to
    :py:class:`irsim.world.world.World`.

    Args:
        spec: Dict from YAML, e.g. ``{name: perlin, width: 200, height: 200, ...}``.

    Returns:
        Generator instance with ``.generate()`` called (has ``.grid``).
    """
    name = spec.get("name")
    if not name or name not in GridMapGenerator.registry:
        known = ", ".join(GridMapGenerator.registry)
        raise ValueError(
            f"Unknown or missing grid_generator name: {name!r}. Known: {known}"
        )
    cls = GridMapGenerator.registry[name]
    params = {
        k: v
        for k, v in spec.items()
        if k != "name" and k in cls.yaml_param_names
    }
    return cls(**params).generate()


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
    "Map",
    "ObstacleMap",
    "Perlin2dMap",
    "PerlinGridGenerator",
    "ImageGridGenerator",
    "GridMapGenerator",
    "build_grid_from_generator",
    "resolve_obstacle_map",
]

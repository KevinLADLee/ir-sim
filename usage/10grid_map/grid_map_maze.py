"""
Example: maze escape task via YAML obstacle_map dict spec.

The YAML uses ``obstacle_map: { name: maze, ... }`` so the grid is generated
at load time without an external PNG. The robot starts in the center room and
the goal is placed at the outer exit.
"""

from pathlib import Path

import irsim
from irsim.lib.path_planners import AStarPlanner

env = irsim.make(str(Path(__file__).with_suffix(".yaml")), save_ani=False, full=False)

env_map = env.get_map(resolution=0.1)
planner = AStarPlanner(env_map)

robot_state = env.get_robot_state()
robot_info = env.get_robot_info()
goal_xy = robot_info.goal[:2, 0].tolist()
trajectory = planner.planning(robot_state, goal_xy, show_animation=False)

if trajectory is not None:
    env.draw_trajectory(trajectory, traj_type="r-")
env.render()

env.end(5)

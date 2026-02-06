"""
Example: RRT* path planning on the world occupancy map.

Loads a world from YAML, builds the map, plans a path from robot state to goal
using RRT*, then draws the trajectory and shows the result.
"""

import numpy as np
import irsim
from irsim.lib.path_planners import RRTStar

env = irsim.make("path_planning.yaml", save_ani=False, full=False)

env_map = env.get_map(resolution=0.3)
planner = RRTStar(
    env_map,
    robot_radius=0.3,
    expand_dis=1.5,
    max_iter=500,
    search_until_max_iter=False,
)

robot_state = env.get_robot_state()
robot_info = env.get_robot_info()
trajectory = planner.planning(robot_state, robot_info.goal, show_animation=True)

if trajectory is not None:
    # RRT* returns (rx, ry); draw_trajectory expects (2, n) array
    env.draw_trajectory(np.array(trajectory), traj_type="r-")

env.end(5)

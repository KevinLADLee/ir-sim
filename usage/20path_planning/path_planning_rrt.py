"""
Example: RRT (Rapidly-exploring Random Tree) path planning.

RRT only finds a **feasible** path: it connects each new node to the **nearest**
node and stops as soon as the goal is reached. No path-length optimisation —
the result is often longer and more winding than RRT*.

Compare with path_planning_rrt_star.py to see RRT*'s shorter, optimised paths.
"""

import numpy as np

import irsim
from irsim.lib.path_planners import RRT

env = irsim.make("path_planning.yaml", save_ani=False, full=False)

env_map = env.get_map(resolution=0.1)
planner = RRT(
    env_map,
    robot_radius=0.3,
    expand_dis=1.5,
    max_iter=5000,
    search_until_max_iter=True,
)

robot_state = env.get_robot_state()
robot_info = env.get_robot_info()
trajectory = planner.planning(robot_state, robot_info.goal, show_animation=True)

if trajectory is not None:
    env.draw_trajectory(np.array(trajectory), traj_type="r-")
    path_cost = planner.end.cost
    n_nodes = len(planner.node_list)
    print(f"[RRT] path cost = {path_cost:.3f}, tree nodes = {n_nodes}")
else:
    print("[RRT] no feasible path found")

env.end(5)

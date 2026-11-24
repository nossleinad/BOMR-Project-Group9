"""Global navigation demo: map -> padded obstacles -> visibility graph -> shortest path -> plot.

Run with:  conda activate thymio-project  then  python main.py
"""
import matplotlib.pyplot as plt

import visualization as vis
import global_nav_mix  # shapely padding + our own visibility graph and A*
import global_nav_lib  # shapely padding + extremitypathfinder (visibility graph + A*)

# Thymio: 110 x 112 mm. Its position (from vision) is the centre between the wheels.
ROBOT_RADIUS = 80             # mm, farthest point of the body from the wheel-axle centre
MARGIN = 20                   # mm, safety margin for vision / localisation errors
PADDING = ROBOT_RADIUS + MARGIN
SEED = None                   # put an integer to get the same map again

# 1. Map (later: from the vision module, in mm, origin bottom-left)
obstacles, start, heading, goal = vis.random_map(n_obstacles=4, padding=PADDING, seed=SEED)

# 2. Path planning with both versions, plotted side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, planner in zip(axes, [global_nav_mix, global_nav_lib]):
    path, padded, edges = planner.plan_path(obstacles, start, goal, PADDING, vis.MAP_SIZE)
    print(planner.__name__, "waypoints (mm):", path)
    vis.plot_map(ax, obstacles, padded, start, heading, goal, path, edges, planner.__name__)

plt.tight_layout()
plt.show()

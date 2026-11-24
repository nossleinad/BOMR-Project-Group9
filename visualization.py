"""Random map generation and plotting for the global navigation.

Conventions (same as the vision module):
    - units are millimetres
    - origin at the bottom-left corner of the map, x to the right, y up
    - an obstacle is a list of (x, y) corners, in order around the shape
    - obstacles = list of obstacles  -> [[(x1, y1), (x2, y2), ...], [...], ...]
    - heading is in radians, 0 = along +x, counter-clockwise positive
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, box

from global_nav_mix import pad_obstacles, polygons_of  # same padding as the planners

MAP_SIZE = (1189, 841)  # mm (width, height), A0 sheet in landscape

MIN_GAP = 100           # mm, minimal free space between two random obstacles (< 2 * padding
                        # on purpose: some padded obstacles merge, as can happen in reality)
MIN_START_GOAL = 500    # mm, minimal distance between start and goal


def random_obstacle(rng):
    """Random convex polygon (4 to 7 corners) placed somewhere in the map."""
    n = rng.integers(4, 8)
    # Corners on a circle, sorted by angle -> always convex
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 2 * np.pi / n)
    corners = np.column_stack([np.cos(angles), np.sin(angles)])
    # Stretch and rotate the circle (keeps the polygon convex)
    corners *= rng.uniform(50, 140, size=2)  # half-length and half-width in mm
    a = rng.uniform(0, np.pi)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = corners @ rotation.T + rng.uniform([0, 0], MAP_SIZE)
    return [tuple(c) for c in corners.round(1)]


def random_point_in(rng, region):
    """Random point inside a shapely Polygon."""
    x_min, y_min, x_max, y_max = region.bounds
    while True:
        p = Point(rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))
        if region.contains(p):
            return (round(p.x, 1), round(p.y, 1))


def random_map(n_obstacles=4, padding=100, seed=None):
    """Random obstacles, start position, start heading and goal.

    Start and goal are drawn in the same piece of free space (same padding as the planners),
    so a path always exists.
    Returns obstacles (list of lists of (x, y)), start (x, y), heading (rad), goal (x, y).
    """
    if seed is None:
        seed = np.random.randint(10_000)
    print("Map seed:", seed)
    rng = np.random.default_rng(seed)

    while True:  # new map until start and goal can be placed far enough from each other
        obstacles = random_obstacles(rng, n_obstacles)

        # Padded obstacles can split the free space in pieces: use the biggest one
        _, free_space = pad_obstacles(obstacles, padding, MAP_SIZE)
        region = max(polygons_of(free_space), key=lambda p: p.area)
        for _ in range(100):
            start, goal = random_point_in(rng, region), random_point_in(rng, region)
            if np.hypot(goal[0] - start[0], goal[1] - start[1]) > MIN_START_GOAL:
                heading = rng.uniform(-np.pi, np.pi)
                return obstacles, start, heading, goal


def random_obstacles(rng, n_obstacles):
    """n random convex obstacles, at least MIN_GAP from each other and from the walls."""
    map_area = box(0, 0, *MAP_SIZE).buffer(-MIN_GAP, join_style="mitre")
    obstacles = []
    while len(obstacles) < n_obstacles:
        obstacle = Polygon(random_obstacle(rng))
        inside_map = map_area.contains(obstacle)
        far_enough = all(obstacle.distance(Polygon(o)) > MIN_GAP for o in obstacles)
        if inside_map and far_enough:
            obstacles.append(list(obstacle.exterior.coords)[:-1])
    return obstacles


def path_length(path):
    """Total length of a path given as a list of (x, y)."""
    return sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))


def plot_map(ax, obstacles, padded, start, heading, goal, path=None, edges=(), title=""):
    """Draw the map, the padded obstacles, the visibility graph and the path on `ax`."""
    # Map border
    ax.plot([0, MAP_SIZE[0], MAP_SIZE[0], 0, 0], [0, 0, MAP_SIZE[1], MAP_SIZE[1], 0], "k", lw=2)

    # Visibility graph edges (thin grey lines)
    for a, b in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.85", lw=0.7, zorder=1)

    # Padded obstacles (dashed red) and real obstacles (filled black)
    for poly in padded:
        xs, ys = zip(*poly)
        ax.plot(xs + xs[:1], ys + ys[:1], "r--", lw=1, zorder=2)
    for obstacle in obstacles:
        ax.fill(*zip(*obstacle), color="0.2", zorder=3)

    # Path with numbered waypoints
    if path:
        xs, ys = zip(*path)
        ax.plot(xs, ys, "b-o", lw=2, ms=4, zorder=4)
        for i, (x, y) in enumerate(path):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), color="b")
        title += f"  |  length = {path_length(path):.0f} mm"
    else:
        title += "  |  NO PATH FOUND"

    # Start (position + heading arrow) and goal
    ax.plot(*start, "o", color="tab:blue", ms=10, zorder=5)
    ax.arrow(start[0], start[1], 80 * np.cos(heading), 80 * np.sin(heading),
             width=8, color="tab:blue", zorder=5)
    ax.plot(*goal, "s", color="tab:green", ms=14, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(-20, MAP_SIZE[0] + 20)
    ax.set_ylim(-20, MAP_SIZE[1] + 20)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)

"""Global navigation - "library" version.

    shapely             -> grows (pads) the obstacles (same padding as the mix version, fair comparison)
    extremitypathfinder -> builds the visibility graph AND finds the shortest path (A*)

extremitypathfinder describes the map as ONE boundary polygon (the robot stays inside) with holes
(the obstacles). This is exactly a piece of our free space: its outer outline comes from the walls
and from the padded obstacles touching them, its holes are the padded obstacles in the middle.
"""
from extremitypathfinder import PolygonEnvironment
from shapely.geometry import Point
from shapely.geometry.polygon import orient

from global_nav_mix import pad_obstacles, polygons_of, EPS


def plan_path(obstacles, start, goal, padding, map_size, wall_padding=None):
    """Shortest path from start to goal avoiding the padded obstacles.

    Same inputs / outputs as global_nav_mix.plan_path.
    """
    blocked, free_space = pad_obstacles(obstacles, padding, map_size, wall_padding)
    padded = [list(p.exterior.coords)[:-1] for p in polygons_of(blocked)]

    # The free space can be cut in several separate pieces: keep the one where the robot is
    region = [p for p in polygons_of(free_space) if p.buffer(EPS).covers(Point(start))]
    if not region:
        print("[lib] the start is inside a padded obstacle or too close to a wall")
        return None, padded, []
    region = region[0]
    if not region.buffer(EPS).covers(Point(goal)):
        print("[lib] the goal cannot be reached from the start")
        return None, padded, []

    # orient(sign=1): boundary counter-clockwise, holes clockwise, as the library expects
    region = orient(region, sign=1.0)
    boundary = list(region.exterior.coords)[:-1]
    holes = [list(hole.coords)[:-1] for hole in region.interiors]

    environment = PolygonEnvironment()
    environment.store(boundary, holes)                          # builds the visibility graph
    path, length = environment.find_shortest_path(start, goal)  # A* on it

    # graph = networkx graph between "extremities" (corners where a path can bend)
    coords = environment.coords
    edges = [(tuple(coords[i]), tuple(coords[j])) for i, j in environment.graph.edges]
    path = [(float(x), float(y)) for x, y in path] if path else None
    return path, padded, edges

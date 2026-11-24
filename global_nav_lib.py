"""Global navigation: shortest path from the robot to the goal around the obstacles.

    shapely             -> grows (pads) the obstacles and computes the free space
    extremitypathfinder -> builds the visibility graph AND finds the shortest path (A*)

The robot is reduced to a point (the centre of its body): every obstacle is grown by `padding`,
and the walls are pushed inwards by `wall_padding`. The region left is the free space, where
the robot centre can go without the body touching anything.

extremitypathfinder describes the map as ONE boundary polygon (the robot stays inside) with holes
(the obstacles). This is exactly a piece of our free space: its outer outline comes from the walls
and from the padded obstacles touching them, its holes are the padded obstacles in the middle.
"""
from extremitypathfinder import PolygonEnvironment
from shapely.geometry import Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

EPS = 1e-3  # mm, tolerance: a point exactly on the border of the free space is accepted


def pad_obstacles(obstacles, padding, map_size, wall_padding=None):
    """Grow the obstacles by `padding` and compute the free space of the robot centre.

    buffer(padding): exact growing = rounded corners (disc of radius `padding` around each corner),
        drawn with many small segments -> many useless graph nodes.
    join_style="mitre", mitre_limit=1.0: each corner is instead cut flat by ONE segment tangent to
        that disc -> 2 corners per original corner, and the shape still contains the exact
        rounded one (never closer than `padding` to the obstacle).
    unary_union: obstacles whose padded shapes overlap are merged into one.
    wall_padding: how far the map borders are pushed inwards (None -> same as `padding`,
        0 -> the robot centre may go up to the map edge).
    """
    if wall_padding is None:
        wall_padding = padding
    grown = [Polygon(o).buffer(padding, join_style="mitre", mitre_limit=1.0)
             for o in obstacles if len(o) >= 3]  # a polygon needs at least 3 corners
    blocked = unary_union(grown)
    workspace = box(wall_padding, wall_padding,
                    map_size[0] - wall_padding, map_size[1] - wall_padding)
    free_space = workspace.difference(blocked)
    return blocked, free_space


def polygons_of(geometry):
    """A shapely Polygon or MultiPolygon -> list of Polygons."""
    return list(getattr(geometry, "geoms", [geometry]))


def plan_path(obstacles, start, goal, padding, map_size, wall_padding=None):
    """Shortest path from start to goal avoiding the padded obstacles.

    Inputs (mm): obstacles = [[(x, y), ...], ...], start = (x, y), goal = (x, y),
                 map_size = (width, height)
    Returns:
        path   : list of (x, y) waypoints in mm, start first and goal last (None if no path)
        padded : padded obstacles, list of lists of (x, y) (for drawing)
        edges  : visibility graph edges, list of ((x, y), (x, y)) (for drawing)
    """
    blocked, free_space = pad_obstacles(obstacles, padding, map_size, wall_padding)
    padded = [list(p.exterior.coords)[:-1] for p in polygons_of(blocked)]
    start, goal = tuple(map(float, start)), tuple(map(float, goal))

    # The free space can be cut in several separate pieces: keep the one where the robot is
    region = [p for p in polygons_of(free_space) if p.buffer(EPS).covers(Point(start))]
    if not region:
        print("[global nav] the start is inside a padded obstacle or outside the map")
        return None, padded, []
    region = region[0]
    if not free_space.buffer(EPS).covers(Point(goal)):
        print("[global nav] the goal is inside a padded obstacle or outside the map")
        return None, padded, []
    if not region.buffer(EPS).covers(Point(goal)):
        print("[global nav] the goal cannot be reached from the start (blocked by obstacles)")
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

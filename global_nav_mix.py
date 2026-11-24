"""Global navigation - "mix" version.

    shapely   -> grows (pads) the obstacles and checks if a straight segment is collision-free
    our code  -> builds the visibility graph and runs A* (same structure as Exercise 5)

The robot is reduced to a point (the centre between its wheels): every obstacle is grown by
`padding`, and the walls are pushed inwards by `padding`. The region left is the free space
where the robot centre can go without touching anything.
"""
from heapq import heappush, heappop
import numpy as np
from shapely.geometry import Point, Polygon, LineString, box
from shapely.ops import unary_union
from shapely.prepared import prep

EPS = 1e-3  # mm, tolerance: a segment that just touches a corner of the free space is accepted


def pad_obstacles(obstacles, padding, map_size):
    """Grow the obstacles by `padding` and compute the free space of the robot centre.

    buffer(padding): exact growing = rounded corners (disc of radius `padding` around each corner),
        drawn with many small segments -> many useless graph nodes.
    join_style="mitre", mitre_limit=1.0: each corner is instead cut flat by ONE segment tangent to
        that disc -> 2 corners per original corner, and the shape still contains the exact
        rounded one (never closer than `padding` to the obstacle).
    unary_union: obstacles whose padded shapes overlap are merged into one.
    """
    grown = [Polygon(o).buffer(padding, join_style="mitre", mitre_limit=1.0) for o in obstacles]
    blocked = unary_union(grown)
    workspace = box(padding, padding, map_size[0] - padding, map_size[1] - padding)
    free_space = workspace.difference(blocked)
    return blocked, free_space


def polygons_of(geometry):
    """A shapely Polygon or MultiPolygon -> list of Polygons."""
    return list(getattr(geometry, "geoms", [geometry]))


def corners_of(geometry):
    """All corners (outer boundary and holes) of a Polygon / MultiPolygon, as (x, y) tuples."""
    corners = []
    for poly in polygons_of(geometry):
        for ring in [poly.exterior, *poly.interiors]:
            corners += [tuple(c) for c in ring.coords[:-1]]
    return corners


def visibility_graph(nodes, free_space):
    """Connect every pair of nodes that see each other (segment fully inside the free space).

    Returns graph = {node: [(neighbour, distance), ...]} and the list of edges (for plotting).
    """
    free = prep(free_space.buffer(EPS))  # prepared geometry = faster repeated tests
    graph = {node: [] for node in nodes}
    edges = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if free.covers(LineString([a, b])):
                d = np.hypot(b[0] - a[0], b[1] - a[1])
                graph[a].append((b, d))
                graph[b].append((a, d))
                edges.append((a, b))
    return graph, edges


def heuristic(a, b):
    """Euclidean distance: never longer than the real path -> A* finds the shortest path."""
    return np.hypot(a[0] - b[0], a[1] - b[1])


def a_star(graph, S, G):
    """A* on the visibility graph (same variables as the grid_search of Exercise 5)."""
    came_from = {}                       # to reconstruct the path
    g_costs = {S: 0}                     # cost from start to the node
    explored = set()                     # nodes already expanded
    open_set = [(heuristic(S, G), 0, S)]  # priority queue (f_cost, g_cost, node)

    while open_set:
        current_f_cost, current_g_cost, current = heappop(open_set)
        if current in explored:          # old, more expensive copy of this node
            continue
        explored.add(current)
        if current == G:
            break

        for neighbor, distance in graph[current]:
            tentative_g_cost = current_g_cost + distance
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                came_from[neighbor] = current
                f_cost = tentative_g_cost + heuristic(neighbor, G)
                heappush(open_set, (f_cost, tentative_g_cost, neighbor))

    if G not in explored:
        return None  # no path

    path = [G]
    while path[-1] != S:
        path.append(came_from[path[-1]])
    return path[::-1]


def plan_path(obstacles, start, goal, padding, map_size):
    """Shortest path from start to goal avoiding the padded obstacles.

    Returns:
        path   : list of (x, y) waypoints in mm, start first and goal last (None if no path)
        padded : padded obstacles, list of lists of (x, y) (for plotting)
        edges  : visibility graph edges, list of ((x, y), (x, y)) (for plotting)
    """
    blocked, free_space = pad_obstacles(obstacles, padding, map_size)
    padded = [list(p.exterior.coords)[:-1] for p in polygons_of(blocked)]

    start, goal = tuple(map(float, start)), tuple(map(float, goal))
    for name, p in [("start", start), ("goal", goal)]:
        if not free_space.buffer(EPS).covers(Point(p)):
            print(f"[mix] the {name} is inside a padded obstacle or too close to a wall")
            return None, padded, []

    nodes = [start, goal] + corners_of(free_space)
    graph, edges = visibility_graph(nodes, free_space)
    path = a_star(graph, start, goal)
    return path, padded, edges

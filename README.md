# Thymio project – Vision + Global navigation

## Setup

```bash
conda env create -f environment.yml
conda activate thymio-bomr
python main.py          # camera 1 -> vision -> global path, drawn live (r = replan, e = graph, q = quit)
python main.py 0        # other camera index
python main.py map.png  # image file -> writes main_debug.png
python global_nav_demo.py   # global navigation alone, on random maps (matplotlib)
```

`environment.yml` is the **one environment for every part** (vision, global/local navigation,
filtering, motion control, notebooks); versions are pinned. In VS Code / Jupyter pick the
interpreter or kernel `thymio-bomr`. Filtering tests: `cd FilteringnMotionControl && python test_project.py`.

- numpy uses **OpenBLAS**, not MKL (`libblas=*=*openblas`): with MKL, `np.linalg` (Kalman filter,
  extremitypathfinder) crashed Python on Windows without any error message.
- Adding a package: add it to `environment.yml` (pinned), then `conda env update -f environment.yml --prune`.

Shared constants (map size, robot radius, padding, planner choice) are in `config.py`.

## Files

| File | Role |
|------|------|
| `main.py` | Camera → `vision.py` (every frame) → global path planned once (frozen, `r` to replan) → drawn on the camera view |
| `config.py` | Shared constants and conventions (map size, robot radius, padding, planner) |
| `vision.py` | Homography from the 4 orange beacons, obstacles (raw, not padded), goal, robot pose |
| `global_nav_demo.py` | Global navigation alone: random map → path planning (both versions) → plot |
| `visualization.py` | Random map generator + plot of the map, padding, visibility graph and path |
| `global_nav_mix.py` | **Mix version**: shapely for padding + collision tests, our own visibility graph and A* |
| `global_nav_lib.py` | **Library version**: shapely for padding, extremitypathfinder for visibility graph + shortest path |

Both planners have the same function:

```python
path, padded, edges = plan_path(obstacles, start, goal, padding, map_size)
```

## Data format (agreed with vision)

All in **millimetres**, origin **bottom-left**, x right, y up.

```python
obstacles = [[(x1, y1), (x2, y2), (x3, y3), ...],   # obstacle 1, corners in order around it
             [(x1, y1), (x2, y2), ...],              # obstacle 2
             ...]
start   = (x, y)        # centre of the Thymio BODY (midpoint of the blue/green markers)
heading = theta         # radians, 0 = +x, counter-clockwise positive
goal    = (x, y)        # centre of the goal
```

A list of lists is needed (not one flat list of points): the planner must know which corners
belong to the same obstacle to (1) grow each obstacle and (2) know that a line between two
corners of the same obstacle goes *through* it. `cv2.findContours` + `cv2.approxPolyDP`
already return one ordered contour per obstacle. Vision must flip y (images have y down)
and convert pixels to mm.

The random map generator uses the same padding, puts obstacles at least 100 mm apart and from
the walls (some padded obstacles merge on purpose), and draws start and goal in the same piece
of free space, so a path always exists.

Output for motion control: `path = [(x0, y0), (x1, y1), ..., (xg, yg)]` in mm, start first, goal last.

## Method

> **Values used in the project** (decided when merging with vision, see `config.py` and `TEAM_NOTES.md`):
> robot point = centre of the body, `PADDING = 56 + 10 = 66 mm`, walls NOT padded (`wall_padding=0`).
> The text below describes the original design.

1. **Padding (configuration space).** The robot is reduced to one point, the centre between
   its wheels. Each obstacle is grown by `PADDING = ROBOT_RADIUS + MARGIN` and the walls are
   moved inwards by the same amount. If the point stays out of the grown obstacles, the real
   robot does not touch the real obstacles, even when it turns on the spot.
   - `ROBOT_RADIUS = 80 mm`: distance from the wheel-axle centre to the farthest point of the
     body (Thymio is 110 × 112 mm, the axle is not in the middle) — **measure it on the robot**.
   - `MARGIN = 20 mm`: error of vision / localisation / motion control.
2. **Visibility graph.** Nodes = start, goal and the corners of the free space.
   Two nodes are linked if the straight segment between them stays in the free space.
   Edge cost = Euclidean length. The shortest path in a polygonal map always goes from
   corner to corner, so this graph contains it.
3. **Shortest path.** A* with Euclidean distance to the goal as heuristic (both versions). It
   never overestimates the remaining distance, so A* returns the same optimal path as Dijkstra
   while exploring fewer nodes.

Check done: on 1000 random maps both versions give the same path length and no path segment
enters a padded obstacle.

## Libraries and their parameters

### shapely (both versions)
Geometry library (wrapper of GEOS, the engine used in GIS software / PostGIS).
- `Polygon(corners).buffer(padding, join_style="mitre", mitre_limit=1.0)` grows the polygon by
  `padding` (Minkowski sum with a disc of radius `padding`).
  - The exact result has rounded corners (`join_style="round"`, the default), drawn with many
    small segments → many useless graph nodes, and the segments cut slightly inside the margin.
  - `join_style="mitre"` keeps straight edges; `mitre_limit=1.0` cuts each corner with ONE
    segment tangent to the rounded corner. Result: 2 corners per original corner, and the
    shape always *contains* the exact rounded one (never closer than `padding` to the
    obstacle, at most √2·padding at the cut corners). A larger limit gives long spikes at
    sharp corners that block passages for nothing.
- `unary_union(...)`: merges grown obstacles that overlap (two obstacles too close for the
  robot to pass between them become one).
- `box(...).difference(...)`: free space = shrunk map minus grown obstacles.
- `free.covers(LineString([a, b]))`: is the segment entirely in the free space? It is
  slightly enlarged (`buffer(EPS)`, EPS = 0.001 mm) so a segment that touches a corner exactly
  is accepted despite floating-point rounding.
- `prep(...)`: pre-processes the geometry once so the many `covers` tests are faster.

### extremitypathfinder 2.8.0 (library version only)
Visibility graph + A* for polygon maps (Vinther et al., *Pathfinding in Two-dimensional Worlds*).
The map is one **boundary** polygon (the robot stays inside) with **holes** (obstacles).
- Our free space (shrunk map minus padded obstacles) is exactly that: shapely gives it as
  pieces; we take the piece that contains the start. If the goal is not in it → no path.
- `orient(region, sign=1.0)`: boundary counter-clockwise, holes clockwise, as the library requires.
- `store(boundary, holes)` computes the visibility graph, but only between **extremities**
  (corners pointing into the free space, i.e. reflex corners): a shortest path can only bend
  there, so the other corners are useless. Smaller graph than ours, same result.
- `find_shortest_path(start, goal)` adds start and goal to the graph and runs A*
  (Euclidean heuristic).
- Version pinned to 2.8.0: 3.0.0 returned paths crossing merged obstacles in our tests.
- Why not pyvisgraph (the classic one)? It knows only obstacles, not walls. Cutting the obstacles
  at the walls made it slide along the wall *under* an obstacle; our test caught it.

### heapq (mix version, standard library)
Priority queue used for A*, exactly as in the Exercise 5 solution.

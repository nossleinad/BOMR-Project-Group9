# Thymio project – Vision + Global navigation

## Setup

```bash
conda env create -f environment.yml
conda activate thymio-bomr
python main.py          # camera 1 -> vision -> global path, drawn live (r = replan, e = graph, q = quit)
python main.py 0        # other camera index
python main.py map.png  # image file -> writes main_debug.png
python vision.py        # vision alone (camera 1, or an image file as argument)
```

`environment.yml` is the **one environment for every part** (vision, global/local navigation,
filtering, motion control, notebooks); versions are pinned. In VS Code / Jupyter pick the
interpreter or kernel `thymio-bomr`. Filtering tests: `cd FilteringnMotionControl && python test_project.py`.

- numpy uses **OpenBLAS**, not MKL (`libblas=*=*openblas`): with MKL, `np.linalg` (Kalman filter,
  extremitypathfinder) crashed Python on Windows without any error message.
- Adding a package: add it to `environment.yml` (pinned), then `conda env update -f environment.yml --prune`.

## Files

| File | Role |
|------|------|
| `main.py` | Camera → `vision.py` (every frame) → global path planned once (frozen, `r` to replan) → drawn on the camera view |
| `config.py` | Shared constants and conventions (map size, robot radius, padding) |
| `vision.py` | Homography from the 4 orange beacons, obstacles (raw, not padded), goal, robot pose |
| `global_nav_lib.py` | Global navigation: shapely for padding, extremitypathfinder for visibility graph + shortest path |

```python
path, padded, edges = global_nav_lib.plan_path(obstacles, start, goal, padding, map_size, wall_padding)
```

## Data format (agreed between vision and global navigation)

All in **millimetres**, origin **bottom-left**, x right, y up.

```python
obstacles = [[(x1, y1), (x2, y2), (x3, y3), ...],   # obstacle 1, corners in order around it
             [(x1, y1), (x2, y2), ...],              # obstacle 2
             ...]
start   = (x, y)        # centre of the Thymio BODY (midpoint of the blue/green markers)
heading = theta         # radians, 0 = +x, counter-clockwise positive (blue -> green marker)
goal    = (x, y)        # centre of the red goal
```

A list of lists is needed (not one flat list of points): the planner must know which corners
belong to the same obstacle to (1) grow each obstacle and (2) know that a line between two
corners of the same obstacle goes *through* it. `cv2.findContours` + `cv2.approxPolyDP`
already return one ordered contour per obstacle. Vision flips y (images have y down)
and converts pixels to mm.

Output for motion control: `path = [(x0, y0), (x1, y1), ..., (xg, yg)]` in mm, start first, goal
last (`None` if there is no path).

## Method

1. **Padding (configuration space).** The robot is reduced to one point, the centre of its
   body. Each obstacle is grown by `PADDING_MM = ROBOT_RADIUS_MM + MARGIN_MM = 56 + 10 = 66 mm`
   (`config.py`). If the point stays out of the grown obstacles, the real robot does not touch
   the real obstacles. The walls are not grown (`WALL_PADDING_MM = 0`).
   - `ROBOT_RADIUS_MM = 56`: half the body size (Thymio ≈ 110 × 112 mm). The rear corners are
     ~79 mm from the centre and can clip an obstacle when the robot turns right next to it.
   - `MARGIN_MM = 10`: error of vision / localisation / motion control.
2. **Visibility graph.** Nodes = start, goal and the corners of the free space where a path can
   bend. Two nodes are linked if the straight segment between them stays in the free space.
   Edge cost = Euclidean length. The shortest path in a polygonal map always goes from
   corner to corner, so this graph contains it.
3. **Shortest path.** A* with the Euclidean distance to the goal as heuristic. It never
   overestimates the remaining distance, so A* returns the same optimal path as Dijkstra while
   exploring fewer nodes.

Checks done: on 1000 random maps the library gave the same path length as an independent
implementation (our own visibility graph + A*, removed since), and on 500 random maps of the
real size (976 × 413 mm, padding 66 mm) no path segment enters a padded obstacle.

## Libraries and their parameters

### shapely
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
- `box(...).difference(...)`: free space = map minus grown obstacles.
- `covers(Point(p))` on the free space slightly enlarged (`buffer(EPS)`, EPS = 0.001 mm): is the
  start / goal in the free space, despite floating-point rounding?

### extremitypathfinder 2.8.0
Visibility graph + A* for polygon maps (Vinther et al., *Pathfinding in Two-dimensional Worlds*).
The map is one **boundary** polygon (the robot stays inside) with **holes** (obstacles).
- Our free space (map minus padded obstacles) is exactly that: shapely gives it as
  pieces; we take the piece that contains the start. If the goal is not in it → no path.
- `orient(region, sign=1.0)`: boundary counter-clockwise, holes clockwise, as the library requires.
- `store(boundary, holes)` computes the visibility graph, but only between **extremities**
  (corners pointing into the free space, i.e. reflex corners): a shortest path can only bend
  there, so the other corners are useless.
- `find_shortest_path(start, goal)` adds start and goal to the graph and runs A*
  (Euclidean heuristic).
- Version pinned to 2.8.0: 3.0.0 returned paths crossing merged obstacles in our tests.
- Why not pyvisgraph (the classic one)? It knows only obstacles, not walls. Cutting the obstacles
  at the walls made it slide along the wall *under* an obstacle; our test caught it.

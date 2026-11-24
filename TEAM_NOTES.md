# Global navigation – choices and team interfaces

## Choices made

| Topic | Choice | Why |
|---|---|---|
| Method | Visibility graph + A* (shortest path) | Exact shortest path on a polygon map, no grid cells → smooth waypoints for motion control |
| Two versions | **Mix**: shapely (padding, collision test) + our own visibility graph and A* · **Library**: shapely + extremitypathfinder 2.8.0 | Mix = fully explainable; library = cross-check. Both give the same path on 1000 random maps |
| Obstacles | Convex polygons only (a concave one from vision → use its convex hull) | Simple, robust padding |
| Robot reference point | **Centre between the wheels** | The robot turns around this point |
| Padding | `ROBOT_RADIUS (80 mm, to measure) + MARGIN (20 mm) = 100 mm`, also applied to the walls | Robot reduced to a point; can rotate on the spot without touching anything |
| Units / frame | **mm**, origin **bottom-left**, x right, y up, heading in **rad** (0 = +x, counter-clockwise +) | Same as Thymio specs; one convention for all modules |
| Map size | A0 landscape, 1189 × 841 mm (`MAP_SIZE` in `visualization.py`) | Assumption, change to the real map |
| Code | Python modules + `main.py`, conda env `thymio-project`, folder `BOMR/Project` | Report notebook can import the modules |

## Data in / out

```python
# IN (from vision)
obstacles = [[(x1, y1), (x2, y2), ...],   # one list per obstacle, corners IN ORDER around it
             [(x1, y1), ...], ...]
start   = (x, y)      # centre between the wheels, mm
heading = theta       # rad
goal    = (x, y)      # centre of the goal, mm

# OUT (to motion control)
path = [(x0, y0), (x1, y1), ..., (xg, yg)]   # mm, start first, goal last (None if no path)
```

## To tell / ask the teammates

**Vision**
- Send a **list of lists** (one per obstacle, ordered corners), not one flat list: the planner must know which corners belong to the same obstacle. `cv2.findContours` + `cv2.approxPolyDP` give this directly.
- Convert pixels → **mm** and **flip y** (image y goes down, map y goes up).
- Ask: real map size? How accurate are the corners (in mm)? → sets `MARGIN`.
- Ask: is the robot position the **centre between the wheels**? Is the goal given as its centre?
- Ask: obstacles detected once at the start, or every frame? (Path is planned once at start.)

**Motion control**
- Gets `path` as a list of waypoints in mm. Waypoints are corners of padded obstacles: the robot must go **straight between them** and turn at each one.
- Ask: when is a waypoint "reached" (distance threshold)? That threshold should stay below `MARGIN`.

**Filtering (localisation)**
- Same frame and units (mm, rad, origin bottom-left). Ask for the expected position error → `MARGIN`.

**Local navigation**
- After avoiding a physical obstacle, call `plan_path` again from the **current position** (fast: < 0.1 s) rather than aiming for a waypoint that may now be behind or under the obstacle (a common pitfall from past projects).
- Ask: how will local navigation signal "obstacle passed" so global navigation can replan?

**Everyone**
- Measure on the Thymio the distance from the wheel-axle centre to the farthest point of the body → `ROBOT_RADIUS`.

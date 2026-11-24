# Global navigation – choices and team interfaces

## Choices made

| Topic | Choice | Why |
|---|---|---|
| Method | Visibility graph + A* (shortest path) | Exact shortest path on a polygon map, no grid cells → smooth waypoints for motion control |
| Implementation | **Library only** (`global_nav_lib.py`): shapely (padding, free space) + extremitypathfinder 2.8.0 (visibility graph + A*) | Our own version ("mix") was removed after the merge; both gave the same path on 1000 random maps |
| Obstacles | Raw outlines from vision (`approxPolyDP`), passed as they are (shapely also pads concave ones) | **Vision does NOT pad** (`OBSTACLE_DILATION_MM = 0`): padding is done once, by global nav |
| Robot reference point | **Centre of the robot body** = midpoint of the blue (rear) and green (front) markers, which are stuck so that it is the body centre | *Decided at the vision + global nav merge.* ⚠ Not the rotation centre (wheel axle): filtering/control use a unicycle model around the axle |
| Padding | `ROBOT_RADIUS_MM (56) + MARGIN_MM (10) = 66 mm` on the obstacles, **0 on the walls** (`config.py`) | 56 mm = half the body size from the body centre. ⚠ Rear corners are ~79 mm away: they can clip an obstacle when turning next to it. With no wall padding the body can overhang the map edge |
| Units / frame | **mm**, origin **bottom-left**, x right, y up, heading in **rad** (0 = +x, counter-clockwise +) | Same as Thymio specs; one convention for all modules |
| Map size | 976 × 413 mm = distance between the **centres** of the orange beacons (`config.MAP_SIZE`) | Change it in `config.py` only |
| Code | Python modules, `main.py` (camera + vision + global nav), `global_nav_lib.py` (path planning), conda env `thymio-bomr` | Report notebook can import the modules |

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

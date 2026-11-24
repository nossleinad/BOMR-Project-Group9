"""Vision + global navigation.

    camera -> vision (map, obstacles, goal, robot pose) -> global path -> drawn on the camera view

Vision runs on EVERY frame (later the camera also measures the robot pose for the filter).
Obstacles and goal are taken ONCE: the path is planned on the first frame where the beacons,
the robot and the goal are all found, then the obstacles, goal and path are frozen.

Run with:  conda activate thymio-bomr  then
    python main.py            camera 1 (default)
    python main.py 0          another camera index
    python main.py map.png    an image file (plans once, writes main_debug.png)

Keys (camera window):  r = re-detect obstacles/goal and replan from the current robot position
                       e = show / hide the visibility graph      q = quit
"""
import sys
import time

import cv2

import config
import vision


def get_planner(name):
    """Import only the planner we use (the "lib" one needs extremitypathfinder installed)."""
    if name == "lib":
        import global_nav_lib as planner
    else:
        import global_nav_mix as planner
    return planner


planner = get_planner(config.PLANNER)


def make_plan(r):
    """Plan a global path from a vision result. Returns (plan dict, None) or (None, reason)."""
    if r["robot"] is None:
        return None, "robot not found"
    if r["goal"] is None:
        return None, "goal not found"
    start = r["robot"][:2]
    t0 = time.perf_counter()
    path, padded, edges = planner.plan_path(r["obstacles"], start, r["goal"],
                                            config.PADDING_MM, config.MAP_SIZE,
                                            config.WALL_PADDING_MM)
    dt = time.perf_counter() - t0
    if path is None:
        return None, "no path (start/goal inside a padded obstacle, or goal unreachable)"
    plan = {"obstacles": r["obstacles"], "goal": r["goal"], "path": path,
            "padded": padded, "edges": edges}
    print(f"[main] path planned in {dt * 1000:.0f} ms, {len(path)} waypoints (mm):")
    print("       " + " -> ".join(f"({x:.0f},{y:.0f})" for x, y in path))
    return plan, None


def draw(dbg, plan, status, show_edges):
    if plan is not None:
        vision.draw_plan(dbg, plan["padded"], plan["path"], plan["edges"] if show_edges else ())
    cv2.putText(dbg, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(dbg, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return dbg


def run_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)          # CAP_DSHOW: opens faster on Windows
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index} - try another index (0, 1, 2...).")

    H_cache, plan, want_plan, last_reason, show_edges = None, None, True, None, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r, dbg = vision.process(frame, H_cache)
        if r is None:                                      # no homography yet: raw frame
            status = "corner beacons not found"
        else:
            H_cache = r["homography"]                      # keep the last good warp
            if want_plan:
                plan, reason = make_plan(r)
                if plan is not None:
                    want_plan = False
                elif reason != last_reason:                # print each failure reason once
                    print("[main] cannot plan yet:", reason)
                last_reason = reason
            status = (f"path: {len(plan['path'])} waypoints" if plan else f"waiting: {last_reason}")
            status += "   [r] replan  [e] graph  [q] quit"
        cv2.imshow("vision + global navigation", draw(dbg, plan, status, show_edges))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            plan, want_plan, last_reason = None, True, None
        if key == ord("e"):
            show_edges = not show_edges
    cap.release()
    cv2.destroyAllWindows()


def run_image(filename):
    img = cv2.imread(filename)
    if img is None:
        sys.exit(f"Could not read the image '{filename}' - check the file name.")
    r, dbg = vision.process(img)
    if r is None:
        sys.exit("Could not find the 4 corner beacons - check the orange HSV range.")
    vision.print_result(r)
    plan, reason = make_plan(r)
    if plan is None:
        print("[main] cannot plan:", reason)
    status = f"path: {len(plan['path'])} waypoints" if plan else f"no plan: {reason}"
    cv2.imwrite("main_debug.png", draw(dbg, plan, status, show_edges=True))
    print("[main] wrote main_debug.png")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "1"       # default: camera 1 (external webcam)
    if src.isdigit():
        run_camera(int(src))
    else:
        run_image(src)

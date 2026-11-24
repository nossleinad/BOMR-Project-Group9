

import sys
import math
import cv2
import numpy as np


from config import MAP_W_MM, MAP_H_MM   # distance between the CENTRES of the corner markers (shared with global nav)
PX_PER_MM = 1.0

ARUCO_DICT = cv2.aruco.DICT_4X4_50
CORNER_IDS = [0, 1, 2, 3]       # marker IDs at: top-left, top-right, bottom-right, bottom-left of the map
ROBOT_ID = 4                    # marker ID on the robot

HSV = {
    "red":   [((0, 100, 80), (8, 255, 255)), ((170, 100, 80), (179, 255, 255))],   # goal (red wraps around 0)
    "green": [((40, 70, 35), (100, 255, 255))],                                       # obstacles
}

MIN_GOAL_AREA_MM2 = 100         # smaller red blobs = noise
MIN_OBSTACLE_AREA_MM2 = 2000    
MIN_GOAL_AREA = MIN_GOAL_AREA_MM2 * PX_PER_MM ** 2
MIN_OBSTACLE_AREA = MIN_OBSTACLE_AREA_MM2 * PX_PER_MM ** 2
CAMERA_INDEX = 1

_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                    cv2.aruco.DetectorParameters())


def detect_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def find_homography(img):

    markers = detect_markers(img)
    if not all(i in markers for i in CORNER_IDS):
        return None
    src = np.array([markers[i].mean(axis=0) for i in CORNER_IDS], dtype=np.float32)   
    W, H = int(MAP_W_MM * PX_PER_MM), int(MAP_H_MM * PX_PER_MM)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)           
    return cv2.getPerspectiveTransform(src, dst), (W, H)


def raw_to_mm(pts, M):

    w = cv2.perspectiveTransform(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2), M).reshape(-1, 2)
    return [px_to_mm(p) for p in w]


def px_to_mm(p):

    return (float(p[0]) / PX_PER_MM, MAP_H_MM - float(p[1]) / PX_PER_MM)


def detect_robot(img, M):
    """Robot pose (x, y, theta) in mm / rad from the robot marker, or None if not visible.
    Detected on the RAW image (sharper than the warped one), then converted to the map."""
    markers = detect_markers(img)
    if ROBOT_ID not in markers:
        return None
    tl, tr, br, bl = raw_to_mm(markers[ROBOT_ID], M)
    x = (tl[0] + tr[0] + br[0] + bl[0]) / 4                       
    y = (tl[1] + tr[1] + br[1] + bl[1]) / 4
    front = ((tl[0] + tr[0]) / 2, (tl[1] + tr[1]) / 2)               
    theta = math.atan2(front[1] - y, front[0] - x)
    return x, y, theta


def color_mask(hsv_img, name):
    mask = np.zeros(hsv_img.shape[:2], np.uint8)
    for lo, hi in HSV[name]:
        mask |= cv2.inRange(hsv_img, np.array(lo), np.array(hi))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes
    return mask


def detect_goal(warped):
    """Centre of the biggest red blob, in mm, or None."""
    mask = color_mask(cv2.cvtColor(warped, cv2.COLOR_BGR2HSV), "red")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_GOAL_AREA]
    if not contours:
        return None
    m = cv2.moments(max(contours, key=cv2.contourArea))
    return px_to_mm((m["m10"] / m["m00"], m["m01"] / m["m00"]))


def detect_obstacles(warped):
    mask = color_mask(cv2.cvtColor(warped, cv2.COLOR_BGR2HSV), "green")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for c in contours:
        if cv2.contourArea(c) < MIN_OBSTACLE_AREA:                      # too small: noise
            continue
        poly = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)  # real corners only
        obstacles.append([px_to_mm(p[0]) for p in poly])
    return obstacles


def process(img, hom=None):

    hom = hom or find_homography(img)
    if hom is None:
        return None, img
    M, size = hom
    warped = cv2.warpPerspective(img, M, size)
    result = {
        "homography": hom,
        "obstacles": detect_obstacles(warped),
        "goal": detect_goal(warped),
        "robot": detect_robot(img, M),
    }
    return result, draw_debug(warped, result)


def draw_debug(warped, r):
    out = warped.copy()
    to_px = lambda p: (int(p[0] * PX_PER_MM), int((MAP_H_MM - p[1]) * PX_PER_MM))
    for poly in r["obstacles"]:
        cv2.polylines(out, [np.array([to_px(p) for p in poly])], True, (255, 0, 255), 2)       # magenta
        for p in poly:
            cv2.circle(out, to_px(p), 5, (255, 0, 255), -1)
    if r["goal"] is not None:
        cv2.drawMarker(out, to_px(r["goal"]), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    if r["robot"] is not None:
        x, y, th = r["robot"]
        c = to_px((x, y))
        tip = to_px((x + 60 * math.cos(th), y + 60 * math.sin(th)))
        cv2.circle(out, c, 6, (0, 200, 255), -1)
        cv2.arrowedLine(out, c, tip, (0, 200, 255), 3, tipLength=0.3)
    return out


def mm_to_px(p):
    """Map point in mm (origin bottom-left, y up) -> pixel of the warped image (y down)."""
    return (int(round(p[0] * PX_PER_MM)), int(round((MAP_H_MM - p[1]) * PX_PER_MM)))


def draw_plan(out, padded=(), path=None, edges=()):
    """Draw the global navigation output on top of draw_debug's image (in place)."""
    for a, b in edges:                                           # visibility graph (thin grey)
        cv2.line(out, mm_to_px(a), mm_to_px(b), (200, 200, 200), 1)
    for poly in padded:                                          # padded obstacles (cyan)
        cv2.polylines(out, [np.array([mm_to_px(p) for p in poly])], True, (255, 255, 0), 2)
    if path:                                                     # path + numbered waypoints
        pts = np.array([mm_to_px(p) for p in path])
        cv2.polylines(out, [pts], False, (255, 120, 0), 3)
        for i, p in enumerate(pts):
            cv2.circle(out, tuple(int(v) for v in p), 6, (255, 120, 0), -1)
            cv2.putText(out, str(i), (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
    return out


def print_result(r):
    print(f"obstacles ({len(r['obstacles'])}):")
    for i, poly in enumerate(r["obstacles"]):
        print(f"  {i}: " + ", ".join(f"({x:.0f},{y:.0f})" for x, y in poly))
    print("goal (mm):", None if r["goal"] is None else tuple(round(v) for v in r["goal"]))
    if r["robot"] is not None:
        x, y, th = r["robot"]
        print(f"robot: x={x:.0f} mm  y={y:.0f} mm  heading={math.degrees(th):.1f} deg")
    else:
        print("robot: not found")
    print("-" * 40)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else str(CAMERA_INDEX)

    if src.isdigit():                                    # live camera
        cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            sys.exit(f"Could not open camera {src} - try another index (0, 1, 2...).")
        hom, failed, frame_no = None, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:                                     # dropped frame: skip it
                failed += 1
                if failed > 50:
                    print("Camera stopped sending images.")
                    break
                continue
            failed = 0
            frame_no += 1

            if hom is None:
                hom = find_homography(frame)               # computed once: the camera doesn't move
                if hom is not None:
                    print("Map found (4 corner markers) - homography fixed. Restart if the camera moves.")

            r, dbg = process(frame, hom)
            cv2.imshow("vision", dbg)
            key = cv2.waitKey(1) & 0xFF
            if r is not None and (key == ord("p") or frame_no % 60 == 0):   # print on 'p' or every ~2 s
                print_result(r)
            if key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:                                                  # image file
        img = cv2.imread(src)
        if img is None:
            sys.exit(f"Could not read the image '{src}' - check the file name.")
        r, dbg = process(img)
        if r is None:
            sys.exit("Could not find the 4 corner markers (IDs %s)." % CORNER_IDS)
        print_result(r)
        cv2.imwrite("vision_debug.png", dbg)
        print("debug image written to vision_debug.png")
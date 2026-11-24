
import sys
import math
import cv2
import numpy as np


from config import MAP_W_MM, MAP_H_MM   # shared with global navigation

PX_PER_MM = 1.0
# Obstacles are returned RAW (real outline): the robot-size padding is done by global navigation
# (config.PADDING_MM). Put a value > 0 here only to grow them in vision as well.
OBSTACLE_DILATION_MM = 0


HSV = {
    "orange": [((5, 120, 120), (20, 255, 255))],
    "red":    [((0, 120, 80), (4, 255, 255)), ((170, 120, 80), (179, 255, 255))],  # red wraps around 0
    "green":  [((60, 80, 50), (90, 255, 255))],
    "blue":   [((95, 120, 50), (130, 255, 255))],
}

BLACK_V_MAX = 70         
MIN_BLOB_AREA = 100

BLACK_S_MAX = 80  

         
def color_mask(hsv_img, name):

    binary_mask = np.zeros(hsv_img.shape[:2], np.uint8)
    for low , high in HSV[name]:
        binary_mask |= cv2.inRange(hsv_img, np.array(low) , np.array(high))
    kernel = np.ones((5,5),np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)   

    return binary_mask


def blobs(mask, max_area_frac=0.10):

    blob_list = []
    contours,_=cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours :
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA or area > max_area_frac * MAP_H_MM * MAP_W_MM:
            continue
        m = cv2.moments(cnt)
        blob_list.append (((m["m10"] / m["m00"] , m["m01"] / m["m00"]), cnt, area))
    blob_list.sort(key=lambda b: b[2], reverse=True)

    return [(centroids) for centroids , cnt , area in blob_list]


def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel() 
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def px_to_mm(p): 
    return (p[0] / PX_PER_MM , MAP_H_MM - p[1]/ PX_PER_MM)


def find_homography(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    found = blobs(color_mask(hsv, "orange"))
    if len(found) < 4:
        return None
    src = order_corners(found[:4])
    W, H = int(MAP_W_MM * PX_PER_MM), int(MAP_H_MM * PX_PER_MM)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst), (W, H)





def detect_obstacles(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # 1. black = dark (low V) AND colourless (low S) -> a dark green or blue is no longer "black"
    mask_black = ((hsv[:, :, 2] < BLACK_V_MAX) & (hsv[:, :, 1] < BLACK_S_MAX)).astype(np.uint8) * 255
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 2. keep only the real obstacles BEFORE padding, so small shadows don't get blown up
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = [c for c in contours if cv2.contourArea(c) >= 20 * MIN_BLOB_AREA]
    clean = np.zeros_like(mask_black)
    cv2.drawContours(clean, keep, -1, 255, thickness=-1)        # redraw them filled

    # 3. pad (off by default: global navigation pads the obstacles itself)
    pad_px = int(OBSTACLE_DILATION_MM * PX_PER_MM)
    if pad_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad_px + 1, 2 * pad_px + 1))
        clean = cv2.dilate(clean, kernel)

    # 4. outline -> corners -> mm (small fixed epsilon so the corners don't cut into the padding)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for c in contours:
        poly = cv2.approxPolyDP(c, 3, True)                    # 3 px instead of 2% of the perimeter
        obstacles.append([px_to_mm(p[0]) for p in poly])
    return obstacles


def detect_goal(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red_mask = color_mask(hsv , "red")
    red_blob = blobs(red_mask, max_area_frac=0.1)
    if not red_blob:             
        return None
    return px_to_mm(red_blob[0])


def detect_robot(warped):

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    
    blue_mask = color_mask(hsv , "blue")
    green_mask = color_mask(hsv , "green")
    
    green_blob = blobs(green_mask,max_area_frac=0.1)
    blue_blob = blobs(blue_mask, max_area_frac=0.1)
    if not blue_blob or not green_blob:
        return None         #let kalman filter later
    
    green = green_blob[0]
    blue = min(blue_blob, key=lambda bl: math.dist(bl, green)) 
    
    bm, gm = px_to_mm(blue), px_to_mm(green)

    # position = midpoint of the markers = centre of the robot body; heading = blue -> green
    x, y = (bm[0] + gm[0]) / 2, (bm[1] + gm[1]) / 2
    theta = math.atan2(gm[1] - bm[1], gm[0] - bm[0])

    return x , y , theta


def process(img, H_cache=None):

    hom = find_homography(img) or H_cache
    if hom is None:
        return None, img
    M, size = hom
    warped = cv2.warpPerspective(img, M, size)
    result = {
        "homography": hom,
        "obstacles": detect_obstacles(warped),
        "goal": detect_goal(warped),
        "robot": detect_robot(warped),
    }
    return result, draw_debug(warped, result)
 

def draw_debug(warped, r):
    out = warped.copy()
    to_px = lambda p: (int(p[0] * PX_PER_MM), int((MAP_H_MM - p[1]) * PX_PER_MM))
    for poly in r["obstacles"]:
        cv2.polylines(out, [np.array([to_px(p) for p in poly])], True, (255, 0, 255), 2)
        for p in poly:
            cv2.circle(out, to_px(p), 5, (255, 0, 255), -1)
    if r["goal"]:
        cv2.drawMarker(out, to_px(r["goal"]), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    if r["robot"]:
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
    if r["robot"]:
        x, y, th = r["robot"]
        print(f"robot: x={x:.0f} mm  y={y:.0f} mm  heading={math.degrees(th):.1f} deg")
    else:
        print("robot: not found")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "1"     # default: camera 1 (external webcam)

    if src.isdigit():                                    # a number -> live camera
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)  # CAP_DSHOW: opens faster on Windows
        if not cap.isOpened():
            sys.exit(f"Could not open camera {src} - try another index (0, 1, 2...).")
        H_cache = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r, dbg = process(frame, H_cache)
            if r is not None:
                H_cache = r["homography"]                # keep the last good warp
            cv2.imshow("vision", dbg)
            if cv2.waitKey(1) & 0xFF == ord("q"):        # press q to quit
                break
        cap.release()
        cv2.destroyAllWindows()

    else:                                                # otherwise -> image file
        img = cv2.imread(src)
        if img is None:
            sys.exit(f"Could not read the image '{src}' - check the file name.")
        r, dbg = process(img)
        if r is None:
            sys.exit("Could not find the 4 corner beacons - check the orange HSV range.")
        print_result(r)
        cv2.imwrite("vision_debug.png", dbg)
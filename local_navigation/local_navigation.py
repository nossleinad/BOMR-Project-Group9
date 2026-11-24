"""
local_navigation.py
Avoids circular obstacles detected by the proximity sensors.
States: GLOBAL (follow the path) <-> LOCAL (go around the obstacle)
"""

import math

ENTER_THRESHOLD = 2100     # prox value at ~10 cm -> switch to LOCAL
EXIT_THRESHOLD = 1000      # lower than ENTER_THRESHOLD (hysteresis)
DANGER_THRESHOLD = 3500    # very close -> turn on the spot
MIN_LOCAL_STEPS = 5     # minimum number of steps spent in LOCAL

OBSTACLE_RADIUS = 4.0     # cm
ROBOT_HALF_WIDTH = 5.5    # cm
MARGIN = 2.0              # cm

TURN_SPEED = 100          # motor units, TURN_AWAY phase
ARC_SPEED = 100           # motor units, ARC phase

REACH_RADIUS = 3.0        # cm

K_RADIAL = 0.25                    # rad/cm
MAX_CORRECTION = math.radians(60)  # rad
K_HEADING = 120                    # motor units/rad
W_MAX = 150                        # motor units

SENSOR_ANGLES_DEG = [40, 20, 0, -20, -40]  # sensors 0..4, + = left
SENSOR_OFFSET = 8.0                        # cm, axle centre -> front sensors
MEASURE_THRESHOLD = 1900
PROX_A = -396.05                           # value = PROX_A * d + PROX_B
PROX_B = 5105.4


class LocalNavigation:
    """State machine that switches between global navigation and local
    obstacle avoidance."""

    def __init__(self):
        """Start in GLOBAL with no obstacle detected."""
        self.state = "GLOBAL"
        self.phase = None
        self.side = None
        self.steps_in_local = 0
        self.centre = None
        self.centre_estimates = []

    def update(self, prox, pose, waypoint, map_is_free=None):
        """Run one control step. Called at every loop iteration by the main.

        prox        : list of 7 values (prox.horizontal)
        pose        : (x, y, theta) from the filter
        waypoint    : (x, y) current target of the global path
        map_is_free : function (x, y) -> bool given by global navigation
        returns     : (state, left_speed, right_speed),
                      speeds are None when state == "GLOBAL"
        """
        if self.state == "GLOBAL":
            if self.obstacle_detected(prox):
                self.state = "LOCAL"
                self.phase = "TURN_AWAY"
                self.steps_in_local = 0
                self.side = self.choose_side(prox, pose, waypoint, map_is_free)
                self.centre = None
                self.centre_estimates = []
            else:
                return "GLOBAL", None, None

        self.steps_in_local += 1
        self.update_centre(prox, pose)

        if self.can_leave(prox, pose, waypoint):
            self.state = "GLOBAL"
            return "GLOBAL", None, None

        left, right = self.avoid(prox, pose)
        return "LOCAL", left, right

    def obstacle_detected(self, prox):
        """Return True if one of the front sensors (0 to 4) is above
        ENTER_THRESHOLD."""
        for i in range(5):
            if prox[i] > ENTER_THRESHOLD:
                return True
        return False

    def update_centre(self, prox, pose):
        """Each time a sensor sees the obstacle well, compute an estimate of
        its centre and average all estimates since entering LOCAL."""
        if max(prox[:5]) > MEASURE_THRESHOLD:
            self.centre_estimates.append(estimate_obstacle_centre(prox, pose))
            xs = [c[0] for c in self.centre_estimates]
            ys = [c[1] for c in self.centre_estimates]
            self.centre = (sum(xs) / len(xs), sum(ys) / len(ys))

    def choose_side(self, prox, pose, waypoint, map_is_free):
        """Choose once, when entering LOCAL, on which side to go around.

        Go to the side whose sensors see the obstacle less (the outer sensor
        counts double). If both sides are equal (obstacle straight ahead),
        go to the side of the waypoint.
        Returns "LEFT" or "RIGHT".
        """
        left_value = prox[0] * 2 + prox[1] * 1
        right_value = prox[4] * 2 + prox[3] * 1

        if left_value < right_value:
            self.side = "LEFT"
            return "LEFT"
        elif right_value < left_value:
            self.side = "RIGHT"
            return "RIGHT"
        else:
            angle = angle_to(pose, waypoint)
            if angle > 0:
                self.side = "LEFT"
                return "LEFT"
            else:
                self.side = "RIGHT"
                return "RIGHT"

    def avoid(self, prox, pose):
        """Compute the wheel speeds while in LOCAL.

        TURN_AWAY : turn on the spot toward the chosen side until the front
                    sensors are clear, then switch to ARC.
        ARC       : go around the obstacle with the P controller. If a sensor
                    sees the obstacle too close, switch back to TURN_AWAY.
        Returns (left_speed, right_speed).
        """
        if self.phase == "TURN_AWAY":
            if all(prox[i] < EXIT_THRESHOLD for i in range(5)):
                self.phase = "ARC"
            elif self.side == "LEFT":
                return -TURN_SPEED, TURN_SPEED
            else:
                return TURN_SPEED, -TURN_SPEED

        if self.phase == "ARC":
            if any(prox[i] > DANGER_THRESHOLD for i in range(5)):
                self.phase = "TURN_AWAY"
                if self.side == "LEFT":
                    return -TURN_SPEED, TURN_SPEED
                return TURN_SPEED, -TURN_SPEED
            return self.circle_p_controller(pose)

    def circle_p_controller(self, pose):
        """Keep the robot on a circle of radius R around the obstacle centre.

        P term 1: distance error (r - R) -> correction of the desired heading
                  (too far: steer toward the centre, too close: steer away).
        P term 2: heading error -> rotation speed of the robot.
        s = -1 if we went LEFT (obstacle on the right, clockwise),
        s = +1 if we went RIGHT (obstacle on the left, counter-clockwise).
        Returns (left_speed, right_speed).
        """
        x, y, theta = pose
        cx, cy = self.centre
        R = OBSTACLE_RADIUS + ROBOT_HALF_WIDTH + MARGIN
        s = -1 if self.side == "LEFT" else +1

        r = math.hypot(x - cx, y - cy)
        phi = math.atan2(y - cy, x - cx)
        tangent = phi + s * math.pi / 2

        correction = K_RADIAL * (r - R)
        correction = max(-MAX_CORRECTION, min(MAX_CORRECTION, correction))
        desired = tangent + s * correction

        e = wrap_angle(desired - theta)
        w = K_HEADING * e
        w = max(-W_MAX, min(W_MAX, w))

        v = ARC_SPEED * max(0.0, math.cos(e))
        return int(v - w), int(v + w)

    def can_leave(self, prox, pose, waypoint):
        """Return True when the robot can go back to GLOBAL, i.e. when all of
        these hold:
        - at least MIN_LOCAL_STEPS steps spent in LOCAL,
        - all front sensors below EXIT_THRESHOLD,
        - phase is ARC,
        - the waypoint is no longer behind the obstacle (pledge idea): it is
          straight ahead or on the free side.
        """
        if self.steps_in_local >= MIN_LOCAL_STEPS and \
           all(prox[i] < EXIT_THRESHOLD for i in range(5)) and \
           self.phase == "ARC":
            angle = angle_to(pose, waypoint)
            if self.side == "LEFT" and angle >= -0.1:
                return True
            elif self.side == "RIGHT" and angle <= 0.1:
                return True
        return False


def angle_to(pose, point):
    """Angle of the point seen from the robot, relative to its heading,
    in [-pi, pi] (positive = to the left)."""
    return wrap_angle(math.atan2(point[1] - pose[1], point[0] - pose[0]) - pose[2])


def wrap_angle(a):
    """Bring an angle back to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def prox_to_distance(value):
    """Convert a raw proximity value into a distance [cm] with the
    calibration, limited to the sensor range."""
    d = (value - PROX_B) / PROX_A
    return min(max(d, 3.5), 8.5)


def estimate_obstacle_centre(prox, pose):
    """Estimate the obstacle centre in the world frame from one reading.
    The strongest sensor gives the direction, its value gives the distance,
    and the centre is one obstacle radius behind the measured surface."""
    x, y, theta = pose
    i = max(range(5), key=lambda k: prox[k])
    d = prox_to_distance(prox[i])
    angle = theta + math.radians(SENSOR_ANGLES_DEG[i])
    r = SENSOR_OFFSET + d + OBSTACLE_RADIUS
    return (x + r * math.cos(angle), y + r * math.sin(angle))


def next_waypoint_index(path, index, pose, avoiding=False):
    """Return the index of the waypoint the robot should go to.

    - Skip the waypoint if it is reached.
    - While avoiding: also skip it if the robot is closer to the next waypoint
      than the current waypoint is (the robot has gone past it, or it was
      under the obstacle and the robot went around it).
    The goal (last waypoint) is never skipped.
    """
    robot = (pose[0], pose[1])
    while index < len(path) - 1:
        a = path[index]
        b = path[index + 1]
        reached = math.dist(robot, a) < REACH_RADIUS
        passed = avoiding and math.dist(robot, b) < math.dist(a, b)
        if reached or passed:
            index += 1
        else:
            break
    return index

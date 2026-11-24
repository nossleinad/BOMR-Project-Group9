"""
local_navigation_sim.py
Tests local_navigation.py without the Thymio: a simulated robot with
simulated proximity sensors drives along a path with circular obstacles.

Run:  python local_navigation_sim.py
It prints, for each scenario, whether the goal was reached and whether the
robot touched an obstacle, then saves local_navigation_sim_results.png.

The constants still set to ... in local_navigation.py are replaced by test
values (TEST_VALUES below). Values you already measured are kept.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import local_navigation as ln
from local_navigation import LocalNavigation, next_waypoint_index, angle_to

TEST_VALUES = {
    "ENTER_THRESHOLD": 2000,
    "EXIT_THRESHOLD": 1000,
    "DANGER_THRESHOLD": 3500,
    "MIN_LOCAL_STEPS": 5,
    "OBSTACLE_RADIUS": 4.0,
    "MARGIN": 3.0,
    "TURN_SPEED": 100,
    "ARC_SPEED": 100,
}
for name, value in TEST_VALUES.items():
    if getattr(ln, name) is Ellipsis:
        setattr(ln, name, value)

DT = 0.1              # s, control period
SPEED_TO_CMS = 0.04   # cm/s per motor unit
WHEEL_BASE = 9.5      # cm
SENSOR_RANGE = 13.0   # cm

# Front sensors: position (cm, robot frame, from the axle centre) and angle (deg)
SENSOR_POSE = [(6.3, 4.0, 40), (7.4, 2.1, 20), (7.9, 0.0, 0),
               (7.4, -2.1, -20), (6.3, -4.0, -40)]
# Points of the robot outline, used to detect collisions
OUTLINE = [(-3.0, 5.5), (-3.0, -5.5), (2.0, 5.5), (2.0, -5.5), (6.3, 4.0),
           (7.4, 2.1), (7.9, 0.0), (7.4, -2.1), (6.3, -4.0), (-3.0, 0.0)]


def sim_prox(pose, obstacles, rng=None):
    """Simulated prox.horizontal: each sensor is a cone of 5 rays, the
    closest hit is converted to a value with the calibration line."""
    x, y, th = pose
    values = [0] * 7
    for i, (sx, sy, sa) in enumerate(SENSOR_POSE):
        sa = math.radians(sa)
        px = x + sx * math.cos(th) - sy * math.sin(th)
        py = y + sx * math.sin(th) + sy * math.cos(th)
        best = math.inf
        for da in np.radians([-12, -6, 0, 6, 12]):
            dx, dy = math.cos(th + sa + da), math.sin(th + sa + da)
            for ox, oy, r in obstacles:
                fx, fy = px - ox, py - oy
                b = fx * dx + fy * dy
                c = fx * fx + fy * fy - r * r
                disc = b * b - c
                if c < 0:                       # sensor inside the obstacle
                    best = 0.0
                elif disc >= 0 and -b - math.sqrt(disc) >= 0:
                    best = min(best, -b - math.sqrt(disc))
        if best <= SENSOR_RANGE:
            v = ln.PROX_A * best + ln.PROX_B
            if rng is not None:
                v *= 1 + 0.08 * rng.standard_normal()
            values[i] = int(np.clip(v, 0, 5000))
    return values


def collides(pose, obstacles):
    """True if a point of the robot outline is inside an obstacle."""
    x, y, th = pose
    for bx, by in OUTLINE:
        px = x + bx * math.cos(th) - by * math.sin(th)
        py = y + bx * math.sin(th) + by * math.cos(th)
        if any(math.hypot(px - ox, py - oy) < r for ox, oy, r in obstacles):
            return True
    return False


def motion_control(pose, waypoint):
    """Simple stand-in for the teammates' controller (GLOBAL state)."""
    a = angle_to(pose, waypoint)
    v = 150 * max(0.0, math.cos(a))
    return v - 150 * a, v + 150 * a


def run(path, obstacles, rng=None, t_max=90):
    """Simulate one run. Returns (reached, collided, trajectory)."""
    nav = LocalNavigation()
    pose = [path[0][0], path[0][1], 0.0]
    index = 1
    collided = False
    traj = []
    for _ in range(int(t_max / DT)):
        index = next_waypoint_index(path, index, pose,
                                    avoiding=(nav.state == "LOCAL"))
        if index == len(path) - 1 and math.dist(pose[:2], path[-1]) < 3:
            return True, collided, traj
        if rng is None:
            measured = tuple(pose)
        else:                                   # noisy pose from the "filter"
            measured = (pose[0] + rng.normal(0, 0.7),
                        pose[1] + rng.normal(0, 0.7),
                        pose[2] + rng.normal(0, 0.05))
        state, left, right = nav.update(sim_prox(pose, obstacles, rng),
                                        measured, path[index])
        if state == "GLOBAL":
            left, right = motion_control(pose, path[index])
        traj.append((pose[0], pose[1], state))

        left, right = np.clip([left, right], -500, 500) * SPEED_TO_CMS
        v, w = (left + right) / 2, (right - left) / WHEEL_BASE
        pose[0] += v * math.cos(pose[2]) * DT
        pose[1] += v * math.sin(pose[2]) * DT
        pose[2] += w * DT
        collided |= collides(pose, obstacles)
    return False, collided, traj


SCENARIOS = {
    "centre":      ([(0, 0), (100, 0)], [(50, 0, 4)]),
    "left":        ([(0, 0), (100, 0)], [(50, 4, 4)]),
    "right":       ([(0, 0), (100, 0)], [(50, -4, 4)]),
    "on_waypoint": ([(0, 0), (60, 0), (60, 60)], [(60, 0, 4)]),
    "before_turn": ([(0, 0), (50, 0), (80, 40)], [(30, 1, 4)]),
    "two":         ([(0, 0), (120, 0)], [(40, 1, 4), (85, -2, 4)]),
}

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, (path, obstacles)) in zip(axes.flat, SCENARIOS.items()):
        reached, collided, traj = run(path, obstacles)
        print(f"{name:12s} reached={reached}  collided={collided}")
        ax.plot(*zip(*path), "k--", lw=1, label="global path")
        for ox, oy, r in obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, color="grey"))
        for state, colour in (("GLOBAL", "tab:blue"), ("LOCAL", "tab:red")):
            pts = [(x, y) for x, y, s in traj if s == state]
            if pts:
                ax.plot(*zip(*pts), ".", ms=2, color=colour, label=state)
        ax.set_title(f"{name}: reached={reached}, collided={collided}")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("local_navigation_sim_results.png", dpi=120)

    rng = np.random.default_rng(1)
    results = [run([(0, 0), (100, 0)], [(50, rng.uniform(-6, 6), 4)], rng)
               for _ in range(40)]
    print("40 noisy runs: reached", sum(r[0] for r in results),
          "collisions", sum(r[1] for r in results))

"""
local_navigation_sim.py
Tests local_navigation.py without the Thymio: a simulated robot with
simulated proximity sensors drives along a path with circular obstacles.
Units: mm and rad. The robot moves around its wheel axle, but the module
receives the pose of the body centre, as given by vision.

Run:  python local_navigation_sim.py
It prints, for each scenario, whether the goal was reached and whether the
robot touched an obstacle, then saves local_navigation_sim_results.png.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import local_navigation as ln
from local_navigation import LocalNavigation, next_waypoint_index, angle_to

DT = 0.1              # s, control period
SPEED_TO_MMS = 0.4    # mm/s per motor unit
WHEEL_BASE = 95.0     # mm
AXLE_TO_CENTRE = 24.5 # mm, wheel axle -> body centre (forward)
SENSOR_RANGE = 85.0   # mm, the prox reads 0 beyond

# Front sensors: position (mm, robot frame, from the axle centre) and angle (deg)
SENSOR_POSE = [(63, 40, 40), (74, 21, 20), (79, 0, 0),
               (74, -21, -20), (63, -40, -40)]
# Points of the robot outline (mm, from the axle centre), used to detect collisions
OUTLINE = [(-30, 55), (-30, -55), (20, 55), (20, -55), (63, 40),
           (74, 21), (79, 0), (74, -21), (63, -40), (-30, 0)]


def body_centre(axle_pose):
    """Pose of the body centre from the pose of the axle centre."""
    x, y, th = axle_pose
    return (x + AXLE_TO_CENTRE * math.cos(th), y + AXLE_TO_CENTRE * math.sin(th), th)


def sim_prox(pose, obstacles, rng=None):
    """Simulated prox.horizontal (pose = axle): each sensor is a cone of 5
    rays, the closest hit is converted to a value with the calibration line."""
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
    """True if a point of the robot outline is inside an obstacle (pose = axle)."""
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
    """Simulate one run. path[0] is the start of the body centre.
    Returns (reached, collided, trajectory of the body centre)."""
    nav = LocalNavigation()
    axle = [path[0][0] - AXLE_TO_CENTRE, path[0][1], 0.0]
    index = 1
    collided = False
    traj = []
    for _ in range(int(t_max / DT)):
        pose = body_centre(axle)
        index = next_waypoint_index(path, index, pose,
                                    avoiding=(nav.state == "LOCAL"))
        if index == len(path) - 1 and math.dist(pose[:2], path[-1]) < 30:
            return True, collided, traj
        if rng is None:
            measured = pose
        else:                                   # noisy pose from the "filter"
            measured = (pose[0] + rng.normal(0, 7),
                        pose[1] + rng.normal(0, 7),
                        pose[2] + rng.normal(0, 0.05))
        state, left, right = nav.update(sim_prox(axle, obstacles, rng),
                                        measured, path[index])
        if state == "GLOBAL":
            left, right = motion_control(pose, path[index])
        traj.append((pose[0], pose[1], state))

        left, right = np.clip([left, right], -500, 500) * SPEED_TO_MMS
        v, w = (left + right) / 2, (right - left) / WHEEL_BASE
        axle[0] += v * math.cos(axle[2]) * DT
        axle[1] += v * math.sin(axle[2]) * DT
        axle[2] += w * DT
        collided |= collides(axle, obstacles)
    return False, collided, traj


R_OBS = ln.OBSTACLE_RADIUS
SCENARIOS = {
    "centre":      ([(0, 0), (1000, 0)], [(500, 0, R_OBS)]),
    "left":        ([(0, 0), (1000, 0)], [(500, 40, R_OBS)]),
    "right":       ([(0, 0), (1000, 0)], [(500, -40, R_OBS)]),
    "on_waypoint": ([(0, 0), (600, 0), (600, 600)], [(600, 0, R_OBS)]),
    "before_turn": ([(0, 0), (500, 0), (800, 400)], [(300, 10, R_OBS)]),
    "two":         ([(0, 0), (1200, 0)], [(400, 10, R_OBS), (850, -20, R_OBS)]),
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
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("local_navigation_sim_results.png", dpi=120)

    rng = np.random.default_rng(1)
    results = [run([(0, 0), (1000, 0)], [(500, rng.uniform(-60, 60), R_OBS)], rng)
               for _ in range(40)]
    print("40 noisy runs: reached", sum(r[0] for r in results),
          "collisions", sum(r[1] for r in results))

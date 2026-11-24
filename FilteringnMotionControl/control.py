"""
Setup: The robot will follow a trajectory that connects a set of vertices with line segments.
Implementation: 2 different control scenarios:
    1. Turning at a vertex
    2. Following a line segment

The canonical state is [x1, x2, theta]
The canonical input is [v, omega], where v stands for velocity (mm/s) and omega for angular velocity (rad/s)
"""

import numpy as np

from conversions import wrap


class LineSegment:
    def __init__(self, p1, p2):
        self.p1 = np.asarray(p1, dtype=float)
        self.p2 = np.asarray(p2, dtype=float)
        diff = self.p2-self.p1
        self.length = np.linalg.norm(diff)
        self.theta = np.arctan2(diff[1], diff[0])

    def dist(self, p):
        """Signed distance from the line: > 0 when p is to the left of p1 -> p2."""
        p0 = np.asarray(p, dtype=float)-self.p1
        diff = self.p2-self.p1
        projection = np.dot(p0, diff)/np.dot(diff, diff)*diff
        side = np.sign(diff[0]*p0[1] - diff[1]*p0[0])
        return side*np.linalg.norm(p0-projection)

    def progress(self, p):
        """Distance travelled along the segment (>= self.length once p is past p2)."""
        p0 = np.asarray(p, dtype=float)-self.p1
        return np.dot(p0, self.p2-self.p1)/self.length


def turn_gain(state, ref, K_p=1.0):
    theta = state[2]
    return np.array([0, K_p * wrap(ref-theta)])


def follow_gain(state, ls, K_p=np.ones(2)):
    """Returns omega (v is chosen by the caller)."""
    k1, k2 = K_p
    dist = ls.dist(state[0:2])
    heading_err = wrap(state[2] - ls.theta)
    return -k1*dist-k2*heading_err

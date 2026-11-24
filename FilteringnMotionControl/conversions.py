"""Conversions between the canonical input [v, omega] (mm/s, rad/s) and Thymio motor units [left, right]."""

import numpy as np

S = 0.329605  # mm/s per Thymio speed unit
D = 95                  # wheel base [mm]


def tr_actuators(input, s=S, d=D):
    v, omega = input
    return 1/s*np.array([v-d/2*omega, v+d/2*omega])


def invtr_actuators(actuators, s=S, d=D):
    v1, v2 = s*np.asarray(actuators, dtype=float)
    return np.array([(v1+v2)/2, (v2-v1)/d])


def wrap(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2*np.pi) - np.pi

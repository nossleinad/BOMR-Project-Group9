"""Sanity checks for conversions.py, control.py and filtering.py.

Run with `python test_project.py` (or `pytest test_project.py` if pytest is installed)."""

import numpy as np

from control import LineSegment, follow_gain, turn_gain
from conversions import invtr_actuators, tr_actuators, wrap
from filtering import OMEGA_EPS, ProjectThymio, motion_model


def test_conversions_roundtrip():
    v, w = invtr_actuators(tr_actuators(np.array([50.0, 0.3])))
    assert np.isclose(v, 50) and np.isclose(w, 0.3)
    left, right = tr_actuators([0, 1.0])
    assert left < 0 < right, "omega > 0 is counter-clockwise: right wheel faster"
    assert np.isclose(wrap(3 * np.pi / 2), -np.pi / 2)


def test_line_segment():
    ls = LineSegment([0, 0], [100, 0])
    assert abs(ls.theta) < 1e-12 and np.isclose(ls.length, 100)
    assert np.isclose(ls.dist([50, 10]), 10) and np.isclose(ls.dist([50, -10]), -10), "signed, > 0 on the left"
    assert np.isclose(ls.progress([30, 5]), 30) and ls.progress([120, 0]) > ls.length
    diag = LineSegment([100, 100], [200, 200])
    assert np.isclose(diag.theta, np.pi / 4)
    assert np.isclose(diag.dist([100, 200]), np.sqrt(2) * 50), "measured from p1, not the origin"


def test_gains():
    u = turn_gain(np.array([0, 0, 3.0]), -3.0)
    assert u[0] == 0 and u[1] > 0, "returns [0, omega] and turns the short way"
    ls = LineSegment([0, 0], [100, 0])
    assert follow_gain(np.array([50, 10, 0.0]), ls, K_p=(0.02, 1.5)) < 0, "left of the line -> turn right"
    assert follow_gain(np.array([50, 0, 0.3]), ls, K_p=(0.02, 1.5)) < 0, "heading left of the line -> turn right"
    assert np.isclose(follow_gain(np.array([50, 0, 0.0]), ls), 0)


def test_motion_model_jacobians():
    """A and V (through Q) must be the Jacobians of predict wrt the state and the input."""
    alphas = np.array([1.0, 2.0, 3.0, 4.0])
    pt = ProjectThymio(0.1, np.zeros((3, 3)), np.eye(3), alphas=alphas)
    cases = [(np.array([10, 5, 0.3]), np.array([50, 0.4])),
             (np.array([10, 5, 2.0]), np.array([60, 0.0])),     # straight: small-omega branch
             (np.array([10, 5, -2.5]), np.array([60, 5e-4])),   # small-omega branch, omega != 0
             (np.array([10, 5, 1.0]), np.array([60, -2e-3])),   # exact branch, close to the switch
             (np.array([0, 0, -1.0]), np.array([0, -0.8]))]     # turning on the spot
    e = 1e-6
    for mu, u in cases:
        A, C, Q, R = pt.lm(mu, u)
        assert np.all(np.isfinite(A)) and np.all(np.isfinite(Q))
        A_num = np.column_stack([(pt.predict(mu + e * d, u) - pt.predict(mu - e * d, u)) / (2 * e) for d in np.eye(3)])
        assert np.allclose(A, A_num, atol=1e-4)
        V_num = np.column_stack([(pt.predict(mu, u + e * d) - pt.predict(mu, u - e * d)) / (2 * e) for d in np.eye(2)])
        M = np.diag([alphas[0] * u[0] ** 2 + alphas[1] * u[1] ** 2, alphas[2] * u[0] ** 2 + alphas[3] * u[1] ** 2])
        assert np.allclose(Q, V_num @ M @ V_num.T, rtol=1e-4, atol=1e-4)


def test_small_omega_switch_is_continuous():
    """Just below and just above OMEGA_EPS the two branches must agree. predict and A agree to
    O((omega*Ts)^2). V, and so Q, only agrees to O(omega*Ts), since the midpoint limit's derivative
    wrt omega is first-order accurate."""
    pt = ProjectThymio(0.1, np.zeros((3, 3)), np.eye(3), alphas=np.array([1.0, 2.0, 3.0, 4.0]))
    for th in np.linspace(-np.pi, np.pi, 37):
        mu = np.array([5.0, -3.0, th])
        lo, hi = np.array([60, OMEGA_EPS * (1 - 1e-9)]), np.array([60, OMEGA_EPS * (1 + 1e-9)])
        assert np.allclose(pt.predict(mu, lo), pt.predict(mu, hi), rtol=0, atol=1e-8)
        (A_lo, _, Q_lo, _), (A_hi, _, Q_hi, _) = pt.lm(mu, lo), pt.lm(mu, hi)
        assert np.allclose(A_lo, A_hi, rtol=0, atol=1e-8)
        assert np.abs(Q_lo - Q_hi).max() < 1e-4 * np.abs(Q_hi).max()


def test_motion_model_straight_line():
    pt = ProjectThymio(0.1, np.zeros((3, 3)), np.eye(3))
    mu, sigma = np.array([0.0, 0.0, np.pi / 2]), np.eye(3)
    for _ in range(10):
        mu, sigma = motion_model(mu, sigma, np.array([50.0, 0.0]), pt)
    assert np.allclose(mu, [0, 50, np.pi / 2], atol=1e-12)  # exact: no heading drift when omega == 0
    assert np.all(np.isfinite(sigma)) and np.all(np.linalg.eigvalsh(sigma) > 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")

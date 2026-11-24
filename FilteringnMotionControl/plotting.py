"""Plots for demo.ipynb: the live path view (mm), the time series after a run, and a text summary."""

import math

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import Polygon

from conversions import wrap

# Approximate Thymio footprint in the body frame [mm] (origin = wheel-axle midpoint, x forward)
_BODY = np.array([[-30, -55], [55, -55], [80, -30], [80, 30], [55, 55], [-30, 55]], dtype=float)
_CHI2_95 = 5.991  # 95% quantile, 2 dof


def ellipse_points(mu, sigma, n=60):
    """95% confidence ellipse of the position belief."""
    w, U = np.linalg.eigh(sigma[:2, :2])
    t = np.linspace(0, 2 * np.pi, n)
    return (U @ np.diag(np.sqrt(np.maximum(w, 0) * _CHI2_95)) @ np.vstack([np.cos(t), np.sin(t)])).T + mu[:2]


class PathPlot:
    """Planned path (vertices and segment lengths in mm) plus the live estimate, 95% ellipse and robot outline.

    Call show() to display it in the notebook, then update() whenever it should be redrawn."""

    def __init__(self, vertices, show_truth=False):
        self.fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=90)
        self.ax = ax
        V = np.asarray(vertices, dtype=float)

        ax.plot(V[:, 0], V[:, 1], "--", color="0.55", lw=1.5, label="planned path", zorder=1)
        ax.scatter(V[:, 0], V[:, 1], s=40, color="0.25", zorder=3)
        for i, (x, y) in enumerate(V):
            ax.annotate(f"V{i} ({x:.0f}, {y:.0f})", (x, y), xytext=(7, 7), textcoords="offset points",
                        fontsize=9, color="0.2")
        for a, b in zip(V[:-1], V[1:]):
            d = b - a
            L = np.linalg.norm(d)
            if L == 0:
                continue
            n = np.array([-d[1], d[0]]) / L
            ang = math.degrees(math.atan2(d[1], d[0]))
            if ang > 90 or ang < -90:
                ang -= 180 * np.sign(ang)
            ax.annotate(f"{L:.0f} mm", (a + b) / 2, xytext=tuple(14 * n), textcoords="offset points",
                        ha="center", va="center", rotation=ang, rotation_mode="anchor", fontsize=9,
                        color="#1f5fbf")

        (self.est_line,) = ax.plot([], [], "-", color="#1f77b4", lw=2, label="estimate (motion model)", zorder=4)
        (self.ell_line,) = ax.plot([], [], "-", color="#1f77b4", lw=1, alpha=0.6, label="95% ellipse", zorder=4)
        self.body = Polygon(_BODY, closed=True, fc="#1f77b4", alpha=0.15, ec="#1f77b4", zorder=2)
        ax.add_patch(self.body)
        (self.head_line,) = ax.plot([], [], "-", color="#0b3d75", lw=3, solid_capstyle="round", zorder=5)
        self.truth_line = None
        if show_truth:
            (self.truth_line,) = ax.plot([], [], "-", color="#d62728", lw=1, alpha=0.8, label="sim truth", zorder=3)
        self.text = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", family="monospace",
                            fontsize=8, bbox=dict(fc="white", ec="0.8", alpha=0.85))

        m = 120
        self.lims = [V[:, 0].min() - m, V[:, 0].max() + m, V[:, 1].min() - m, V[:, 1].max() + m]
        self._apply_lims()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Vertex following: motion model only")
        self.est, self.truth = [], []
        self.handle = None
        plt.close(self.fig)  # shown through show(), not at the end of the cell

    def _apply_lims(self):
        self.ax.set_xlim(self.lims[0], self.lims[1])
        self.ax.set_ylim(self.lims[2], self.lims[3])

    def show(self):
        self.handle = display(self.fig, display_id=True)

    def update(self, mu, sigma, truth=None, t=None, phase="", target_idx=None, target=None, heading_ref=None,
               u_cmd=None, wheels=None, status=None):
        """Redraw with the current belief. The status box is built from the keyword arguments
        unless `status` is given."""
        self.est.append(mu[:2].copy())
        E = np.asarray(self.est)
        self.est_line.set_data(E[:, 0], E[:, 1])

        c, s = np.cos(mu[2]), np.sin(mu[2])
        self.body.set_xy(_BODY @ np.array([[c, s], [-s, c]]) + mu[:2])
        self.head_line.set_data([mu[0], mu[0] + 80 * c], [mu[1], mu[1] + 80 * s])
        pts = ellipse_points(mu, sigma)
        self.ell_line.set_data(pts[:, 0], pts[:, 1])

        if truth is not None and self.truth_line is not None:
            self.truth.append(truth[:2].copy())
            T = np.asarray(self.truth)
            self.truth_line.set_data(T[:, 0], T[:, 1])

        x0, x1, y0, y1 = self.lims  # grow the view if the robot leaves it
        if not (x0 + 60 < mu[0] < x1 - 60 and y0 + 60 < mu[1] < y1 - 60):
            self.lims = [min(x0, mu[0] - 150), max(x1, mu[0] + 150), min(y0, mu[1] - 150), max(y1, mu[1] + 150)]
            self._apply_lims()

        if status is None:
            status = self._status(mu, sigma, t, phase, target_idx, target, heading_ref, u_cmd, wheels)
        self.text.set_text(status)
        if self.handle is not None:
            self.handle.update(self.fig)

    def finish(self, mu, sigma, outcome, truth=None):
        """Final redraw, with the outcome at the top of the status box."""
        self.update(mu, sigma, truth=truth, status=f"{outcome}\n" + self._status(mu, sigma, *[None] * 7))

    @staticmethod
    def _status(mu, sigma, t, phase, target_idx, target, heading_ref, u_cmd, wheels):
        sd = np.sqrt(np.diag(sigma))
        lines = []
        if t is not None:
            head = f"t = {t:5.1f} s   {phase:<6s}"
            if target is not None:
                head += f" -> V{target_idx} ({target[0]:.0f}, {target[1]:.0f})"
            lines.append(head)
        lines.append(f"est    x {mu[0]:7.1f} mm  y {mu[1]:7.1f} mm  th {math.degrees(mu[2]):6.1f} deg")
        lines.append(f"1-sig  x {sd[0]:7.1f} mm  y {sd[1]:7.1f} mm  th {math.degrees(sd[2]):6.1f} deg")
        if target is not None:
            line = f"to target {np.linalg.norm(np.asarray(target) - mu[:2]):6.1f} mm"
            if heading_ref is not None:
                line += f", heading err {math.degrees(wrap(heading_ref - mu[2])):6.1f} deg"
            lines.append(line)
        if u_cmd is not None:
            line = f"cmd v {u_cmd[0]:5.1f} mm/s  w {u_cmd[1]:5.2f} rad/s"
            if wheels is not None:
                line += f"  wheels {np.asarray(wheels).astype(int)}"
            lines.append(line)
        return "\n".join(lines)


def print_summary(outcome, log, reached):
    print(f"== {outcome} after {log[-1]['t'] if log else 0:.1f} s ==")
    for r in reached:
        line = (f"V{r['vertex']} ({r['target'][0]:.0f}, {r['target'][1]:.0f}) at t = {r['t']:5.1f} s: "
                f"est ({r['est'][0]:6.1f}, {r['est'][1]:6.1f}) mm, "
                f"est. error {np.linalg.norm(r['est'] - r['target']):4.1f} mm, "
                f"1-sigma ({r['sd'][0]:.1f}, {r['sd'][1]:.1f}) mm")
        if r.get("truth") is not None:
            line += f", sim true error {np.linalg.norm(r['truth'][:2] - r['target']):5.1f} mm"
        print(line)


def plot_timeseries(log, reached):
    """Uncertainty and inputs over time. The dotted lines mark when each vertex was reached."""
    t = np.array([r["t"] for r in log])
    sd = np.array([np.sqrt(np.diag(r["sigma"])) for r in log])
    theta = np.unwrap(np.array([r["mu"][2] for r in log]))
    u_cmd = np.array([r["u_cmd"] for r in log])
    u_meas = np.array([r["u_meas"] for r in log])
    phases = [r.get("phase", "") for r in log]

    fig, axs = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    axs[0].plot(t, sd[:, 0], label="x")
    axs[0].plot(t, sd[:, 1], label="y")
    axs[0].set_ylabel("1-sigma [mm]")
    axs[1].plot(t, np.degrees(sd[:, 2]))
    axs[1].set_ylabel("1-sigma theta [deg]")

    axs[2].plot(t, np.degrees(theta), label="estimate")
    if "heading_ref" in log[0]:
        ref = np.array([r["heading_ref"] for r in log])
        ref = theta + wrap(ref - theta)  # same branch as the unwrapped estimate
        axs[2].plot(t, np.degrees(ref), "--", label="reference")
    axs[2].set_ylabel("theta [deg]")

    axs[3].plot(t, u_meas[:, 0], alpha=0.6, label="measured")
    axs[3].plot(t, u_cmd[:, 0], label="command")
    axs[3].set_ylabel("v [mm/s]")
    for name, color in (("cruise", "#2ca02c"), ("dock", "#ff7f0e")):
        mask = np.array([p == name for p in phases])
        if mask.any():
            axs[3].fill_between(t, 0, 1, where=mask, transform=axs[3].get_xaxis_transform(), color=color,
                                alpha=0.12, lw=0, label=name)
    axs[4].plot(t, u_meas[:, 1], alpha=0.6, label="measured")
    axs[4].plot(t, u_cmd[:, 1], label="command")
    axs[4].set_ylabel("omega [rad/s]")
    axs[4].set_xlabel("t [s]")
    for ax in axs:
        for r in reached:
            ax.axvline(r["t"], color="0.6", ls=":", lw=1)
        ax.grid(True, alpha=0.3)
    for ax in (axs[0], axs[2], axs[3], axs[4]):
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Uncertainty, heading and inputs over time")
    fig.tight_layout()
    plt.show()

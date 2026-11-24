import numpy as np

# rad/s. Below this, lm/predict use the straight-line limit of the model (midpoint heading). The exact
# formulas lose accuracy as |omega| -> 0 (V's rounding error grows like v*eps_machine/omega^2), while the
# limit's error grows with omega*Ts. At 1e-3 the jump at the switch is ~1e-9 in predict and A, and ~3e-5
# (relative) in Q.
OMEGA_EPS = 1e-3


class ProjectThymio:
    def __init__(self, Ts, Q, R, alphas=0.1*np.ones(4)):
        self.Ts = Ts
        self.Q0 = Q
        self.R = R
        self.C = np.eye(3)
        self.alphas = np.ravel(alphas)

    def process_state_input(self, mu, input):
        _, _, theta = mu
        v, omega = input
        Ts = self.Ts

        ctp = np.cos(theta+omega*Ts)
        stp = np.sin(theta+omega*Ts)
        cos_diff = ctp - np.cos(theta)
        sin_diff = stp - np.sin(theta)
        return theta, v, omega, Ts, ctp, stp, cos_diff, sin_diff


    def lm(self, mu, input):
        theta, v, omega, Ts, ctp, stp, cos_diff, sin_diff = self.process_state_input(mu, input)

        if abs(omega) > OMEGA_EPS:
            A = np.eye(3) + np.array([[0, 0, v/omega*(cos_diff)],
                                      [0, 0, v/omega*(sin_diff)],
                                      [0, 0, 0]])
            V = np.array([[sin_diff/omega, -v/(omega**2)*sin_diff + v*Ts/omega*ctp],
                          [-cos_diff/omega, v/(omega**2)*cos_diff + v*Ts/omega*stp],
                          [0, Ts]])
        else:
            # Jacobians of the straight-line limit in predict (midpoint heading)
            cm, sm = np.cos(theta + omega*Ts/2), np.sin(theta + omega*Ts/2)
            A = np.array([[1, 0, -v*Ts*sm],
                          [0, 1, v*Ts*cm],
                          [0, 0, 1]])
            V = np.array([[Ts*cm, -0.5*v*Ts**2*sm],
                          [Ts*sm, 0.5*v*Ts**2*cm],
                          [0, Ts]])
        M = np.diag([self.alphas[0]*v**2 + self.alphas[1]*omega**2,
                     self.alphas[2]*v**2 + self.alphas[3]*omega**2])
        Q = V@M@np.transpose(V)+self.Q0

        return A, self.C, Q, self.R

    def predict(self, mu, input):
        theta, v, omega, Ts, ctp, stp, cos_diff, sin_diff = self.process_state_input(mu, input)
        if abs(omega) > OMEGA_EPS:
            return mu + np.array([v/omega*sin_diff, -v/omega*cos_diff, omega*Ts])
        # Straight-line limit: drive along the midpoint heading (error vs. the exact arc ~ (omega*Ts)^2)
        thm = theta + omega*Ts/2
        return mu + np.array([v*Ts*np.cos(thm), v*Ts*np.sin(thm), omega*Ts])

    def measure(self, mu_pred):
        return self.C@mu_pred


def motion_model(mu, sigma, input, pt):
    A, C, Q, R = pt.lm(mu, input)
    mu_pred = pt.predict(mu, input)
    sigma_pred = A@sigma@np.transpose(A)+Q
    return mu_pred, sigma_pred

def kalman(mu, sigma, input, meas, pt):
    A, C, Q, R = pt.lm(mu, input)

    mu_pred = pt.predict(mu, input)
    sigma_pred = A@sigma@np.transpose(A)+Q
    innovation = meas - pt.measure(mu_pred)
    S = C@sigma_pred@np.transpose(C)+R
    K = sigma_pred@np.transpose(C)@np.linalg.inv(S)

    mu_upd = mu_pred + K@innovation
    sigma_upd = (np.eye(np.shape(sigma)[0])-K@C)@sigma_pred
    return mu_upd, sigma_upd

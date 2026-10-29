"""SOC estimators: Coulomb counting and extended Kalman filters on 1RC / 2RC
Thevenin models.

All estimators consume the *measured* current and voltage and use their own
model parameters (normally identified from a characterisation test, see
:mod:`batterysoc.identify`), not the plant's true parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cell import OcvCurve


def init_soc_from_rest(v_rest: float, ocv: OcvCurve) -> float:
    """Initial SOC from a rested terminal voltage (OCV inversion)."""
    return float(ocv.soc_from_ocv(v_rest))


class CoulombCounter:
    """z[k+1] = z[k] - eta * I[k] dt / (3600 Q)."""

    def __init__(self, z0: float, capacity_ah: float, eta: float = 1.0):
        self.z = z0
        self.q = capacity_ah
        self.eta = eta

    def run(self, current: np.ndarray, dt: float) -> np.ndarray:
        dz = self.eta * np.asarray(current) * dt / (3600.0 * self.q)
        z = self.z - np.concatenate([[0.0], np.cumsum(dz)[:-1]])
        self.z = float(z[-1] - dz[-1])
        return z


@dataclass
class EcmEstimatorParams:
    """Estimator-side ECM: tables over SOC (identified) and capacity."""
    soc_grid: np.ndarray
    r0: np.ndarray
    r_rc: np.ndarray        # (n_rc, n_grid)
    tau_rc: np.ndarray      # (n_rc,)
    capacity_ah: float

    @property
    def n_rc(self) -> int:
        return self.tau_rc.size

    def with_capacity(self, q: float) -> "EcmEstimatorParams":
        return EcmEstimatorParams(self.soc_grid, self.r0, self.r_rc, self.tau_rc, q)


class SocEkf:
    """Extended Kalman filter with state x = [z, v_1, ..., v_n].

    Process model (ZOH):  z' = z - I dt / (3600 Q),  v_i' = a_i v_i + R_i (1 - a_i) I
    Measurement:          V = OCV(z) - R0(z) I - sum v_i
    Because the process Jacobian is diagonal, the covariance prediction is an
    element-wise product, which keeps the per-sample cost low.

    ``current_sigma`` is deliberately larger than the sensor's white noise: it
    also has to cover the unmodelled current offset/gain error and capacity
    error, otherwise the SOC variance collapses and voltage errors end up in
    the RC states instead of correcting SOC.
    """

    def __init__(self, params: EcmEstimatorParams, ocv: OcvCurve, z0: float, dt: float = 1.0,
                 p0_soc: float = 0.05 ** 2, current_sigma: float = 0.2, voltage_sigma: float = 0.01,
                 rc_sigma: float = 1e-4):
        self.p = params
        self.ocv = ocv
        self.dt = dt
        n = 1 + params.n_rc
        self.x = np.zeros(n)
        self.x[0] = z0
        self.P = np.diag([p0_soc] + [1e-4] * params.n_rc)
        self.a = np.exp(-dt / params.tau_rc)
        self.f = np.concatenate([[1.0], self.a])
        sz = current_sigma * dt / (3600.0 * params.capacity_ah)
        self.Q = np.diag([sz ** 2] + [rc_sigma ** 2] * params.n_rc)
        self.R = voltage_sigma ** 2
        self.h = np.empty(n)
        self.h[1:] = -1.0

    def _tables(self, n_table: int = 1001):
        """Resample the SOC-dependent parameter tables onto a uniform grid so
        the per-sample loop can use index arithmetic instead of np.interp."""
        g = np.linspace(0.0, 1.0, n_table)
        r0 = np.interp(g, self.p.soc_grid, self.p.r0)
        rc = [np.interp(g, self.p.soc_grid, row) for row in self.p.r_rc]
        return n_table - 1, r0.tolist(), [r.tolist() for r in rc]

    def run(self, current: np.ndarray, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter a whole log. Returns (SOC estimate at each sample after the
        measurement update with that sample, and its standard deviation).

        The loop works on Python floats and lists (n <= 3 states), which is
        several times faster than small-matrix NumPy calls per sample.
        """
        n_steps = current.size
        z_hist = np.empty(n_steps)
        s_hist = np.empty(n_steps)
        n = self.x.size
        x = self.x.tolist()
        P = self.P.tolist()
        Q = np.diag(self.Q).tolist()
        R = self.R
        f = self.f.tolist()
        a = self.a.tolist()
        nt, r0_tab, rc_tab = self._tables()
        kq = self.dt / (3600.0 * self.p.capacity_ah)
        lookup = self.ocv.lookup
        cur = current.tolist()
        vol = voltage.tolist()
        rng_n = range(n)
        for k in range(n_steps):
            i = cur[k]
            z = x[0]
            zc = 0.0 if z < 0.0 else (1.0 if z > 1.0 else z)
            idx = int(zc * nt)
            if idx >= nt:
                idx = nt - 1
            v_ocv, docv = lookup(z)
            v_pred = v_ocv - r0_tab[idx] * i
            for j in range(1, n):
                v_pred -= x[j]
            h = [docv] + [-1.0] * (n - 1)
            ph = [sum(P[r][c] * h[c] for c in rng_n) for r in rng_n]
            s = sum(h[r] * ph[r] for r in rng_n) + R
            innov = vol[k] - v_pred
            kg = [ph[r] / s for r in rng_n]
            x = [x[r] + kg[r] * innov for r in rng_n]
            P = [[P[r][c] - kg[r] * ph[c] for c in rng_n] for r in rng_n]
            z_hist[k] = x[0]
            s_hist[k] = P[0][0] ** 0.5 if P[0][0] > 0 else 0.0
            # time update (diagonal Jacobian)
            zc = 0.0 if x[0] < 0.0 else (1.0 if x[0] > 1.0 else x[0])
            idx = int(zc * nt)
            if idx >= nt:
                idx = nt - 1
            x[0] -= i * kq
            for j in range(1, n):
                x[j] = a[j - 1] * x[j] + rc_tab[j - 1][idx] * (1.0 - a[j - 1]) * i
            P = [[P[r][c] * f[r] * f[c] + (Q[r] if r == c else 0.0) for c in rng_n] for r in rng_n]
        self.x = np.array(x)
        self.P = np.array(P)
        return z_hist, s_hist

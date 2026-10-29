"""State-of-health estimation from partial-charge features.

Features are computed from the *measured* charge data of each cycle, so no
full discharge is required:

* ``dq_a`` / ``dq_b``: charge accepted during CC charge while the voltage
  climbs through two windows (3.80-3.95 V and 3.95-4.10 V). Both shrink as
  capacity fades.
* ``q_cv``: charge accepted during the CV phase (grows with resistance).
* ``r_dc``: DC resistance from the voltage step at the start of charge.

``GprSohModel`` maps features to SOH (capacity / rated capacity) with a
Gaussian process (ARD RBF + white noise kernel) and returns a predictive
standard deviation. Linear-regression and cycle-count models serve as
baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .cycler import CycleLog

FEATURES = ("dq_a", "dq_b", "q_cv", "r_dc")
WINDOW_A = (3.80, 3.95)
WINDOW_B = (3.95, 4.10)


def _window_charge(i: np.ndarray, v: np.ndarray, lo: float, hi: float, dt: float) -> float:
    """Charge (Ah) between the first crossings of lo and hi, using a running
    median of the voltage to suppress sensor noise."""
    vs = np.convolve(v, np.ones(15) / 15, mode="same")
    a = np.flatnonzero(vs >= lo)
    b = np.flatnonzero(vs >= hi)
    if a.size == 0 or b.size == 0 or b[0] <= a[0]:
        return float("nan")
    return float(-np.sum(i[a[0]:b[0]]) * dt / 3600.0)


def charge_features(log: CycleLog, dt: float = 1.0) -> np.ndarray:
    cc = np.flatnonzero(log.phase == 1)
    cv = np.flatnonzero(log.phase == 2)
    i, v = log.i_meas, log.v_meas
    i_cc, v_cc = i[cc], v[cc]
    dq_a = _window_charge(i_cc, v_cc, *WINDOW_A, dt)
    dq_b = _window_charge(i_cc, v_cc, *WINDOW_B, dt)
    q_cv = float(-np.sum(i[cv]) * dt / 3600.0) if cv.size else 0.0
    k = cc[0]
    v_before = np.mean(v[k - 10:k])
    i_before = np.mean(i[k - 10:k])
    r_dc = float((np.mean(v[k:k + 2]) - v_before) / -(np.mean(i[k:k + 2]) - i_before))
    return np.array([dq_a, dq_b, q_cv, r_dc])


@dataclass
class SohPrediction:
    mean: np.ndarray
    std: np.ndarray


class GprSohModel:
    def __init__(self, seed: int = 0):
        n = len(FEATURES)
        # targets are normalised, so unit signal variance is the natural start
        kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.full(n, 2.0), (1e-1, 1e2))
                  + WhiteKernel(1e-2, (1e-6, 1.0)))
        self.scaler = StandardScaler()
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                                           random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GprSohModel":
        self.gp.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> SohPrediction:
        m, s = self.gp.predict(self.scaler.transform(X), return_std=True)
        return SohPrediction(m, s)


class LinearSohModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr = LinearRegression()

    def fit(self, X, y):
        self.lr.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X) -> SohPrediction:
        return SohPrediction(self.lr.predict(self.scaler.transform(X)), np.zeros(len(X)))


class CycleCountSohModel:
    """Population-average fade curve SOH(n) = c0 + c1 sqrt(n) + c2 n fitted
    on the training cells; ignores the measured data entirely."""

    def fit(self, cycles: np.ndarray, y: np.ndarray):
        A = np.column_stack([np.ones_like(cycles, dtype=float), np.sqrt(cycles), cycles])
        self.coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, cycles: np.ndarray) -> SohPrediction:
        A = np.column_stack([np.ones_like(cycles, dtype=float), np.sqrt(cycles), cycles])
        return SohPrediction(A @ self.coef, np.zeros(len(cycles)))

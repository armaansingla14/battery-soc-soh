"""Lithium-ion cell model: OCV curve, 2RC Thevenin equivalent circuit,
SOC-dependent parameters and cycle-ageing (capacity fade, resistance growth).

Sign convention: current I > 0 is discharge. Terminal voltage

    V = OCV(z) - R0 * I - v1 - v2,      dv_i/dt = -v_i / tau_i + I / C_i

The model is used as the "plant" for simulation. Estimators only get noisy
measurements of I and V and their own (identified) model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import PchipInterpolator

# Representative NMC/graphite open-circuit voltage curve (monotonic).
OCV_SOC = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
OCV_V = np.array([3.00, 3.22, 3.36, 3.47, 3.57, 3.635, 3.685, 3.74, 3.81, 3.89, 3.975, 4.065, 4.12, 4.19])


class OcvCurve:
    """Monotonic OCV(z) with fast table look-ups for OCV, dOCV/dz and the
    inverse SOC(OCV)."""

    def __init__(self, soc=OCV_SOC, volts=OCV_V, n_table: int = 2001):
        self._pchip = PchipInterpolator(soc, volts, extrapolate=True)
        self._dpchip = self._pchip.derivative()
        self.z_grid = np.linspace(0.0, 1.0, n_table)
        self.v_grid = self._pchip(self.z_grid)
        self.dv_grid = self._dpchip(self.z_grid)
        self._n = n_table - 1
        if np.any(np.diff(self.v_grid) <= 0):
            raise ValueError("OCV curve must be strictly increasing")

    def __call__(self, z):
        return self._pchip(np.clip(z, 0.0, 1.0))

    def slope(self, z):
        return self._dpchip(np.clip(z, 0.0, 1.0))

    def soc_from_ocv(self, v):
        return np.interp(v, self.v_grid, self.z_grid)

    def lookup(self, z: float) -> tuple[float, float]:
        """Scalar (OCV, dOCV/dz) by linear table interpolation (fast path for
        per-sample estimator loops)."""
        if z <= 0.0:
            return self.v_grid[0] + self.dv_grid[0] * z, self.dv_grid[0]
        if z >= 1.0:
            return self.v_grid[-1] + self.dv_grid[-1] * (z - 1.0), self.dv_grid[-1]
        f = z * self._n
        i = int(f)
        t = f - i
        v = self.v_grid[i] + t * (self.v_grid[i + 1] - self.v_grid[i])
        d = self.dv_grid[i] + t * (self.dv_grid[i + 1] - self.dv_grid[i])
        return v, d


@dataclass(frozen=True)
class CellParams:
    capacity_ah: float = 3.0
    r0: float = 0.030          # ohmic resistance at mid SOC [ohm]
    r1: float = 0.012
    tau1: float = 12.0         # [s]
    r2: float = 0.018
    tau2: float = 250.0        # [s]
    v_max: float = 4.2
    v_min: float = 3.0

    def r0_at(self, z):
        """Ohmic resistance rises at low SOC."""
        z = np.clip(z, 0.0, 1.0)
        return self.r0 * (1.0 + 0.8 * (1.0 - z) ** 6 + 0.05 * z)

    def r1_at(self, z):
        z = np.clip(z, 0.0, 1.0)
        return self.r1 * (1.0 + 1.5 * np.exp(-12.0 * z))

    def r2_at(self, z):
        z = np.clip(z, 0.0, 1.0)
        return self.r2 * (1.0 + 1.0 * np.exp(-10.0 * z))


@dataclass(frozen=True)
class AgeingParams:
    """Capacity fade Q(n) = Q0 (1 - a sqrt(n) - b n); resistances grow in
    proportion to the lost capacity."""
    a_sqrt: float = 0.004
    b_lin: float = 1.5e-4
    r_growth: float = 2.5      # relative resistance increase per unit fade

    def capacity_fraction(self, n_cycles: float) -> float:
        return float(1.0 - self.a_sqrt * np.sqrt(n_cycles) - self.b_lin * n_cycles)


def aged_params(fresh: CellParams, age: AgeingParams, n_cycles: float) -> CellParams:
    frac = age.capacity_fraction(n_cycles)
    grow = 1.0 + age.r_growth * (1.0 - frac)
    return replace(fresh, capacity_ah=fresh.capacity_ah * frac, r0=fresh.r0 * grow,
                   r1=fresh.r1 * grow, r2=fresh.r2 * grow)


@dataclass
class CellState:
    soc: float
    v1: float = 0.0
    v2: float = 0.0


def simulate_open_loop(p: CellParams, ocv: OcvCurve, state: CellState, current: np.ndarray, dt: float):
    """Vectorised plant simulation for a known current profile.

    RC states use the exact zero-order-hold discretisation with parameters
    evaluated at the SOC at the start of each step (computed from the
    integrated current first). Returns (soc, v1, v2, voltage) arrays where
    element k is the state *after* applying current[k] for dt, and the
    voltage is the terminal voltage during step k.
    """
    i = np.asarray(current, dtype=float)
    dz = i * dt / (3600.0 * p.capacity_ah)
    soc_after = state.soc - np.cumsum(dz)
    soc_before = np.concatenate([[state.soc], soc_after[:-1]])
    a1 = np.exp(-dt / p.tau1)
    a2 = np.exp(-dt / p.tau2)
    b1 = p.r1_at(soc_before) * (1 - a1) * i
    b2 = p.r2_at(soc_before) * (1 - a2) * i
    v1 = _first_order_scan(a1, b1, state.v1)
    v2 = _first_order_scan(a2, b2, state.v2)
    v1_before = np.concatenate([[state.v1], v1[:-1]])
    v2_before = np.concatenate([[state.v2], v2[:-1]])
    volt = ocv(soc_before) - p.r0_at(soc_before) * i - v1_before - v2_before
    return soc_after, v1, v2, volt


def _first_order_scan(a: float, b: np.ndarray, x0: float) -> np.ndarray:
    """x[k] = a x[k-1] + b[k] with x[-1] = x0 (closed form via lfilter)."""
    from scipy.signal import lfilter
    y, _ = lfilter([1.0], [1.0, -a], b, zi=[a * x0])
    return y


def step(p: CellParams, ocv: OcvCurve, s: CellState, current: float, dt: float) -> float:
    """Advance one step in place; returns the terminal voltage during the step."""
    z = s.soc
    v = float(ocv(z)) - float(p.r0_at(z)) * current - s.v1 - s.v2
    a1 = np.exp(-dt / p.tau1)
    a2 = np.exp(-dt / p.tau2)
    s.v1 = a1 * s.v1 + float(p.r1_at(z)) * (1 - a1) * current
    s.v2 = a2 * s.v2 + float(p.r2_at(z)) * (1 - a2) * current
    s.soc = z - current * dt / (3600.0 * p.capacity_ah)
    return v

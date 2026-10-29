"""Battery cycler simulation: CC-CV charge, rests and a dynamic discharge
profile, with a sensor model for the logged current and voltage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cell import CellParams, CellState, OcvCurve, simulate_open_loop, step


@dataclass(frozen=True)
class SensorModel:
    current_sigma: float = 0.02     # A, white noise
    current_bias: float = 0.03      # A, constant offset (reads high)
    current_gain_err: float = 0.005 # relative gain error
    voltage_sigma: float = 0.002    # V
    voltage_lsb: float = 0.001      # V, ADC resolution

    def measure(self, i_true: np.ndarray, v_true: np.ndarray, rng: np.random.Generator):
        i_meas = i_true * (1 + self.current_gain_err) + self.current_bias + rng.normal(0, self.current_sigma, i_true.size)
        v_meas = v_true + rng.normal(0, self.voltage_sigma, v_true.size)
        v_meas = np.round(v_meas / self.voltage_lsb) * self.voltage_lsb
        return i_meas, v_meas


# Calibrated laboratory cycler channel used for characterisation tests.
LAB_SENSORS = SensorModel(current_sigma=0.005, current_bias=0.001, current_gain_err=0.0005,
                          voltage_sigma=0.0005, voltage_lsb=0.0001)


@dataclass(frozen=True)
class CycleProtocol:
    dt: float = 1.0
    charge_c_rate: float = 1.0
    cv_cutoff_c: float = 0.05
    rest_start_s: float = 600.0
    rest_after_charge_s: float = 1200.0
    rest_end_s: float = 1200.0
    discharge_mean_c: float = 1.0


@dataclass
class CycleLog:
    t: np.ndarray
    i_true: np.ndarray
    v_true: np.ndarray
    soc_true: np.ndarray        # SOC at the *start* of each sample
    i_meas: np.ndarray
    v_meas: np.ndarray
    phase: np.ndarray           # 0 rest, 1 CC charge, 2 CV charge, 3 discharge
    capacity_ah: float


def dynamic_discharge_profile(n: int, c_amp: float, mean_c: float, rng: np.random.Generator) -> np.ndarray:
    """Piecewise-constant drive-like current: mostly discharge pulses of
    0.3-2.5 C, some regenerative (charging) pulses and short stops."""
    out = np.empty(n)
    k = 0
    while k < n:
        dur = int(rng.integers(5, 60))
        u = rng.random()
        if u < 0.12:
            level = -rng.uniform(0.2, 0.8)       # regen
        elif u < 0.2:
            level = 0.0                          # stop
        else:
            level = rng.gamma(4.0, mean_c / 4.0 * 1.15)
            level = min(level, 2.5)
        out[k:k + dur] = level * c_amp
        k += dur
    return out


def _append(buf, i, v, z, ph):
    buf["i"].append(i)
    buf["v"].append(v)
    buf["z"].append(z)
    buf["ph"].append(ph)


def run_cycle(p: CellParams, ocv: OcvCurve, state: CellState, proto: CycleProtocol, sensors: SensorModel,
              rng: np.random.Generator) -> CycleLog:
    """Simulate one full cycle starting from ``state`` (modified in place):
    rest -> CC charge -> CV charge -> rest -> dynamic discharge to v_min -> rest."""
    dt = proto.dt
    buf = {"i": [], "v": [], "z": [], "ph": []}

    def open_loop(current, phase):
        z0 = state.soc
        soc, v1, v2, volt = simulate_open_loop(p, ocv, state, current, dt)
        z_start = np.concatenate([[z0], soc[:-1]])
        state.soc, state.v1, state.v2 = float(soc[-1]), float(v1[-1]), float(v2[-1])
        _append(buf, current, volt, z_start, np.full(current.size, phase))
        return soc, v1, v2, volt

    def open_loop_until(current, phase, stop):
        """Run ``current`` but truncate at the first sample where stop(volt)."""
        z0, v10, v20 = state.soc, state.v1, state.v2
        soc, v1, v2, volt = simulate_open_loop(p, ocv, state, current, dt)
        hit = np.flatnonzero(stop(volt))
        n = hit[0] if hit.size else current.size
        n = max(n, 1)
        z_start = np.concatenate([[z0], soc[:-1]])[:n]
        state.soc, state.v1, state.v2 = float(soc[n - 1]), float(v1[n - 1]), float(v2[n - 1])
        _append(buf, current[:n], volt[:n], z_start, np.full(n, phase))
        del v10, v20
        return n

    open_loop(np.zeros(int(proto.rest_start_s / dt)), 0)

    # CC charge (negative current) until the voltage limit.
    i_cc = -proto.charge_c_rate * p.capacity_ah
    max_cc = int(1.3 * 3600 / proto.charge_c_rate / dt)
    open_loop_until(np.full(max_cc, i_cc), 1, lambda v: v >= p.v_max)

    # CV phase: closed loop, current chosen so that V == v_max.
    cutoff = proto.cv_cutoff_c * p.capacity_ah
    cv_i, cv_v, cv_z = [], [], []
    for _ in range(int(3 * 3600 / dt)):
        z = state.soc
        r0 = float(p.r0_at(z))
        i = (float(ocv(z)) - state.v1 - state.v2 - p.v_max) / r0
        i = max(i, i_cc)
        if -i < cutoff:
            break
        cv_z.append(z)
        cv_v.append(step(p, ocv, state, i, dt))
        cv_i.append(i)
    if cv_i:
        _append(buf, np.array(cv_i), np.array(cv_v), np.array(cv_z), np.full(len(cv_i), 2))

    open_loop(np.zeros(int(proto.rest_after_charge_s / dt)), 0)

    # Dynamic discharge until the lower voltage limit.
    n_max = int(2.5 * 3600 / proto.discharge_mean_c / dt)
    prof = dynamic_discharge_profile(n_max, p.capacity_ah, proto.discharge_mean_c, rng)
    open_loop_until(prof, 3, lambda v: v <= p.v_min)

    open_loop(np.zeros(int(proto.rest_end_s / dt)), 0)

    i_true = np.concatenate(buf["i"])
    v_true = np.concatenate(buf["v"])
    i_meas, v_meas = sensors.measure(i_true, v_true, rng)
    return CycleLog(t=np.arange(i_true.size) * dt, i_true=i_true, v_true=v_true,
                    soc_true=np.concatenate(buf["z"]), i_meas=i_meas, v_meas=v_meas,
                    phase=np.concatenate(buf["ph"]).astype(np.int8), capacity_ah=p.capacity_ah)

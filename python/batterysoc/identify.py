"""Equivalent-circuit parameter identification from a simulated HPPC-style
characterisation test.

Test (on a fresh cell, measured through the same noisy sensors):
full charge -> rest, then repeated: 10 s 1C pulse, 40 s rest, discharge
10 % SOC at 1C, 1 h rest; until the lower voltage limit is reached.

From each step the following are extracted:
* OCV point: rested voltage at the end of each 1 h rest,
* R0: instantaneous voltage jump at the pulse edges / current step,
* RC branches: least-squares fit of the relaxation after the 10 % discharge
  with one or two exponentials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from .cell import CellParams, CellState, OcvCurve, simulate_open_loop
from .cycler import SensorModel
from .estimators import EcmEstimatorParams


@dataclass
class HppcLog:
    i_meas: np.ndarray
    v_meas: np.ndarray
    i_true: np.ndarray
    v_true: np.ndarray
    marks: list           # per SOC step: dict of sample indices
    dt: float


def run_hppc(p: CellParams, ocv: OcvCurve, sensors: SensorModel, rng: np.random.Generator,
             dt: float = 1.0, rest_s: float = 3600.0) -> HppcLog:
    """Simulate the characterisation test. After the 10 % steps a final C/10
    discharge to the lower voltage limit and a rest add an OCV point close to
    empty."""
    st = CellState(soc=1.0)
    c = p.capacity_ah
    seg_i, seg_v, marks = [], [], []
    pos = 0

    def add(current, stop=None):
        nonlocal pos
        soc, v1, v2, volt = simulate_open_loop(p, ocv, st, current, dt)
        n = current.size
        if stop is not None:
            hit = np.flatnonzero(stop(volt))
            n = max(int(hit[0]), 1) if hit.size else n
        st.soc, st.v1, st.v2 = float(soc[n - 1]), float(v1[n - 1]), float(v2[n - 1])
        seg_i.append(current[:n])
        seg_v.append(volt[:n])
        pos += n

    add(np.zeros(int(rest_s / dt)))
    n_step = int(0.1 * 3600 / dt)
    for _ in range(9):
        m = {"pulse_start": pos}
        add(np.full(int(10 / dt), c))           # 1C pulse
        m["pulse_end"] = pos
        add(np.zeros(int(40 / dt)))
        m["dis_start"] = pos
        add(np.full(n_step, c))                 # 10 % discharge at 1C
        m["dis_end"] = pos
        add(np.zeros(int(rest_s / dt)))
        m["rest_end"] = pos
        marks.append(m)
    add(np.full(int(2 * 3600 / dt), 0.1 * c), stop=lambda v: v <= p.v_min)
    add(np.zeros(int(rest_s / dt)))
    final_rest_end = pos
    i_true = np.concatenate(seg_i)
    v_true = np.concatenate(seg_v)
    i_meas, v_meas = sensors.measure(i_true, v_true, rng)
    log = HppcLog(i_meas, v_meas, i_true, v_true, marks, dt)
    log.final_rest_end = final_rest_end
    return log


def _relax_model_1(t, v_inf, a1, tau1):
    return v_inf - a1 * np.exp(-t / tau1)


def _relax_model_2(t, v_inf, a1, tau1, a2, tau2):
    return v_inf - a1 * np.exp(-t / tau1) - a2 * np.exp(-t / tau2)


@dataclass
class IdentifiedModel:
    soc_points: np.ndarray      # SOC at each rest (from counted charge)
    ocv_points: np.ndarray
    r0: np.ndarray
    r_rc: np.ndarray            # (n_rc, n_points)
    tau_rc: np.ndarray
    capacity_ah: float
    fit_rmse_v: float

    def estimator_params(self, capacity_ah: float | None = None) -> EcmEstimatorParams:
        order = np.argsort(self.soc_points)
        return EcmEstimatorParams(self.soc_points[order], self.r0[order], self.r_rc[:, order], self.tau_rc,
                                  capacity_ah or self.capacity_ah)

    def ocv_curve(self) -> OcvCurve:
        """OCV curve from the measured rest points, extended to 0 and 1 SOC
        with the end slopes."""
        z = self.soc_points
        v = self.ocv_points
        order = np.argsort(z)
        z, v = z[order], v[order]
        if z[0] > 0.0:
            slope = (v[1] - v[0]) / (z[1] - z[0])
            v = np.concatenate([[v[0] - slope * z[0]], v])
            z = np.concatenate([[0.0], z])
        if z[-1] < 1.0:
            slope = (v[-1] - v[-2]) / (z[-1] - z[-2])
            v = np.concatenate([v, [v[-1] + slope * (1.0 - z[-1])]])
            z = np.concatenate([z, [1.0]])
        return OcvCurve(z, v)


def identify(log: HppcLog, n_rc: int = 2, nominal_capacity_ah: float | None = None) -> IdentifiedModel:
    dt = log.dt
    i, v = log.i_meas, log.v_meas
    # charge removed (measured current) up to each rest; total capacity is
    # not reached by the test, so the nominal capacity anchors the SOC scale
    q_cum = np.concatenate([[0.0], np.cumsum(i) * dt / 3600.0])
    cap = nominal_capacity_ah if nominal_capacity_ah else float(q_cum[log.marks[-1]["rest_end"]]) / 0.9
    soc_pts, ocv_pts, r0s, rcs, taus, rmses = [], [], [], [], [], []
    first = log.marks[0]["pulse_start"]
    soc_pts.append(1.0 - q_cum[first] / cap)
    ocv_pts.append(float(np.mean(v[first - 30:first])))
    for m in log.marks:
        a, b = m["pulse_start"], m["pulse_end"]
        di_on = np.mean(i[a:a + 1]) - np.mean(i[a - 5:a])
        dv_on = np.mean(v[a - 5:a]) - np.mean(v[a:a + 1])
        di_off = np.mean(i[b - 1:b]) - np.mean(i[b:b + 1])
        dv_off = np.mean(v[b:b + 1]) - np.mean(v[b - 1:b])
        r0 = 0.5 * (dv_on / di_on + dv_off / di_off)

        d0, d1, e = m["dis_start"], m["dis_end"], m["rest_end"]
        i_dis = float(np.mean(i[d0:d1]))
        t_dis = (d1 - d0) * dt
        t = np.arange(e - d1 - 1) * dt
        y = v[d1 + 1:e]
        v_inf0 = float(np.mean(y[-60:]))
        if n_rc == 1:
            p0 = [v_inf0, 0.03, 60.0]
            popt, _ = curve_fit(_relax_model_1, t, y, p0=p0, bounds=([2.5, 0, 1], [4.5, 1, 5000]), maxfev=20000)
            amps, tau = [popt[1]], [popt[2]]
            fit = _relax_model_1(t, *popt)
        else:
            p0 = [v_inf0, 0.02, 10.0, 0.03, 300.0]
            popt, _ = curve_fit(_relax_model_2, t, y, p0=p0,
                                bounds=([2.5, 0, 1, 0, 50], [4.5, 1, 50, 1, 5000]), maxfev=20000)
            amps, tau = [popt[1], popt[3]], [popt[2], popt[4]]
            fit = _relax_model_2(t, *popt)
        rmses.append(float(np.sqrt(np.mean((fit - y) ** 2))))
        # branch voltage at the end of the discharge: v_i = R_i I (1 - exp(-T/tau_i));
        # the first sample of the rest is already 1 dt into the relaxation
        r_rc = [A * np.exp(dt / tt) / (i_dis * (1 - np.exp(-t_dis / tt))) for A, tt in zip(amps, tau)]
        soc_pts.append(1.0 - q_cum[e] / cap)
        ocv_pts.append(float(np.mean(v[e - 30:e])))
        # R0 / RC values belong to the SOC at the pulse
        r0s.append((1.0 - q_cum[a] / cap, r0))
        rcs.append(r_rc)
        taus.append(tau)
    end = getattr(log, "final_rest_end", None)
    if end is not None:
        soc_pts.append(1.0 - q_cum[end] / cap)
        ocv_pts.append(float(np.mean(v[end - 30:end])))
    taus = np.median(np.array(taus), axis=0)
    z_r0 = np.array([z for z, _ in r0s])
    r0_arr = np.array([r for _, r in r0s])
    soc_pts = np.array(soc_pts)
    # interpolate R0 / RC tables onto the OCV SOC points
    order = np.argsort(z_r0)
    r0_on = np.interp(soc_pts, z_r0[order], r0_arr[order])
    rc = np.array(rcs).T  # (n_rc, n_steps)
    rc_on = np.array([np.interp(soc_pts, z_r0[order], row[order]) for row in rc])
    return IdentifiedModel(soc_points=soc_pts, ocv_points=np.array(ocv_pts), r0=r0_on, r_rc=rc_on,
                           tau_rc=np.asarray(taus), capacity_ah=cap, fit_rmse_v=float(np.mean(rmses)))

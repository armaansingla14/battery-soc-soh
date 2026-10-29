"""Simulation of a small fleet of cells cycled to end of test.

Each cell gets its own fresh capacity (+-2 %) and ageing rates. Cycles are
simulated every ``stride`` cycles over the cell's life; the ageing model is a
function of cycle count only, so skipped cycles do not change the logged ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cell import AgeingParams, CellParams, CellState, OcvCurve, aged_params
from .cycler import CycleLog, CycleProtocol, SensorModel, run_cycle
from .soh import charge_features


@dataclass
class CellSpec:
    cell_id: int
    fresh: CellParams
    ageing: AgeingParams


@dataclass
class CellHistory:
    spec: CellSpec
    cycles: list = field(default_factory=list)
    capacity_ah: list = field(default_factory=list)
    features: list = field(default_factory=list)
    logs: dict = field(default_factory=dict)       # cycle -> compact log (eval cycles only)

    @property
    def soh(self) -> np.ndarray:
        return np.array(self.capacity_ah) / 3.0


def make_fleet(n_cells: int, rng: np.random.Generator, rated_ah: float = 3.0) -> list[CellSpec]:
    specs = []
    for k in range(n_cells):
        fresh = CellParams(capacity_ah=rated_ah * rng.uniform(0.98, 1.02), r0=0.030 * rng.uniform(0.9, 1.1))
        age = AgeingParams(a_sqrt=rng.uniform(0.002, 0.006), b_lin=rng.uniform(0.8e-4, 2.5e-4),
                           r_growth=rng.uniform(1.5, 3.0))
        specs.append(CellSpec(k, fresh, age))
    return specs


@dataclass
class CompactLog:
    i_meas: np.ndarray
    v_meas: np.ndarray
    soc_true: np.ndarray
    phase: np.ndarray
    capacity_ah: float


def simulate_cell(spec: CellSpec, ocv: OcvCurve, n_cycles: int, stride: int, sensors: SensorModel,
                  rng: np.random.Generator, keep_logs_every: int = 0,
                  proto: CycleProtocol = CycleProtocol()) -> CellHistory:
    hist = CellHistory(spec)
    state = CellState(soc=0.1)
    for n in range(0, n_cycles + 1, stride):
        p = aged_params(spec.fresh, spec.ageing, n)
        log: CycleLog = run_cycle(p, ocv, state, proto, sensors, rng)
        hist.cycles.append(n)
        hist.capacity_ah.append(p.capacity_ah)
        hist.features.append(charge_features(log, proto.dt))
        if keep_logs_every and (n // stride) % keep_logs_every == 0:
            hist.logs[n] = CompactLog(log.i_meas.astype(np.float32), log.v_meas.astype(np.float32),
                                      log.soc_true.astype(np.float32), log.phase, p.capacity_ah)
    return hist

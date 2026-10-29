#!/usr/bin/env python
"""Characterisation, SOH regression and SOC estimation experiments.

Pipeline (all simulated):
1. HPPC test on a reference fresh cell -> identified 1RC and 2RC models and an
   OCV curve (lab-grade sensors).
2. Fleet of cells cycled for 500 cycles with field-grade sensors (current
   bias, gain error, noise; voltage noise and quantisation).
3. SOH models trained on some cells, evaluated on held-out cells.
4. SOC estimators evaluated on the held-out cells over their whole life.

Writes results/*.csv and results/summary.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from batterysoc import cell, cycler, estimators as E, fleet, identify, soh  # noqa: E402

RESULTS = ROOT / "results"
RATED_AH = 3.0


def write_csv(path: Path, rows: list[dict]):
    keys = list(rows[0])
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v) for v in r.values()) + "\n")


def characterise(ocv, seed, md):
    rng = np.random.default_rng(seed)
    ref = cell.CellParams()
    log = identify.run_hppc(ref, ocv, cycler.LAB_SENSORS, rng)
    models = {n: identify.identify(log, n, nominal_capacity_ah=RATED_AH) for n in (1, 2)}
    m2 = models[2]
    oc = m2.ocv_curve()
    z = np.linspace(0.05, 0.95, 91)
    ocv_err = np.abs(oc(z) - ocv(z))

    # voltage prediction error of the identified models on a dynamic cycle
    lg = cycler.run_cycle(ref, ocv, cell.CellState(soc=0.1), cycler.CycleProtocol(), cycler.LAB_SENSORS, rng)
    rows = []
    for n, m in models.items():
        p = m.estimator_params()
        st_z = float(lg.soc_true[0])
        v_rc = np.zeros(p.n_rc)
        a = np.exp(-1.0 / p.tau_rc)
        v_pred = np.empty(lg.t.size)
        z_k = st_z
        for k in range(lg.t.size):
            i = lg.i_meas[k]
            zc = min(max(z_k, 0.0), 1.0)
            v_pred[k] = float(oc(zc)) - np.interp(zc, p.soc_grid, p.r0) * i - v_rc.sum()
            rr = np.array([np.interp(zc, p.soc_grid, row) for row in p.r_rc])
            v_rc = a * v_rc + rr * (1 - a) * i
            z_k -= i / 3600.0 / p.capacity_ah
        err = v_pred - lg.v_true
        mask = lg.soc_true > 0.1
        rows.append(dict(model=f"{n}RC", tau_s=" / ".join(f"{t:.1f}" for t in p.tau_rc),
                         relax_fit_rmse_mv=1000 * m.fit_rmse_v,
                         cycle_v_rmse_mv=1000 * float(np.sqrt(np.mean(err[mask] ** 2))),
                         cycle_v_max_mv=1000 * float(np.max(np.abs(err[mask])))))
    write_csv(RESULTS / "identification.csv", rows)
    md += ["## Model identification (HPPC on a fresh reference cell, lab sensors)", "",
           f"- True time constants {ref.tau1:.0f} s / {ref.tau2:.0f} s; identified 2RC: "
           f"{m2.tau_rc[0]:.1f} s / {m2.tau_rc[1]:.1f} s.",
           f"- Identified OCV curve vs true, SOC 5-95 %: mean abs error {1000 * ocv_err.mean():.1f} mV, "
           f"max {1000 * ocv_err.max():.1f} mV.", "",
           "Voltage prediction of the identified models on a full dynamic cycle (same cell, SOC > 10 %):", "",
           "| model | time constants (s) | relaxation fit RMSE | cycle voltage RMSE | cycle max error |",
           "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['tau_s']} | {r['relax_fit_rmse_mv']:.2f} mV | {r['cycle_v_rmse_mv']:.1f} mV | "
                  f"{r['cycle_v_max_mv']:.1f} mV |")
    md.append("")
    return models, oc


def simulate_fleet(ocv, args, md):
    rng = np.random.default_rng(args.seed + 1)
    specs = fleet.make_fleet(args.cells, rng, RATED_AH)
    hists = []
    for s in specs:
        t0 = time.time()
        is_test = s.cell_id >= args.cells - args.test_cells
        keep = 2 if is_test else 0    # keep full logs every 10 cycles for test cells
        h = fleet.simulate_cell(s, ocv, args.cycles, args.stride, cycler.SensorModel(), rng, keep_logs_every=keep)
        hists.append(h)
        print(f"cell {s.cell_id}: SOH {h.soh[0]:.3f} -> {h.soh[-1]:.3f}, {len(h.cycles)} cycles logged "
              f"({time.time() - t0:.0f}s)", flush=True)
    n_steps = sum(sum(lg.i_meas.size for lg in h.logs.values()) for h in hists)
    md += ["## Fleet", "",
           f"{args.cells} cells, {args.cycles} cycles each (every {args.stride}th cycle simulated and logged: "
           f"{len(hists[0].cycles)} cycles per cell). Cells {args.cells - args.test_cells}-{args.cells - 1} are "
           "held out for testing.", "",
           "| cell | fresh SOH | SOH at last cycle | role |", "|---|---|---|---|"]
    for h in hists:
        role = "test" if h.spec.cell_id >= args.cells - args.test_cells else "train"
        md.append(f"| {h.spec.cell_id} | {h.soh[0]:.3f} | {h.soh[-1]:.3f} | {role} |")
    md.append("")
    return hists, n_steps


def soh_experiment(hists, args, md):
    n_test = args.test_cells
    train, test = hists[:-n_test], hists[-n_test:]
    Xtr = np.vstack([np.array(h.features) for h in train])
    ytr = np.concatenate([h.soh for h in train])
    ctr = np.concatenate([np.array(h.cycles, float) for h in train])
    Xte = np.vstack([np.array(h.features) for h in test])
    yte = np.concatenate([h.soh for h in test])
    cte = np.concatenate([np.array(h.cycles, float) for h in test])
    gpr = soh.GprSohModel(seed=args.seed).fit(Xtr, ytr)
    preds = {
        "GPR (ARD RBF + white)": gpr.predict(Xte),
        "linear regression": soh.LinearSohModel().fit(Xtr, ytr).predict(Xte),
        "cycle-count fade curve": soh.CycleCountSohModel().fit(ctr, ytr).predict(cte),
    }
    rows = []
    for name, pr in preds.items():
        e = pr.mean - yte
        row = dict(model=name, n_train=len(ytr), n_test=len(yte), mae_pct=100 * np.mean(np.abs(e)),
                   rmse_pct=100 * np.sqrt(np.mean(e ** 2)), max_abs_pct=100 * np.max(np.abs(e)))
        if name.startswith("GPR"):
            row["coverage_95"] = float(np.mean(np.abs(e) <= 1.96 * pr.std))
            row["mean_std_pct"] = 100 * float(np.mean(pr.std))
        rows.append(row)
    write_csv(RESULTS / "soh.csv", [{**r, **{k: r.get(k, "") for k in ("coverage_95", "mean_std_pct")}} for r in rows])
    g = rows[0]
    md += ["## SOH estimation (held-out cells)", "",
           f"Features from each cycle's measured charge data: {', '.join(soh.FEATURES)}. "
           f"Training samples {len(ytr)}, test samples {len(yte)}. Errors in SOH percentage points "
           "(SOH = capacity / 3.0 Ah).", "",
           "| model | MAE | RMSE | max abs error |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['mae_pct']:.2f} | {r['rmse_pct']:.2f} | {r['max_abs_pct']:.2f} |")
    md += ["", f"GPR predictive interval: mean std {g['mean_std_pct']:.2f} points, "
           f"empirical coverage of the 95 % interval {100 * g['coverage_95']:.1f} %.",
           f"Learned kernel: `{gpr.gp.kernel_}`", ""]
    # per-cycle GPR SOH for the test cells, used by the SOC estimators
    soh_hat = {}
    for h in test:
        pr = gpr.predict(np.array(h.features)).mean
        soh_hat[h.spec.cell_id] = dict(zip(h.cycles, pr))
    return soh_hat


def ekf_params(models, n_rc, capacity):
    return models[n_rc].estimator_params(capacity)


def scaled_params(models, n_rc, capacity, r_scale):
    p = models[n_rc].estimator_params(capacity)
    return E.EcmEstimatorParams(p.soc_grid, p.r0 * r_scale, p.r_rc * r_scale, p.tau_rc, capacity)


def run_estimators(lg, oc, models, cap_hat, ekf_kw, r_scale=1.0, z0_offset=0.0, oracle=None):
    """Returns dict name -> SOC estimate array for one cycle log."""
    i = lg.i_meas.astype(float)
    v = lg.v_meas.astype(float)
    z0 = E.init_soc_from_rest(float(v[0]), oc) + z0_offset
    p0 = (0.05 + abs(z0_offset)) ** 2
    out = {
        "Coulomb counting, rated capacity": E.CoulombCounter(z0, RATED_AH).run(i, 1.0),
        "Coulomb counting, GPR capacity": E.CoulombCounter(z0, cap_hat).run(i, 1.0),
        "EKF 1RC, rated capacity": E.SocEkf(ekf_params(models, 1, RATED_AH), oc, z0, p0_soc=p0, **ekf_kw).run(i, v)[0],
        "EKF 2RC, rated capacity": E.SocEkf(ekf_params(models, 2, RATED_AH), oc, z0, p0_soc=p0, **ekf_kw).run(i, v)[0],
        "EKF 2RC, GPR capacity": E.SocEkf(ekf_params(models, 2, cap_hat), oc, z0, p0_soc=p0, **ekf_kw).run(i, v)[0],
        "EKF 2RC, GPR capacity + R scaling": E.SocEkf(scaled_params(models, 2, cap_hat, r_scale), oc, z0,
                                                      p0_soc=p0, **ekf_kw).run(i, v)[0],
    }
    if oracle is not None:
        cap_true, r_true = oracle
        out["EKF 2RC, true capacity + true R growth (reference)"] = E.SocEkf(
            scaled_params(models, 2, cap_true, r_true), oc, z0, p0_soc=p0, **ekf_kw).run(i, v)[0]
    return out


def soc_experiment(hists, soh_hat, oc, models, ekf_kw, args, md):
    test = hists[-args.test_cells:]
    acc: dict[str, list] = {}
    stage_acc: dict[tuple, list] = {}
    init_acc: dict[str, list] = {}
    n_samples = 0
    n_cycles = 0
    for h in test:
        cyc = sorted(h.logs)
        pred = soh_hat[h.spec.cell_id]
        r_base = [f[3] for f in h.features[:3]]
        for j, n in enumerate(cyc):
            lg = h.logs[n]
            # causal capacity estimate: GPR SOH from the previous logged cycle
            prev = [c for c in h.cycles if c < n]
            cap_hat = RATED_AH * pred[max(prev)] if prev else RATED_AH
            # resistance growth from the measured DC resistance feature, relative
            # to the same cell's first three logged cycles (median of the last 5)
            r_hist = [f[3] for c, f in zip(h.cycles, h.features) if c < n]
            r_scale = float(np.median(r_hist[-5:]) / np.median(r_base)) if len(r_hist) >= 3 else 1.0
            aged = cell.aged_params(h.spec.fresh, h.spec.ageing, n)
            oracle = (aged.capacity_ah, aged.r0 / cell.CellParams().r0)
            ests = run_estimators(lg, oc, models, cap_hat, ekf_kw, r_scale, oracle=oracle)
            truth = lg.soc_true.astype(float)
            stage = "0-150" if n <= 150 else ("151-350" if n <= 350 else "351-500")
            for name, z in ests.items():
                e = np.abs(z - truth)
                acc.setdefault(name, []).append(e)
                stage_acc.setdefault((name, stage), []).append(e)
            n_samples += truth.size
            n_cycles += 1
            if j % 5 == 0:
                for name, z in run_estimators(lg, oc, models, cap_hat, ekf_kw, r_scale, z0_offset=0.2, oracle=oracle).items():
                    init_acc.setdefault(name, []).append(np.abs(z - truth))
            print(f"cell {h.spec.cell_id} cycle {n}: MAE % " + " ".join(
                f"{100 * np.mean(acc[k][-1]):.2f}" for k in acc), flush=True)
    rows = []
    for name, errs in acc.items():
        e = np.concatenate(errs)
        rows.append(dict(estimator=name, cycles=n_cycles, samples=e.size, mae_pct=100 * e.mean(),
                         rmse_pct=100 * np.sqrt(np.mean(e ** 2)), max_abs_pct=100 * e.max(),
                         **{f"mae_cycles_{s}": (100 * np.concatenate(stage_acc[(name, s)]).mean()
                                                if (name, s) in stage_acc else float("nan"))
                            for s in ("0-150", "151-350", "351-500")},
                         mae_init_err_20pct=100 * np.concatenate(init_acc[name]).mean()))
    write_csv(RESULTS / "soc.csv", rows)
    md += ["## SOC estimation on held-out cells", "",
           f"{n_cycles} logged cycles (every 10th cycle from 0 to {args.cycles}) on {len(test)} test cells, "
           f"{n_samples} samples at 1 Hz. Each cycle: rest, CC-CV charge, rest, dynamic discharge to 3.0 V, rest. "
           "All estimators start from the SOC read off the identified OCV curve at the first (rested) sample. "
           "Errors are absolute SOC errors in percentage points.", "",
           "| estimator | MAE | RMSE | max | MAE cycles 0-150 | 151-350 | 351-500 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['estimator']} | {r['mae_pct']:.2f} | {r['rmse_pct']:.2f} | {r['max_abs_pct']:.2f} | "
                  f"{r['mae_cycles_0-150']:.2f} | {r['mae_cycles_151-350']:.2f} | {r['mae_cycles_351-500']:.2f} |")
    md += ["", f"With a +20 % initial SOC error (every 5th test cycle, {len(next(iter(init_acc.values())))} cycles):", "",
           "| estimator | MAE |", "|---|---|"]
    for r in rows:
        md.append(f"| {r['estimator']} | {r['mae_init_err_20pct']:.2f} |")
    md.append("")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=8)
    ap.add_argument("--test-cells", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=500)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    ocv = cell.OcvCurve()
    md = ["# Experiment summary", "", "All data is simulated (see README for the cell and sensor models).", ""]
    models, oc = characterise(ocv, args.seed, md)
    hists, _ = simulate_fleet(ocv, args, md)
    soh_hat = soh_experiment(hists, args, md)
    ekf_kw = dict(current_sigma=0.2, rc_sigma=1e-4, voltage_sigma=0.01)
    md += ["EKF noise settings (fixed, chosen on a separate fresh-cell development cycle): "
           f"current_sigma {ekf_kw['current_sigma']} A, rc_sigma {ekf_kw['rc_sigma']} V, "
           f"voltage_sigma {ekf_kw['voltage_sigma']} V.", ""]
    soc_experiment(hists, soh_hat, oc, models, ekf_kw, args, md)
    md.append(f"Total runtime: {time.time() - t0:.0f} s.")
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

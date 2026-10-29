# battery-soc-soh

State-of-charge (SOC) and state-of-health (SOH) estimation for lithium-ion
cells, built around a simulated cell so every estimate can be scored against
ground truth. It covers equivalent-circuit modelling and parameter
identification, Coulomb counting, extended Kalman filters, and Gaussian process
regression for SOH. Everything is in Python (NumPy, SciPy, scikit-learn).

All data in this repository is simulated. No physical cells or cycler
hardware were used.

## Features

- **Cell model** (`cell.py`): 2RC Thevenin equivalent circuit with an
  NMC-like OCV curve (PCHIP), SOC-dependent R0/R1/R2, exact zero-order-hold
  discretisation, and cycle ageing: capacity fade
  `Q(n) = Q0 (1 - a sqrt(n) - b n)` with resistance growth proportional to
  lost capacity.
- **Cycler** (`cycler.py`): rest, CC-CV charge (1C to 4.2 V, cut-off C/20),
  rest, a drive-like dynamic discharge (pulses up to 2.5C plus regenerative
  pulses) to 3.0 V, rest. Field sensor model: current offset +30 mA, gain error
  +0.5 %, 20 mA noise; voltage noise 2 mV with 1 mV quantisation. A separate
  calibrated "lab" sensor model is used for characterisation.
- **Parameter identification** (`identify.py`): HPPC-style test (1C pulses,
  10 % SOC steps, 1 h rests, final C/10 step to empty). R0 from the voltage
  step, RC branches from least-squares fits of the relaxation (1 or 2
  exponentials), OCV curve from the rested voltages.
- **SOC estimators** (`estimators.py`): Coulomb counting; EKF on 1RC or 2RC
  models with state `[SOC, v1, (v2)]`, using the identified parameter tables
  and OCV curve (not the plant's true ones). The per-sample loop uses plain
  floats and table look-ups (about 30 us per sample for 2RC).
- **SOH estimation** (`soh.py`): four features from each cycle's measured
  charge data (charge accepted in two voltage windows during CC, CV-phase
  charge, DC resistance at the start of charge); GPR with an ARD RBF + white
  noise kernel, compared with linear regression and a cycle-count fade curve.
- **Fleet simulation** (`fleet.py`): cells with different initial capacity,
  resistance and ageing rates, cycled to 500 cycles.

## Layout

```
python/batterysoc/
  cell.py        OCV curve, 2RC model, ageing
  cycler.py      CC-CV / dynamic-discharge cycles, sensor models
  identify.py    HPPC test and ECM fitting
  estimators.py  Coulomb counting, EKF (1RC / 2RC)
  soh.py         charge features, GPR / linear / cycle-count SOH models
  fleet.py       multi-cell life simulation
python/tests/    pytest suite
scripts/run_experiments.py   full experiment, writes results/
results/         CSV files and summary.md from the last run
```

## How to run

```bash
python -m pytest -q
python scripts/run_experiments.py      # about 8.5 minutes on a desktop CPU
```

## Results

Produced by `python scripts/run_experiments.py` (defaults: 8 cells, 3 held
out for testing, 500 cycles, seed 0). Full output in `results/summary.md`.

**Model identification** (fresh reference cell, lab sensors). Identified
time constants 12.0 s / 249.9 s (true 12 s / 250 s). OCV curve error over
5-95 % SOC: mean 2.1 mV, max 9.6 mV. Voltage prediction on a full dynamic
cycle (SOC > 10 %):

| model | cycle voltage RMSE | max error |
|---|---|---|
| 1RC | 29.8 mV | 84.1 mV |
| 2RC | 6.1 mV | 33.8 mV |

**Fleet.** 8 cells, 500 cycles each. Every 5th cycle is simulated and logged
(101 cycles per cell). End-of-test SOH ranges from 0.794 to 0.885.

**SOH estimation** on the 3 held-out cells (303 cycles), errors in SOH
percentage points:

| model | MAE | RMSE | max |
|---|---|---|---|
| GPR (ARD RBF + white noise) | 1.65 | 2.16 | 5.08 |
| linear regression, same features | 0.50 | 0.60 | 1.46 |
| cycle-count fade curve | 2.21 | 2.56 | 5.77 |

In this setup the GPR did worse than linear regression. It was trained on
only 5 cells, and the held-out cells' feature trajectories fall partly
outside the training range. The GPR also understates its uncertainty: only
26.7 % of test points fall inside its 95 % interval. With these features the
relationship to SOH is close to linear, and the simpler model extrapolates
better. Both data-driven models beat the cycle-count baseline.

**SOC estimation** on the held-out cells: 153 cycles (every 10th cycle from
0 to 500), 1,744,827 samples at 1 Hz. Absolute SOC error in percentage
points. All estimators start from the SOC read from the identified OCV curve
at the first, rested sample of each cycle. "GPR capacity" uses the SOH
estimate from the previous logged cycle.

| estimator | MAE | RMSE | max | MAE cycles 0-150 | 151-350 | 351-500 |
|---|---|---|---|---|---|---|
| Coulomb counting, rated capacity | 5.91 | 7.26 | 20.64 | 3.24 | 6.15 | 8.48 |
| Coulomb counting, GPR capacity | 1.92 | 2.28 | 4.86 | 1.93 | 1.85 | 1.99 |
| EKF 1RC, rated capacity | 1.53 | 2.03 | 7.98 | 1.54 | 1.52 | 1.52 |
| EKF 2RC, rated capacity | 0.81 | 1.12 | 4.68 | 0.51 | 0.80 | 1.15 |
| EKF 2RC, GPR capacity | 1.74 | 2.44 | 8.66 | 0.66 | 1.86 | 2.74 |
| EKF 2RC, GPR capacity + R scaling | 0.99 | 1.39 | 5.68 | 0.82 | 0.96 | 1.21 |
| EKF 2RC, true capacity + true R (reference) | 0.46 | 0.58 | 1.96 | 0.45 | 0.46 | 0.46 |

With a +20 % initial SOC error (33 cycles), Coulomb counting stays wrong
(14.2 % MAE with rated capacity), while the 2RC EKF converges (0.89 % MAE with
rated capacity).

Observations:

- Over the 500 cycles, the 2RC EKF reduces SOC MAE from 5.91 % (Coulomb
  counting with rated capacity) to 0.81 %. Coulomb counting gets worse as the
  cell ages, because the capacity is wrong and the current offset
  accumulates.
- Updating only the capacity in the EKF made it *worse* (1.74 %). With fresh
  resistances the model overestimates the terminal voltage under load. Keeping
  the rated (too large) capacity partly cancels that bias. Scaling the
  resistances by the measured DC-resistance growth recovers most of the
  accuracy (0.99 %). The reference row, with the true capacity and resistance,
  shows what a correctly adapted model reaches (0.46 %). Adapting resistance
  along with capacity matters more than the capacity estimate alone.
- The 2RC model matters: on the same data the 1RC EKF has about twice the
  error.

## Limitations

- Simulated cells only. The plant is itself a 2RC ECM, so the
  estimator/plant mismatch comes only from identification error, parameter
  spread between cells and ageing. Real cells add hysteresis, temperature
  dependence, and diffusion dynamics that a 2RC model does not capture.
- Isothermal (25 C assumed); no temperature model.
- Ageing is a function of cycle count only; there is no calendar ageing,
  knee, or path dependence. Only every 5th cycle is simulated.
- EKF noise settings were chosen by hand on one fresh-cell development cycle
  and not tuned per variant.
- Initial SOC comes from a rested voltage at the start of each cycle (a
  standard BMS reset), so these numbers do not show long-term drift without
  rests.

## License

MIT, see `LICENSE`.

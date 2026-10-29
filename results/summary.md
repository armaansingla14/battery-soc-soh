# Experiment summary

All data is simulated (see README for the cell and sensor models).

## Model identification (HPPC on a fresh reference cell, lab sensors)

- True time constants 12 s / 250 s; identified 2RC: 12.0 s / 249.9 s.
- Identified OCV curve vs true, SOC 5-95 %: mean abs error 2.1 mV, max 9.6 mV.

Voltage prediction of the identified models on a full dynamic cycle (same cell, SOC > 10 %):

| model | time constants (s) | relaxation fit RMSE | cycle voltage RMSE | cycle max error |
|---|---|---|---|---|
| 1RC | 212.9 | 1.34 mV | 29.8 mV | 84.1 mV |
| 2RC | 12.0 / 249.9 | 0.50 mV | 6.1 mV | 33.8 mV |

## Fleet

8 cells, 500 cycles each (every 5th cycle simulated and logged: 101 cycles per cell). Cells 5-7 are held out for testing.

| cell | fresh SOH | SOH at last cycle | role |
|---|---|---|---|
| 0 | 1.000 | 0.822 | train |
| 1 | 0.997 | 0.829 | train |
| 2 | 1.010 | 0.827 | train |
| 3 | 0.998 | 0.860 | train |
| 4 | 1.010 | 0.796 | train |
| 5 | 1.009 | 0.885 | test |
| 6 | 1.001 | 0.794 | test |
| 7 | 1.017 | 0.843 | test |

## SOH estimation (held-out cells)

Features from each cycle's measured charge data: dq_a, dq_b, q_cv, r_dc. Training samples 505, test samples 303. Errors in SOH percentage points (SOH = capacity / 3.0 Ah).

| model | MAE | RMSE | max abs error |
|---|---|---|---|
| GPR (ARD RBF + white) | 1.65 | 2.16 | 5.08 |
| linear regression | 0.50 | 0.60 | 1.46 |
| cycle-count fade curve | 2.21 | 2.56 | 5.77 |

GPR predictive interval: mean std 0.59 points, empirical coverage of the 95 % interval 26.7 %.
Learned kernel: `1.42**2 * RBF(length_scale=[3.94, 2.07, 1.15, 6.07]) + WhiteKernel(noise_level=0.00355)`

EKF noise settings (fixed, chosen on a separate fresh-cell development cycle): current_sigma 0.2 A, rc_sigma 0.0001 V, voltage_sigma 0.01 V.

## SOC estimation on held-out cells

153 logged cycles (every 10th cycle from 0 to 500) on 3 test cells, 1744827 samples at 1 Hz. Each cycle: rest, CC-CV charge, rest, dynamic discharge to 3.0 V, rest. All estimators start from the SOC read off the identified OCV curve at the first (rested) sample. Errors are absolute SOC errors in percentage points.

| estimator | MAE | RMSE | max | MAE cycles 0-150 | 151-350 | 351-500 |
|---|---|---|---|---|---|---|
| Coulomb counting, rated capacity | 5.91 | 7.26 | 20.64 | 3.24 | 6.15 | 8.48 |
| Coulomb counting, GPR capacity | 1.92 | 2.28 | 4.86 | 1.93 | 1.85 | 1.99 |
| EKF 1RC, rated capacity | 1.53 | 2.03 | 7.98 | 1.54 | 1.52 | 1.52 |
| EKF 2RC, rated capacity | 0.81 | 1.12 | 4.68 | 0.51 | 0.80 | 1.15 |
| EKF 2RC, GPR capacity | 1.74 | 2.44 | 8.66 | 0.66 | 1.86 | 2.74 |
| EKF 2RC, GPR capacity + R scaling | 0.99 | 1.39 | 5.68 | 0.82 | 0.96 | 1.21 |
| EKF 2RC, true capacity + true R growth (reference) | 0.46 | 0.58 | 1.96 | 0.45 | 0.46 | 0.46 |

With a +20 % initial SOC error (every 5th test cycle, 33 cycles):

| estimator | MAE |
|---|---|
| Coulomb counting, rated capacity | 14.16 |
| Coulomb counting, GPR capacity | 18.57 |
| EKF 1RC, rated capacity | 1.58 |
| EKF 2RC, rated capacity | 0.89 |
| EKF 2RC, GPR capacity | 1.76 |
| EKF 2RC, GPR capacity + R scaling | 1.16 |
| EKF 2RC, true capacity + true R growth (reference) | 0.55 |

Total runtime: 504 s.

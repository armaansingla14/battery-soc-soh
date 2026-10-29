import numpy as np
import pytest

from batterysoc import cell, cycler, estimators as E, fleet, identify, soh


@pytest.fixture(scope="module")
def ocv():
    return cell.OcvCurve()


@pytest.fixture(scope="module")
def ident(ocv):
    log = identify.run_hppc(cell.CellParams(), ocv, cycler.LAB_SENSORS, np.random.default_rng(0))
    return identify.identify(log, 2, nominal_capacity_ah=3.0)


def test_identification_recovers_parameters(ident, ocv):
    p = cell.CellParams()
    assert ident.tau_rc[0] == pytest.approx(p.tau1, rel=0.1)
    assert ident.tau_rc[1] == pytest.approx(p.tau2, rel=0.1)
    mid = (ident.soc_points > 0.3) & (ident.soc_points < 0.9)
    np.testing.assert_allclose(ident.r0[mid], p.r0_at(ident.soc_points[mid]), rtol=0.08)
    np.testing.assert_allclose(ident.r_rc[1][mid], p.r2_at(ident.soc_points[mid]), rtol=0.1)
    z = np.linspace(0.1, 0.95, 30)
    assert np.max(np.abs(ident.ocv_curve()(z) - ocv(z))) < 0.012


def test_coulomb_counter_exact_with_ideal_current():
    cc = E.CoulombCounter(0.9, 3.0)
    z = cc.run(np.full(3600, 1.5), 1.0)
    assert z[0] == 0.9
    assert cc.z == pytest.approx(0.4)


@pytest.fixture(scope="module")
def cycle_log(ocv):
    p = cell.CellParams()
    return cycler.run_cycle(p, ocv, cell.CellState(soc=0.1), cycler.CycleProtocol(), cycler.SensorModel(),
                            np.random.default_rng(5))


def test_ekf_tracks_soc(ident, cycle_log):
    oc = ident.ocv_curve()
    z0 = E.init_soc_from_rest(float(cycle_log.v_meas[0]), oc)
    z, sd = E.SocEkf(ident.estimator_params(3.0), oc, z0).run(cycle_log.i_meas, cycle_log.v_meas)
    err = np.abs(z - cycle_log.soc_true)
    assert np.mean(err) < 0.02
    assert np.all(sd > 0)


def test_ekf_converges_from_wrong_initial_soc(ident, cycle_log):
    oc = ident.ocv_curve()
    z0 = float(cycle_log.soc_true[0]) + 0.25
    z, _ = E.SocEkf(ident.estimator_params(3.0), oc, z0, p0_soc=0.3 ** 2).run(cycle_log.i_meas, cycle_log.v_meas)
    err = np.abs(z - cycle_log.soc_true)
    assert err[0] > 0.1
    assert np.mean(err[3000:]) < 0.03
    cc = E.CoulombCounter(z0, 3.0).run(cycle_log.i_meas, 1.0)
    assert np.mean(np.abs(cc[3000:] - cycle_log.soc_true[3000:])) > 0.2


def test_charge_features_respond_to_ageing(ocv):
    rng = np.random.default_rng(3)
    spec = fleet.make_fleet(1, rng)[0]
    h = fleet.simulate_cell(spec, ocv, 400, 200, cycler.SensorModel(), rng)
    f = np.array(h.features)
    assert f.shape == (3, len(soh.FEATURES))
    assert np.all(np.isfinite(f))
    assert f[-1, 0] < f[0, 0]        # less charge in the voltage window
    assert f[-1, 3] > f[0, 3]        # higher DC resistance
    assert h.soh[-1] < h.soh[0]


def test_gpr_learns_synthetic_mapping():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(150, 4))
    y = 0.9 + 0.03 * X[:, 0] - 0.01 * X[:, 1] ** 2 + rng.normal(0, 0.002, 150)
    m = soh.GprSohModel().fit(X[:100], y[:100])
    pr = m.predict(X[100:])
    assert np.mean(np.abs(pr.mean - y[100:])) < 0.01
    assert np.all(pr.std > 0)
    lin = soh.LinearSohModel().fit(X[:100], y[:100]).predict(X[100:])
    assert np.mean(np.abs(lin.mean - y[100:])) > np.mean(np.abs(pr.mean - y[100:]))


def test_cycle_count_model():
    n = np.arange(0, 500, 10).astype(float)
    y = 1 - 0.004 * np.sqrt(n) - 1e-4 * n
    m = soh.CycleCountSohModel().fit(n, y)
    np.testing.assert_allclose(m.predict(n).mean, y, atol=1e-9)

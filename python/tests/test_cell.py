import numpy as np
import pytest

from batterysoc import cell, cycler


@pytest.fixture(scope="module")
def ocv():
    return cell.OcvCurve()


def test_ocv_monotonic_and_inverse(ocv):
    z = np.linspace(0, 1, 101)
    v = ocv(z)
    assert np.all(np.diff(v) > 0)
    np.testing.assert_allclose(ocv.soc_from_ocv(v), z, atol=1e-3)
    for zz in (0.0, 0.123, 0.5, 0.999):
        vv, dd = ocv.lookup(zz)
        assert vv == pytest.approx(float(ocv(zz)), abs=1e-4)
        assert dd == pytest.approx(float(ocv.slope(zz)), rel=0.05, abs=0.01)


def test_ocv_rejects_non_monotonic():
    with pytest.raises(ValueError):
        cell.OcvCurve(np.array([0, 0.5, 1.0]), np.array([3.0, 3.9, 3.8]))


def test_coulomb_balance_and_rest_relaxation(ocv):
    p = cell.CellParams()
    st = cell.CellState(soc=0.8)
    i = np.concatenate([np.full(600, 3.0), np.zeros(3000)])
    soc, v1, v2, volt = cell.simulate_open_loop(p, ocv, st, i, 1.0)
    assert soc[-1] == pytest.approx(0.8 - 3.0 * 600 / 3600 / 3.0, abs=1e-12)
    # after a long rest the terminal voltage returns to OCV
    assert volt[-1] == pytest.approx(float(ocv(soc[-1])), abs=2e-3)
    # under load the voltage drop is at least the ohmic drop
    assert float(ocv(0.8)) - volt[1] > 3.0 * p.r0


def test_step_matches_open_loop(ocv):
    p = cell.CellParams()
    rng = np.random.default_rng(0)
    i = rng.uniform(-3, 6, 200)
    s1 = cell.CellState(soc=0.6)
    _, _, _, v_vec = cell.simulate_open_loop(p, ocv, s1, i, 1.0)
    s2 = cell.CellState(soc=0.6)
    v_step = np.array([cell.step(p, ocv, s2, ii, 1.0) for ii in i])
    np.testing.assert_allclose(v_step, v_vec, atol=1e-9)


def test_ageing_reduces_capacity_and_raises_resistance():
    fresh = cell.CellParams()
    age = cell.AgeingParams()
    old = cell.aged_params(fresh, age, 500)
    assert old.capacity_ah < fresh.capacity_ah * 0.85
    assert old.r0 > fresh.r0


def test_cycle_respects_protocol_limits(ocv):
    p = cell.CellParams()
    st = cell.CellState(soc=0.2)
    rng = np.random.default_rng(1)
    log = cycler.run_cycle(p, ocv, st, cycler.CycleProtocol(), cycler.SensorModel(), rng)
    cv = log.phase == 2
    assert np.all(np.abs(log.v_true[cv] - p.v_max) < 1e-6)
    assert log.soc_true.max() > 0.97                   # fully charged
    dis = np.flatnonzero(log.phase == 3)
    assert log.v_true[dis].min() > p.v_min - 0.01        # stops at the cut-off
    assert set(np.unique(log.phase)) == {0, 1, 2, 3}


def test_sensor_model_bias_and_quantisation():
    rng = np.random.default_rng(2)
    s = cycler.SensorModel()
    i, v = s.measure(np.zeros(20000), np.full(20000, 3.7), rng)
    assert np.mean(i) == pytest.approx(s.current_bias, abs=1e-3)
    assert np.allclose(v / s.voltage_lsb, np.round(v / s.voltage_lsb))

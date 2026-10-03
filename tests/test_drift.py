import numpy as np

from safeswap import drift


def _labels(rate, windows, per, rng):
    return [(rng.random(per) < rate).astype(float) for _ in range(windows)]


def test_cusum_rises_under_shift_and_stays_low_without():
    rng = np.random.default_rng(0)
    calm = drift.bernoulli_cusum(_labels(0.1, 40, 5, rng), 0.1, 0.2)
    shifted = drift.bernoulli_cusum(_labels(0.1, 20, 5, rng) + _labels(0.3, 20, 5, rng), 0.1, 0.2)
    assert shifted[-1] > 3 * max(calm.max(), 1)
    assert shifted[:20].max() < shifted[-1]


def test_threshold_calibration_controls_false_alarms():
    rng = np.random.default_rng(1)
    control = [drift.bernoulli_cusum(_labels(0.1, 40, 5, rng), 0.1, 0.2) for _ in range(400)]
    thr = drift.calibrate_threshold(control[:200], 0.05)
    fa = np.mean([drift.first_alarm(r, thr) >= 0 for r in control[200:]])
    assert fa < 0.1


def test_first_alarm_and_mix_z():
    assert drift.first_alarm(np.array([0, 1, 3, 5]), 2) == 2
    assert drift.first_alarm(np.array([0, 1]), 2) == -1
    z = drift.action_mix_z(np.array([150, 300]), np.array([500, 500]), 0.3)
    assert z[0] < 0.1 and z[1] > 10

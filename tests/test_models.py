"""Fast headless checks: models load, policies predict, LQR reproduces vendor gains."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mujoco'))
sys.path.insert(0, str(ROOT / 'lqr'))


def test_mujoco_models_load():
    import mujoco
    for name in ('dp2_nopole.xml', 'dp2_second_order.xml'):
        m = mujoco.MjModel.from_xml_path(str(ROOT / 'mujoco' / name))
        assert m.nu == 2 and m.nq >= 5


def test_first_order_policy_predicts():
    from stable_baselines3 import PPO
    model = PPO.load(ROOT / 'models/first-order/policy.zip', device='cpu')
    with (ROOT / 'models/first-order/vecnormalize.pkl').open('rb') as f:
        norm = pickle.load(f)
    obs = norm.normalize_obs(np.zeros((1, 6), dtype=np.float32))
    act, _ = model.predict(obs, deterministic=True)
    assert act.shape == (1, 2) and np.all(np.isfinite(act))


def test_lqr_reproduces_vendor_gains():
    import lqr_design as lq
    K = lq.solve()
    vendor = np.array([81.2695, -10.0616, -5492.4061, 18921.7098,
                       100.3633, 8.0376, 447.3084, 2962.7738])
    assert np.abs(K[0] - vendor).max() < 1e-3
    poles = np.linalg.eigvals(lq.G - lq.Hd @ K)
    assert np.abs(poles).max() < 1.0


def test_real_trace_metrics():
    import csv
    rows = list(csv.DictReader((ROOT / 'docs/real-trace-100hz.csv').open()))
    t = np.array([int(r['micros']) for r in rows]) / 1e6
    pitch = np.array([int(r['pitch_mrad']) for r in rows]) / 1000.0
    dt = np.diff(t)
    assert 0.009 < np.median(dt) < 0.011          # ~100 Hz
    assert 0.8 < np.rad2deg(np.sqrt(np.mean(pitch ** 2))) < 1.0
    y = pitch[10:] - pitch[10:].mean()
    f = np.fft.rfftfreq(len(y), np.median(dt))
    band = (f > 1) & (f < 40)
    peak = f[band][np.argmax(np.abs(np.fft.rfft(y * np.hanning(len(y))))[band])]
    assert 9.5 < peak < 11.0                       # ~10.2 Hz oscillation

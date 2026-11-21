"""Demo: closed-loop LQR step response on the identified 8-state model.

Shows (a) gain matrix vs vendor MATLAB values, (b) pole map, (c) recovery from a
2 degree tilt. Pure linear-model simulation — no hardware involved.
Writes media/demo_lqr_step.png and results/demo/lqr_step.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lqr'))
import lqr_design as lq  # noqa: E402

VENDOR_K1 = np.array([81.2695, -10.0616, -5492.4061, 18921.7098,
                      100.3633, 8.0376, 447.3084, 2962.7738])


def main():
    G, H, K = lq.G, lq.Hd, lq.solve()

    # 2 deg initial body tilt, zero pole error
    x = np.zeros(8)
    x[2] = np.deg2rad(2.0)
    hist = []
    for _ in range(500):
        u = -(K @ x)
        x = G @ x + H @ u
        hist.append((np.rad2deg(x[2]), np.rad2deg(x[3]), x[0]))
    hist = np.asarray(hist)
    poles = np.linalg.eigvals(G - H @ K)

    out = ROOT / 'results/demo/lqr_step.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        gain_max_dev_vs_vendor=float(np.abs(K[0] - VENDOR_K1).max()),
        pole_magnitudes=sorted(float(abs(p)) for p in poles),
        tilt_peak_deg=float(hist[:, 0].max()),
        tilt_at_0p5s_deg=float(abs(hist[50, 0])),
        tilt_at_5s_deg=float(abs(hist[-1, 0])),
    ), indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(np.arange(len(hist)) * 0.01, hist[:, 0], label='body tilt θ1')
    ax[0].plot(np.arange(len(hist)) * 0.01, hist[:, 1], label='pole angle θ2')
    ax[0].set_xlabel('time (s)'); ax[0].set_ylabel('angle (deg)')
    ax[0].set_title('LQR recovery from 2° tilt'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(poles.real, poles.imag, 'o')
    ax[1].add_patch(plt.Circle((0, 0), 1, fill=False, ls='--', color='grey'))
    ax[1].set_aspect('equal'); ax[1].set_xlim(-1.1, 1.1); ax[1].set_ylim(-1.1, 1.1)
    ax[1].set_xlabel('Re'); ax[1].set_ylabel('Im'); ax[1].set_title('closed-loop poles (z-plane)')
    ax[1].grid(alpha=.3)
    media = ROOT / 'media/demo_lqr_step.png'
    fig.tight_layout(); fig.savefig(media, dpi=150)
    print('max|ΔK| vs vendor = %.2e' % float(np.abs(K[0] - VENDOR_K1).max()))
    print('peak tilt %.2f°, 0.5 s %.3f°, 5 s %.4f°' % (hist[:,0].max(), abs(hist[50,0]), abs(hist[-1,0])))
    print('saved', media.relative_to(ROOT))


if __name__ == '__main__':
    main()

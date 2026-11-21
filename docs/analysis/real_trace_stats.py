"""复算 docs/real-trace-100hz.csv 的指标（周期、倾角 RMS、主频、PWM、行程）。"""
import csv
import numpy as np
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'real-trace-100hz.csv'
rows = list(csv.DictReader(p.open()))
t = np.array([int(r['micros']) for r in rows]) / 1e6
pitch = np.array([int(r['pitch_mrad']) for r in rows]) / 1000.0
pwm = np.array([[int(r['pwm_left']), int(r['pwm_right'])] for r in rows], float)
dt = np.diff(t)

y = pitch[10:] - pitch[10:].mean()
freqs = np.fft.rfftfreq(len(y), np.median(dt))
spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
band = (freqs > 1) & (freqs < 40)

print(f'样本 {len(rows)}，中位周期 {np.median(dt)*1000:.2f} ms')
print(f'倾角 RMS {np.rad2deg(np.sqrt(np.mean(pitch**2))):.2f}°')
print(f'倾角主频 {freqs[band][np.argmax(spec[band])]:.2f} Hz')
print(f'PWM RMS {np.sqrt(np.mean(pwm[:,0]**2)):.0f} / {np.sqrt(np.mean(pwm[:,1]**2)):.0f}')
print(f'相邻步跳变 RMS {np.sqrt(np.mean(np.diff(pwm, axis=0)**2)):.0f}')
print(f'末帧行程 {int(rows[-1]["distance_mm"])} mm')

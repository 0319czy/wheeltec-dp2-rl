#!/usr/bin/env python3
"""Capture and plot one complete N10 revolution from the DP2 USB port."""
import argparse
import csv
import math
from pathlib import Path
import time

import matplotlib
import serial


PACKET_SIZE = 58
PACKETS = 64
MAGIC = b"\x00N10PC\x01\x40"
PAYLOAD_SIZE = PACKET_SIZE * PACKETS
POINTS_PER_PACKET = 16


def capture(port: str) -> bytes:
    link = serial.Serial(port=None, baudrate=115200, timeout=0.2, write_timeout=1)
    link.port = port
    link.rts = False
    link.dtr = True
    link.open()
    with link:
        link.reset_input_buffer()
        link.write(b"?")
        data = bytearray()
        until = time.monotonic() + 16
        while time.monotonic() < until:
            data.extend(link.read(4096))
            start = data.find(MAGIC)
            if start >= 0:
                start += len(MAGIC)
                if len(data) >= start + PAYLOAD_SIZE:
                    return bytes(data[start:start + PAYLOAD_SIZE])
            elif len(data) > 16384:
                del data[:-len(MAGIC)]
    raise TimeoutError(f"No point-cloud response from {port}")


def decode_scan(raw: bytes):
    if not raw or len(raw) > PAYLOAD_SIZE or len(raw) % PACKET_SIZE:
        raise ValueError("Unexpected snapshot length")
    points = []
    for offset in range(0, len(raw), PACKET_SIZE):
        frame = raw[offset:offset + PACKET_SIZE]
        if frame[:3] != b"\xa5\x5a\x3a" or (sum(frame[:-1]) & 255) != frame[-1]:
            raise ValueError(f"Invalid N10 packet at offset {offset}")
        start = int.from_bytes(frame[5:7], "big") / 100
        end = int.from_bytes(frame[55:57], "big") / 100
        step = (end - start) % 360 / (POINTS_PER_PACKET - 1)
        for j in range(POINTS_PER_PACKET):
            index = 7 + 3 * j
            distance = int.from_bytes(frame[index:index + 2], "big")
            angle = (start + j * step) % 360
            points.append((angle, distance, frame[index + 2]))
    wraps = [i for i in range(1, len(points)) if points[i - 1][0] - points[i][0] > 180]
    if len(wraps) < 2:
        raise ValueError("Snapshot does not contain a complete revolution")
    revolution = points[wraps[-2]:wraps[-1]]
    return [(a, d, q) for a, d, q in revolution if 0 < d < 65535], len(revolution)


def save_plot(points, measured, output):
    import matplotlib.pyplot as plt

    rows = []
    for angle, mm, intensity in points:
        distance = mm / 1000
        rad = math.radians(angle)
        rows.append((angle, distance, distance * math.cos(rad), distance * math.sin(rad), intensity))
    if not rows:
        raise ValueError("The scan does not contain valid returns")
    with output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("angle_deg", "distance_m", "x_m", "y_m", "intensity"))
        writer.writerows(rows)

    _, distance, x, y, _ = zip(*rows)
    fig, ax = plt.subplots(figsize=(8.5, 7.3), dpi=160)
    fig.patch.set_facecolor("#0d1523")
    ax.set_facecolor("#111e30")
    ax.grid(color="#31445a", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#7793ae", linewidth=0.8, alpha=0.65)
    ax.axvline(0, color="#7793ae", linewidth=0.8, alpha=0.65)
    dots = ax.scatter(x, y, c=distance, cmap="plasma", s=12, linewidths=0)
    ax.scatter([0], [0], marker="+", c="#ecf5ff", s=125, linewidths=1.8, zorder=5)
    ax.annotate("N10", (0, 0), xytext=(9, 8), textcoords="offset points", color="#ecf5ff")
    ax.set_aspect("equal", adjustable="box")
    extent = max(0.5, max(map(abs, x)), max(map(abs, y))) * 1.07
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
    ax.set_title("N10 360° scan", color="#ecf5ff", fontsize=17, loc="left", pad=18)

    near = ax.inset_axes([0.645, 0.64, 0.32, 0.32])
    near.set_facecolor("#17273b")
    near.scatter(x, y, color="#51d7e9", s=7, linewidths=0)
    near.scatter([0], [0], marker="+", color="#ecf5ff", s=45, linewidths=1.1)
    near.set(xlim=(-1, 1), ylim=(-1, 1), xticks=[-1, 0, 1], yticks=[-1, 0, 1])
    near.set_aspect("equal")
    near.grid(color="#3c526a", linewidth=0.6)
    near.set_axisbelow(True)
    near.set_title("Near view ±1 m", color="#ecf5ff", fontsize=8)
    near.tick_params(colors="#c5d2e0", labelsize=7)
    for spine in near.spines.values():
        spine.set_color("#496177")
    ax.set_xlabel("X (m)", color="#c5d2e0")
    ax.set_ylabel("Y (m)", color="#c5d2e0")
    ax.tick_params(colors="#c5d2e0")
    for spine in ax.spines.values():
        spine.set_color("#496177")
    legend = fig.colorbar(dots, ax=ax, pad=0.025, shrink=0.76)
    legend.set_label("Range (m)", color="#c5d2e0")
    legend.ax.tick_params(colors="#c5d2e0")
    fig.text(0.11, 0.025, f"One revolution  |  {len(rows)} valid returns / {measured} samples",
             color="#aebed0", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.99))
    fig.savefig(output, facecolor=fig.get_facecolor())
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--out", type=Path, required=True, help="PNG output path")
    parser.add_argument("--show", action="store_true", help="Open the matplotlib window")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    raw = capture(args.port)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".bin").write_bytes(raw)
    points, measured = decode_scan(raw)
    save_plot(points, measured, args.out)
    print(f"{len(points)} valid returns from {measured} samples, range "
          f"{min(d for _, d, _ in points) / 1000:.3f}–"
          f"{max(d for _, d, _ in points) / 1000:.3f} m")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()

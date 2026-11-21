#!/usr/bin/env python3
"""Display DP2 N10 scans and record planar lidar odometry."""
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np
import serial
from scipy.spatial import cKDTree

from n10_pointcloud import PACKET_SIZE, decode_scan


FAST_MAGIC = b"\x00N10PC\x02"
FAST_REQUEST = b"\x05"


def point_array(points):
    angle, mm, _ = np.asarray(points, dtype=float).T
    rad = np.deg2rad(angle)
    return np.column_stack((np.cos(rad), np.sin(rad))) * (mm[:, None] / 1000)


def scan_stream(port, stop):
    link = serial.Serial(port=None, baudrate=115200, timeout=0.04, write_timeout=1,
                         exclusive=True)
    link.port = port
    link.rts, link.dtr = False, True
    link.open()
    with link:
        link.reset_input_buffer()
        data = bytearray()
        last_scan = time.monotonic()
        requested_at = -math.inf
        previous = None
        while not stop.is_set():
            now = time.monotonic()
            if now - last_scan > 12:
                raise TimeoutError("No live scan received for 12 seconds")
            if now - requested_at > 2:
                link.write(FAST_REQUEST)
                requested_at = now
            data.extend(link.read(max(1, min(link.in_waiting, 4096))))
            pos = data.find(FAST_MAGIC)
            if pos < 0:
                del data[:-len(FAST_MAGIC)]
                continue
            head = pos + len(FAST_MAGIC)
            if len(data) <= head:
                continue
            count = data[head]
            if not 20 <= count <= 64:
                del data[:head]
                continue
            end = head + 1 + count * PACKET_SIZE
            if len(data) < end:
                continue
            payload = bytes(data[head + 1:end])
            del data[:end]
            try:
                points, measured = decode_scan(payload)
            except ValueError:
                requested_at = -math.inf
                continue
            now = time.monotonic()
            link.write(FAST_REQUEST)
            requested_at = now
            if points and payload != previous:
                previous = payload
                last_scan = now
                yield now, point_array(points), measured


def transform(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array([[c, s], [-s, c]]) + [x, y]


class LidarOdometry:
    """Estimate the current lidar pose in a frame fixed to the first scan."""

    def __init__(self):
        self.anchor = None
        self.anchor_pose = np.zeros(3)
        self.relative_pose = np.zeros(3)
        self.pose = np.zeros(3)

    def set_anchor(self, points):
        self.anchor = points
        self.tree = cKDTree(points)
        _, neighbors = self.tree.query(points, k=7)
        clusters = points[neighbors]
        centered = clusters - clusters.mean(axis=1, keepdims=True)
        eig, axes = np.linalg.eigh(centered.transpose(0, 2, 1) @ centered)
        self.normals = axes[:, :, 0]
        self.use_normal = (eig[:, 0] < 0.24 * eig[:, 1]) & (eig[:, 1] > 1e-6)
        self.anchor_pose = self.pose.copy()
        self.relative_pose[:] = 0

    def update(self, points):
        range_m = np.linalg.norm(points, axis=1)
        points = points[(range_m >= 0.10) & (range_m <= 8.0)]
        if len(points) < 45:
            return self.pose.copy(), False, 0, math.nan
        if self.anchor is None:
            self.set_anchor(points)
            return self.pose.copy(), True, 1, 0

        estimate = self.relative_pose.copy()
        for i in range(18):
            moved = transform(points, estimate)
            dist, index = self.tree.query(moved)
            max_gap = 0.30 if i < 6 else 0.15
            valid = (dist < max_gap) & self.use_normal[index]
            overlap = float(np.mean(valid))
            if np.count_nonzero(valid) < 36 or overlap < 0.35:
                return self.pose.copy(), False, overlap, math.nan
            normals = self.normals[index[valid]]
            delta = moved[valid] - self.anchor[index[valid]]
            residual = np.sum(normals * delta, axis=1)
            relative = moved[valid] - estimate[:2]
            jacobian = np.column_stack((normals, normals[:, 1] * relative[:, 0]
                                         - normals[:, 0] * relative[:, 1]))
            yaw_scale = max(1.0, float(np.sqrt(np.mean(np.sum(relative ** 2, axis=1)))))
            jacobian[:, 2] /= yaw_scale
            weights = np.sqrt(np.minimum(1, 0.025 / np.maximum(abs(residual), 1e-9)))
            step, _, _, singular = np.linalg.lstsq(jacobian * weights[:, None],
                                                  -residual * weights, rcond=None)
            error = float(np.sqrt(np.mean(residual ** 2)))
            if singular[-1] < 0.60 or singular[-1] / singular[0] < 0.017:
                return self.pose.copy(), False, overlap, error
            step[2] /= yaw_scale
            estimate += step
            if i >= 6 and np.linalg.norm(step) < 0.0002:
                break

        if (error > 0.055 or np.linalg.norm(estimate[:2] - self.relative_pose[:2]) > 0.28
                or abs(estimate[2] - self.relative_pose[2]) > math.radians(20)):
            return self.pose.copy(), False, overlap, error

        if np.linalg.norm(estimate[:2] - self.relative_pose[:2]) < 0.004:
            estimate[:2] = self.relative_pose[:2]
        if abs(estimate[2] - self.relative_pose[2]) < math.radians(0.15):
            estimate[2] = self.relative_pose[2]
        self.relative_pose = estimate
        self.pose[:2] = transform(estimate[None, :2], self.anchor_pose)[0]
        self.pose[2] = self.anchor_pose[2] + estimate[2]
        if np.linalg.norm(estimate[:2]) > 0.16 or abs(estimate[2]) > math.radians(9):
            self.set_anchor(points)
        return self.pose.copy(), True, overlap, error


def acquire(args, stop, reset, messages):
    pose_log = args.out / "trajectory.csv"
    odometry = LidarOdometry()
    path = []
    first_time = last_time = None
    index = 0
    segment = 0
    hz = 0.0
    try:
        with pose_log.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("utc", "time_s", "segment", "x_m", "y_m", "yaw_deg", "accepted",
                             "overlap", "rmse_m", "points", "samples"))
            stream.flush()
            for tick, points, measured in scan_stream(args.port, stop):
                if reset.is_set():
                    reset.clear()
                    odometry = LidarOdometry()
                    path.clear()
                    segment += 1
                index += 1
                first_time = tick if first_time is None else first_time
                if last_time is not None:
                    current_hz = 1 / (tick - last_time)
                    hz = current_hz if not hz else 0.30 * current_hz + 0.70 * hz
                last_time = tick
                pose, accepted, overlap, error = odometry.update(points)
                stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                writer.writerow((stamp, f"{tick - first_time:.3f}", segment, f"{pose[0]:.6f}",
                                 f"{pose[1]:.6f}", f"{math.degrees(pose[2]):.4f}",
                                 int(accepted), f"{overlap:.3f}", f"{error:.4f}",
                                 len(points), measured))
                stream.flush()
                if accepted and (not path or np.linalg.norm(pose[:2] - path[-1]) > 0.001):
                    path.append(pose[:2].copy())
                message = (index, tick, points, pose, accepted, np.array(path), hz)
                if messages.full():
                    try:
                        messages.get_nowait()
                    except queue.Empty:
                        pass
                messages.put_nowait(message)
                if args.frames and index >= args.frames:
                    break
    except (OSError, serial.SerialException) as exc:
        messages.put(exc)


def display(args, worker, messages, stop, reset):
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, (lidar, track) = plt.subplots(1, 2, figsize=(13, 6.8), facecolor="#0d1523")
    fig.canvas.manager.set_window_title("N10 Live Point Cloud and Trajectory")
    for axis in (lidar, track):
        axis.set_facecolor("#142237")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.23)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
    lidar.set_title("Live N10 scan", loc="left")
    track.set_title("Estimated trajectory", loc="left")
    lidar.set(xlim=(-args.range, args.range), ylim=(-args.range, args.range))
    lidar.scatter([0], [0], color="white", marker="+", s=90)
    scan = lidar.scatter([], [], s=10, color="#57d6ee")
    line, = track.plot([], [], color="#6de5a5", lw=2)
    direction, = track.plot([], [], color="#ffca71", lw=2)
    status = fig.text(0.08, 0.08, "Waiting for N10 USB frames ...", color="#e4f2ff")
    fig.text(0.08, 0.035, "R: new origin    |    Recording trajectory.csv", color="#a4bbd3", fontsize=9)
    fig.subplots_adjust(bottom=0.18, left=0.08, right=0.96, wspace=0.25)
    last = None
    last_status = None

    def refresh():
        nonlocal last, last_status
        changed = False
        try:
            message = messages.get_nowait()
        except queue.Empty:
            message = None
        if isinstance(message, Exception):
            status.set_text(str(message))
            fig.canvas.draw_idle()
            timer.stop()
            return
        if message is not None:
            index, tick, points, pose, accepted, path, hz = message
            last = message
            scan.set_offsets(points)
            if len(path):
                line.set_data(path[:, 0], path[:, 1])
            x, y, yaw = pose
            direction.set_data([x, x + 0.18 * math.cos(yaw)], [y, y + 0.18 * math.sin(yaw)])
            if len(path):
                low, high = np.minimum(path.min(axis=0), [-0.45, -0.45]), np.maximum(path.max(axis=0), [0.45, 0.45])
                track.set(xlim=(low[0] - 0.2, high[0] + 0.2), ylim=(low[1] - 0.2, high[1] + 0.2))
            changed = True
        if last is not None:
            index, tick, points, pose, accepted, _, hz = last
            age = time.monotonic() - tick
            state = f"Waiting for next scan ({age:.1f} s)" if age > 1.2 else (
                "Tracking" if accepted else "Pose held (scan match rejected)")
            text = (f"{state}   |   {hz:.1f} Hz   |   Scan {index}   |   "
                    f"X {pose[0]:+.3f} m   Y {pose[1]:+.3f} m   Heading {math.degrees(pose[2]):+.1f}°")
            if text != last_status:
                status.set_text(text)
                last_status = text
                changed = True
        if changed:
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key and event.key.lower() == "r":
            reset.set()

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=60)
    timer.add_callback(refresh)
    timer.start()
    fig.canvas.mpl_connect("close_event", lambda _: stop.set())
    worker.start()
    plt.show()
    stop.set()
    worker.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--range", default=3.0, type=float, help="Point cloud half-width in meters")
    parser.add_argument("--frames", default=0, type=int, help="Stop acquisition after N frames")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    messages = queue.Queue(maxsize=1)
    reset = threading.Event()
    worker = threading.Thread(target=acquire, args=(args, stop, reset, messages), daemon=True)
    if args.headless:
        worker.start()
        worker.join()
        last = messages.get_nowait()
        if isinstance(last, Exception):
            raise last
        print(f"Recorded {last[0]} scans in {args.out}")
    else:
        display(args, worker, messages, stop, reset)


if __name__ == "__main__":
    main()

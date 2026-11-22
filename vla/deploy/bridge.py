"""DP2 上位机桥：相机帧 + 指令 → VLA → (v, ω) 速度帧 / 单字符遥控。

默认 --mock：完全不打开串口、不连真车，只在本机跑通
    合成相机帧 → 假 VLA 策略 → 协议编码 → MockRobot 应用 → 遥测回读
的闭环，并打印日志。

真实链路需要同时给 --serial 和 --allow-real（环境变量 DP2_ALLOW_REAL=1），
防止误连真车。

协议见 deploy/PROTOCOL.md，固件改动方案见 deploy/firmware-notes.md。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY / "asr"))

from normalize import INSTRUCTION_TEXTS, normalize  # noqa: E402

WHEEL_RADIUS = 0.0335          # m，来自 assets/lowlevel/dp2_nopole3d.xml
TRACK_HALF = 0.095             # m，轮距 ±0.095
V_LIMIT = 0.35                 # m/s，spec §2
W_LIMIT = 1.0                  # rad/s，spec §2
BUCKET_COLORS = ("红", "蓝", "绿")
RED = (220, 40, 40)
BLUE = (40, 80, 220)
GREEN = (40, 200, 60)

# 假策略（mock）：常量类指令直接用 spec §1.3 的专家动作
MOCK_CONST_ACTIONS = {
    0: (0.0, 0.0),
    1: (0.25, 0.0),
    2: (-0.20, 0.0),
    3: (0.15, 0.8),
    4: (0.15, -0.8),
    5: (0.0, 1.0),
    6: (0.0, -1.0),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MockRobot:
    """二维 mock 车：接收帧、积分位姿、生成遥测；不连接任何硬件。"""

    def __init__(self, seed: int = 0):
        self.pose = [0.0, 0.0, 0.0]            # x, y, yaw
        self.v = 0.0
        self.w = 0.0
        self.time_s = 0.0
        self.rng = np.random.default_rng(seed)
        self.frames_applied = 0
        self.mode = "idle"
        self.last_frame = ""
        self.legacy_v_forward = 0.1675          # movement_speed 0.05 rad / 10 ms
        self.legacy_w_turn = 0.705              # turn_speed 0.02 rad / 10 ms
        # 桶的世界坐标：极坐标 (r, 角度)，spec §1.2 的半径 0.7-1.4 m
        self.buckets = {
            "红": (1.10, math.radians(-28.0)),
            "蓝": (1.00, math.radians(22.0)),
            "绿": (1.20, math.radians(145.0)),
        }

    # ---------- 指令接收 ----------
    def apply_frame(self, frame: str) -> None:
        self.last_frame = frame
        self.frames_applied += 1
        compact = re.search(r"#V(-?\d+),(-?\d+);", frame)
        if compact:
            self.v = int(compact.group(1)) / 1000.0
            self.w = int(compact.group(2)) / 1000.0
            self.mode = "vw"
            return
        param_v = re.search(r"#vc=(-?[0-9.]+);", frame)
        param_w = re.search(r"#wc=(-?[0-9.]+);", frame)
        if param_v or param_w:
            if param_v:
                self.v = float(param_v.group(1))
            if param_w:
                self.w = float(param_w.group(1))
            self.mode = "param"
            return
        if frame and all(char in "AEGCZYXb" for char in frame):
            for char in frame:
                self.apply_legacy(char)
            return

    def apply_legacy(self, char: str) -> None:
        """按固件 Normal() 的语义把单字符翻译成近似 (v, ω)。"""
        self.mode = "legacy"
        if char == "Z":
            self.v = 0.0
            self.w = 0.0
        elif char == "A":
            self.v = self.legacy_v_forward
        elif char == "E":
            self.v = -self.legacy_v_forward
        elif char == "G":
            self.w = self.legacy_w_turn
        elif char == "C":
            self.w = -self.legacy_w_turn

    # ---------- 运动学 ----------
    def step(self, dt: float) -> None:
        self.time_s += dt
        x, y, yaw = self.pose
        x += self.v * math.cos(yaw) * dt
        y += self.v * math.sin(yaw) * dt
        yaw = wrap_angle(yaw + self.w * dt)
        self.pose = [x, y, yaw]

    def telemetry(self) -> dict:
        x, y, yaw = self.pose
        left_mm_s = (self.v - self.w * TRACK_HALF) * 1000.0
        right_mm_s = (self.v + self.w * TRACK_HALF) * 1000.0
        angle = 1.2 + 0.6 * math.sin(self.time_s * 3.0) * (0.2 + abs(self.v))
        return {
            "type": "A",
            "speed_left_mm_s": int(round(left_mm_s)),
            "speed_right_mm_s": int(round(right_mm_s)),
            "battery_percent": 81,
            "angle_deg": round(angle, 2),
            "pose": (round(x, 3), round(y, 3), round(yaw, 3)),
            "t": round(self.time_s, 2),
        }

    def scene(self) -> dict:
        """相对车体的桶方位：{颜色: (方位角 rad, 距离 m)}。"""
        x, y, yaw = self.pose
        result = {}
        for color, (radius, theta) in self.buckets.items():
            bx = radius * math.cos(theta)
            by = radius * math.sin(theta)
            dx = bx - x
            dy = by - y
            dist = math.hypot(dx, dy)
            bearing = wrap_angle(math.atan2(dy, dx) - yaw)
            result[color] = (bearing, dist)
        return result

    # ---------- 合成相机帧 ----------
    def render(self, size: int = 128) -> np.ndarray:
        image = np.zeros((size, size, 3), dtype=np.uint8)
        horizon = int(size * 0.30)
        for row in range(horizon, size):
            shade = int(35 + 55 * (row - horizon) / float(max(1, size - horizon)))
            image[row, :, :] = (shade, shade + 8, shade + 14)
        image[:horizon, :, :] = (12, 14, 22)
        focal = size * 0.9
        for depth in (0.4, 0.7, 1.0, 1.5, 2.2):
            row = horizon + int(focal * 0.09 / depth)
            if horizon <= row < size:
                image[row, :, :] = (90, 95, 105)
        for bearing in (-0.8, -0.4, 0.0, 0.4, 0.8):
            col = int(size / 2.0 + focal * math.tan(bearing) * 0.35)
            if 0 <= col < size:
                image[horizon:, col, :] = (80, 85, 95)
        for color, (bearing, dist) in self.scene().items():
            if dist <= 0.12 or abs(bearing) > 0.62:
                continue
            rgb = {"红": RED, "蓝": BLUE, "绿": GREEN}[color]
            center_u = int(size / 2.0 + focal * math.tan(bearing) * 0.6)
            height = int(clamp(focal * 0.22 / dist, 6, size * 0.55))
            width = int(clamp(focal * 0.14 / dist, 4, size * 0.35))
            base_row = min(size - 1, horizon + int(focal * 0.055 / dist) + height)
            top_row = max(0, base_row - height)
            half_width = int(width / 2)
            left = max(0, center_u - half_width)
            right = min(size, center_u + half_width)
            image[top_row:base_row, left:right, :] = rgb
            rim = max(1, int(width / 5))
            image[top_row:top_row + rim, left:right, :] = tuple(int(c * 0.65) for c in rgb)
        noise = self.rng.integers(-3, 4, size=(size, size, 3), dtype=np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class SyntheticCamera:
    """无相机时的合成相机（mock）。"""

    name = "synthetic"

    def __init__(self, robot: MockRobot, size: int = 128):
        self.robot = robot
        self.size = size

    def read(self):
        return self.robot.render(self.size), self.robot.scene()


class VideoCamera:
    """真实 USB 相机（只在给了 --camera 时使用；本机当前无相机）。"""

    name = "video"

    def __init__(self, index: int = 0, size: int = 128):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("打不开相机 /dev/video%d" % index)
        self.size = size

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("相机读帧失败")
        frame = self.cv2.resize(frame, (self.size, self.size))
        return frame[:, :, ::-1], None

    def close(self) -> None:
        self.cap.release()


class FakeVLAPolicy:
    """假 VLA：常量类指令输出固定 (v, ω)；闭环类用 mock 车给的几何做专家式闭环。"""

    name = "fake"

    def act(self, image, instr_id: int, scene: dict | None = None):
        if instr_id in MOCK_CONST_ACTIONS:
            return MOCK_CONST_ACTIONS[instr_id]
        if scene is None:
            return 0.0, 0.0
        if instr_id == 7:
            bearing, _ = scene["红"]
            return 0.0, clamp(1.5 * bearing, -W_LIMIT, W_LIMIT)
        if instr_id in (8, 9, 10):
            color = {8: "红", 9: "蓝", 10: "绿"}[instr_id]
            bearing, dist = scene[color]
            if dist > 0.95:
                v = 0.32
            else:
                v = clamp(1.0 * (dist - 0.40), 0.0, 0.32)
            return v, clamp(1.5 * bearing, -W_LIMIT, W_LIMIT)
        if instr_id == 11:
            bearing_blue, _ = scene["蓝"]
            bearing_red, dist_red = scene["红"]
            w = clamp(1.5 * bearing_blue, -W_LIMIT, W_LIMIT)
            if dist_red < 0.8:
                cross = math.sin(bearing_blue - bearing_red)
                w += 0.8 * (1.0 if cross >= 0 else -1.0) * (0.8 - dist_red) / 0.8
            return 0.2, clamp(w, -W_LIMIT, W_LIMIT)
        return 0.0, 0.0


class VLAPolicyAdapter:
    """把 vla/infer.py 的 VLAPolicy 适配成 bridge 的 act(image, instr_id, scene)。"""

    def __init__(self, ckpt: str | None = None):
        if not ckpt:
            raise SystemExit("[bridge] --policy vla 需要 --ckpt：vla.infer.VLAPolicy.load() 需要 checkpoint 路径；"
                             "没有 ckpt 时请继续用 --policy fake 做 mock 联调")
        sys.path.insert(0, str(ROOT))
        from vla.infer import VLAPolicy  # type: ignore

        self.policy = VLAPolicy.load(ckpt)
        self.name = "vla"

    def act(self, image, instr_id: int, scene: dict | None = None):
        v, w = self.policy.act(image, INSTRUCTION_TEXTS[instr_id])
        return float(v), float(w)


class MockLink:
    """mock 链路：把帧交给 MockRobot，遥测从 MockRobot 读。"""

    name = "mock"
    mock = True

    def __init__(self, robot: MockRobot):
        self.robot = robot
        self.sent = 0
        self.rx_frames: list[dict] = []

    def send(self, frame: str) -> bool:
        self.robot.apply_frame(frame)
        self.sent += 1
        return True

    def poll(self) -> list[dict]:
        telemetry = self.robot.telemetry()
        self.rx_frames.append(telemetry)
        return [telemetry]

    def close(self) -> None:
        pass


class SerialLink:
    """真实串口 / 蓝牙（需要 --allow-real + DP2_ALLOW_REAL=1）。"""

    name = "serial"
    mock = False

    def __init__(self, port: str, baud: int = 9600):
        import serial

        self.ser = serial.Serial(port, baud, timeout=0.02)
        self.buffer = b""
        self.sent = 0

    def send(self, frame: str) -> bool:
        try:
            self.ser.write(frame.encode("ascii"))
            self.sent += 1
            return True
        except Exception as exc:
            print("[serial] 写失败：%s" % exc)
            return False

    def poll(self) -> list[dict]:
        try:
            chunk = self.ser.read(512)
        except Exception as exc:
            print("[serial] 读失败：%s" % exc)
            return []
        if not chunk:
            return []
        self.buffer += chunk
        frames = []
        while b"$" in self.buffer:
            raw, self.buffer = self.buffer.split(b"$", 1)
            parsed = parse_telemetry(raw.decode("ascii", "ignore"))
            if parsed:
                frames.append(parsed)
        if len(self.buffer) > 4096:
            self.buffer = self.buffer[-512:]
        return frames

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


def parse_telemetry(text: str) -> dict | None:
    """解析固件 $...$ 帧：{A左:右:电量:角度} / {D...} / {E...}。"""
    text = text.strip()
    if len(text) < 3 or not text.startswith("{"):
        return None
    body = text[1:].rstrip("}")
    kind = body[:1]
    values = []
    for part in body[1:].replace(" ", "").split(":"):
        if part == "":
            continue
        try:
            value = float(part)
        except ValueError:
            return None
        if not math.isfinite(value) or abs(value) > 1e9:
            return None
        values.append(value)
    if kind == "A" and len(values) >= 4:
        return {"type": "A", "speed_left_mm_s": int(values[0]),
                "speed_right_mm_s": int(values[1]), "battery_percent": int(values[2]),
                "angle_deg": values[3]}
    if kind == "D" and len(values) >= 5:
        return {"type": "D", "flag_stop": int(values[0]), "voltage_raw": int(values[1]),
                "pwm_left": int(values[2]), "pwm_right": int(values[3]), "app_on": int(values[4])}
    if kind == "E" and len(values) >= 12:
        return {"type": "E", "theta1_deg": values[0] / 100.0, "theta2_deg": values[1] / 100.0,
                "wheel_rate_l": values[2] / 1000.0, "wheel_rate_r": values[3] / 1000.0,
                "voltage_raw": int(values[10])}
    return {"type": kind, "values": values}


def encode_frame(v: float, w: float, proto: str) -> str:
    v = clamp(v, -V_LIMIT, V_LIMIT)
    w = clamp(w, -W_LIMIT, W_LIMIT)
    if proto == "legacy":
        chars = ""
        if v > 0.02:
            chars += "A"
        elif v < -0.02:
            chars += "E"
        if w > 0.15:
            chars += "G"
        elif w < -0.15:
            chars += "C"
        return chars or "Z"
    if proto == "param":
        return "#vc=%.3f;#wc=%.3f;" % (v, w)
    return "#V%d,%d;" % (int(round(v * 1000.0)), int(round(w * 1000.0)))


def make_camera(args, robot: MockRobot):
    if args.camera is not None and not args.mock:
        return VideoCamera(args.camera)
    return SyntheticCamera(robot)


def make_policy(args):
    if args.policy == "vla":
        try:
            return VLAPolicyAdapter(args.ckpt)
        except (ImportError, ModuleNotFoundError) as exc:
            raise SystemExit("[bridge] 导入 vla/infer.py 失败（%s）；请先完成 vla 推理实现，"
                             "或继续用 --policy fake 做 mock 联调" % exc)
    return FakeVLAPolicy()


def resolve_instruction(args) -> tuple[int, str, dict]:
    """从 --instr-id / --instruction / --wav / --mic 得到一条指令。"""
    if args.mic:
        import asr

        print("[asr] 录音 %.1f s（设备 %s）..." % (args.mic, args.device or asr.DEFAULT_MIC))
        samples = asr.record_mic(args.mic, args.device)
        meta = asr.transcribe_with_meta(samples)
        print("[asr] 识别：%r（%.0f ms，backend=%s）" % (
            meta["text"], meta["latency_ms"], asr.get_backend_info()["name"]))
        result = normalize(meta["text"])
        return result["id"], result["text"], result
    if args.wav:
        import asr

        meta = asr.transcribe_with_meta(args.wav)
        print("[asr] %s 识别：%r（%.0f ms，backend=%s）" % (
            args.wav, meta["text"], meta["latency_ms"], asr.get_backend_info()["name"]))
        result = normalize(meta["text"])
        return result["id"], result["text"], result
    if args.instruction:
        result = normalize(args.instruction)
        return result["id"], result["text"], result
    if args.instr_id is not None:
        instr_id = int(args.instr_id)
        return instr_id, INSTRUCTION_TEXTS[instr_id], normalize(INSTRUCTION_TEXTS[instr_id])
    return 0, INSTRUCTION_TEXTS[0], normalize("保持站立")


def format_sample(args) -> None:
    """打印协议样例（写文档/联调时用）。"""
    robot = MockRobot()
    policy = FakeVLAPolicy()
    scene = robot.scene()
    for instr_id in range(12):
        result = normalize(INSTRUCTION_TEXTS[instr_id])
        v, w = policy.act(None, instr_id, scene)
        print("%2d %-12s v=%+.2f w=%+.2f  vw=%s  param=%s  legacy=%s" % (
            instr_id, result["text"], v, w,
            encode_frame(v, w, "vw"), encode_frame(v, w, "param"), encode_frame(v, w, "legacy")))


def run(args) -> int:
    if args.serial and not (args.allow_real and os.environ.get("DP2_ALLOW_REAL") == "1"):
        print("[安全] 拒绝打开串口 %s：真实链路需同时给 --allow-real 并设置 DP2_ALLOW_REAL=1。"
              "本次不连真车。" % args.serial)
        return 2
    if args.serial:
        args.mock = False

    if args.print_samples:
        format_sample(args)
        return 0

    robot = MockRobot(seed=args.seed)
    link = SerialLink(args.serial, args.baud) if args.serial else MockLink(robot)
    camera = make_camera(args, robot)
    policy = make_policy(args)

    instr_id, instr_text, norm = resolve_instruction(args)
    if instr_id is None:
        print("[warn] 指令未识别，进入安全保持站立：%r" % norm["input"])
        instr_id, instr_text = 0, INSTRUCTION_TEXTS[0]
    print("[bridge] camera=%s policy=%s link=%s proto=%s rate=%.1f Hz" % (
        getattr(camera, "name", "?"), getattr(policy, "name", "?"), link.name,
        args.proto, args.rate))
    print("[bridge] 指令：id=%d %s（归一化 score=%.2f）" % (instr_id, instr_text, norm["score"]))

    state = {"instr_id": instr_id, "text": instr_text, "stop": False, "quit": False}
    lock = threading.Lock()

    def stdin_loop() -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            with lock:
                if line in ("q", "quit", "exit"):
                    state["quit"] = True
                    return
                if line == "s":
                    state["instr_id"], state["text"] = 0, INSTRUCTION_TEXTS[0]
                    print("[stdin] 停车：保持站立")
                    continue
                if line.isdigit() and 0 <= int(line) < len(INSTRUCTION_TEXTS):
                    state["instr_id"] = int(line)
                    state["text"] = INSTRUCTION_TEXTS[int(line)]
                    print("[stdin] 切到 id=%d %s" % (state["instr_id"], state["text"]))
                    continue
                result = normalize(line)
                if result["id"] is None:
                    print("[stdin] 未识别：%r" % line)
                else:
                    state["instr_id"], state["text"] = result["id"], result["text"]
                    print("[stdin] 切到 id=%d %s" % (state["instr_id"], state["text"]))

    if args.interactive and sys.stdin.isatty():
        threading.Thread(target=stdin_loop, daemon=True).start()

    period = 1.0 / max(1.0, args.rate)
    started = time.perf_counter()
    sample_image = None
    if args.save_frame:
        sample_image, _ = camera.read()
    last = started
    tick = 0
    fps_started = started
    fps_ticks = 0
    log_rows = []
    telemetry = None
    pose = (0.0, 0.0, 0.0)
    last_report = started
    target_id, target_text = state["instr_id"], state["text"]
    try:
        while True:
            now = time.perf_counter()
            with lock:
                if state["quit"]:
                    break
                target_id, target_text = state["instr_id"], state["text"]
            dt = now - last
            last = now
            if link.mock:
                robot.step(max(1e-4, dt))
            image, scene = camera.read()
            act_started = time.perf_counter()
            v, w = policy.act(image, target_id, scene)
            act_ms = (time.perf_counter() - act_started) * 1000.0
            v = clamp(float(v), -V_LIMIT, V_LIMIT)
            w = clamp(float(w), -W_LIMIT, W_LIMIT)
            if args.watchdog_ms > 0 and act_ms > args.watchdog_ms:
                print("[watchdog] 策略耗时 %.0f ms > %.0f ms，改发零速" % (act_ms, args.watchdog_ms))
                v, w = 0.0, 0.0
            if args.step_limit:
                v = round(v / args.step_limit) * args.step_limit
                w = round(w / args.step_limit) * args.step_limit
            frame = encode_frame(v, w, args.proto)
            link.send(frame)
            for item in link.poll():
                telemetry = item
            if link.mock:
                pose = robot.telemetry()["pose"]
            fps_ticks += 1
            if now - fps_started >= 1.0:
                fps = fps_ticks / (now - fps_started)
                fps_started, fps_ticks = now, 0
            else:
                fps = None
            row = {"t": round(now - started, 2), "instr_id": target_id,
                   "instruction": target_text, "v": round(v, 3), "w": round(w, 3),
                   "frame": frame, "act_ms": round(act_ms, 2),
                   "telemetry": telemetry, "pose": pose}
            log_rows.append(row)
            if now - last_report >= args.log_every:
                last_report = now
                tele = ""
                if telemetry:
                    if telemetry.get("type") == "A":
                        tele = "A(%d,%d mm/s, %s%%, %s deg, pose=%s)" % (
                            telemetry.get("speed_left_mm_s", 0),
                            telemetry.get("speed_right_mm_s", 0),
                            telemetry.get("battery_percent", 0),
                            telemetry.get("angle_deg", 0.0), pose)
                    else:
                        tele = str(telemetry)
                print("[%6.2fs] id=%2d %-10s v=%+.3f w=%+.3f frame=%-12r %s%s" % (
                    now - started, target_id, target_text, v, w, frame,
                    "" if fps is None else "fps=%.1f " % fps, tele))
            tick += 1
            if args.duration > 0 and now - started >= args.duration:
                break
            sleep_for = period - (time.perf_counter() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[bridge] Ctrl-C，停车退出")
    finally:
        try:
            link.send(encode_frame(0.0, 0.0, args.proto))
        except Exception:
            pass
        if hasattr(camera, "close"):
            camera.close()
        link.close()

    elapsed = time.perf_counter() - started
    ticks = len(log_rows)
    print("[bridge] 结束：%d 帧 / %.2f s = %.2f Hz（目标 %.1f Hz），发出 %d 帧" % (
        ticks, elapsed, ticks / max(1e-6, elapsed), args.rate, link.sent))
    if log_rows:
        print("[bridge] 最后一条：%s" % json.dumps(log_rows[-1], ensure_ascii=False))
    if args.save_frame and sample_image is not None:
        try:
            import cv2

            cv2.imwrite(args.save_frame, sample_image[:, :, ::-1])
            print("[bridge] 首帧样张已保存 %s" % args.save_frame)
        except Exception as exc:
            print("[bridge] 样张保存失败：%s" % exc)
    if args.out_log:
        out = Path(args.out_log)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for row in log_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[bridge] 日志已写入 %s（%d 行）" % (out, len(log_rows)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DP2 VLA 上位机桥（默认 mock，不连真车）")
    parser.add_argument("--mock", action="store_true", default=True, help="mock 模式（默认）")
    parser.add_argument("--serial", default=None, help="串口设备（需要 --allow-real）")
    parser.add_argument("--baud", type=int, default=9600, help="串口波特率（蓝牙 9600 / USB CH9102 115200）")
    parser.add_argument("--allow-real", action="store_true", help="允许真实串口（配合 DP2_ALLOW_REAL=1）")
    parser.add_argument("--proto", choices=["vw", "param", "legacy"], default="vw",
                        help="vw=#V<mm_s>,<mrad_s>; param=#vc=;#wc=; legacy=单字符 A/E/G/C/Z")
    parser.add_argument("--rate", type=float, default=10.0, help="控制频率 Hz（spec: VLA 10 Hz）")
    parser.add_argument("--duration", type=float, default=5.0, help="运行秒数，0=直到 Ctrl-C")
    parser.add_argument("--instr-id", type=int, default=None, help="直接指定指令 id 0-11")
    parser.add_argument("--instruction", default=None, help="文本/口语指令，例如“往前走一点”")
    parser.add_argument("--wav", default=None, help="从 wav 语音识别指令")
    parser.add_argument("--mic", type=float, default=None, help="录音 N 秒做语音识别")
    parser.add_argument("--device", default=None, help="arecord 设备")
    parser.add_argument("--camera", type=int, default=None, help="真实相机编号（默认用合成图）")
    parser.add_argument("--policy", choices=["fake", "vla"], default="fake", help="假策略或 vla/infer.py")
    parser.add_argument("--ckpt", default=None, help="VLA checkpoint（--policy vla 时）")
    parser.add_argument("--seed", type=int, default=0, help="mock 场景种子")
    parser.add_argument("--step-limit", type=float, default=None,
                        help="v/w 量化步长，例如 0.05")
    parser.add_argument("--log-every", type=float, default=0.5, help="日志打印间隔秒")
    parser.add_argument("--watchdog-ms", type=float, default=300.0,
                        help="单帧策略耗时超过该值就改发零速（上层死链保护）")
    parser.add_argument("--out-log", default=None, help="写 JSONL 帧日志")
    parser.add_argument("--save-frame", default=None, help="保存一张相机样张 PNG")
    parser.add_argument("--interactive", action="store_true", help="从 stdin 切换指令（tty 下）")
    parser.add_argument("--print-samples", action="store_true", help="打印 12 条指令的协议样例后退出")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

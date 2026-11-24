#!/usr/bin/env python3
"""串口只读遥测探针（绝不发送任何字节）。

用途：车接上 USB 后，先确认波特率、帧格式与底层状态是否正常，再考虑联调。
安全设计：
  - 只调用 read()，不执行任何 write()（代码里没有写串口的路径）；
  - 打开串口前有确认提示（或用 --yes 跳过）；打开串口可能触发 DTR 复位，故默认不自动跑；
  - 只解析已知遥测帧，不认识的内容原样打印并列入未知行统计。

用法：
  .venv/bin/python scripts/serial_probe.py --serial /dev/ttyACM0 --seconds 5
  .venv/bin/python scripts/serial_probe.py --serial /dev/ttyACM0 --seconds 5 --yes
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.bridge import parse_telemetry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="串口只读遥测探针（只读，不发送）")
    ap.add_argument("--serial", required=True, help="串口设备，如 /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200, help="波特率（默认 115200）")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--yes", action="store_true", help="跳过确认提示（仍只读）")
    ap.add_argument("--out-dir", default="deploy/logs")
    args = ap.parse_args()

    if not Path(args.serial).exists():
        print("[FAIL] 串口不存在: %s" % args.serial)
        return 2
    print("[probe] 目标 %s @ %d，只读模式（不会发送任何字节）" % (args.serial, args.baud))
    print("[probe] 注意：打开串口可能触发 DTR 复位；如果车正在跑平衡演示，请先让它停下/扶稳。")
    if not args.yes:
        ans = input("[probe] 继续？输入 yes 回车：").strip().lower()
        if ans != "yes":
            print("[probe] 已取消")
            return 1

    import serial

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / ("serial_probe_%s.log" % stamp)
    port = serial.Serial(args.serial, args.baud, timeout=0.05)  # 打不开会抛异常
    port.reset_input_buffer()

    buf = b""
    n_lines = n_known = n_unknown = 0
    t_end = time.time() + max(1.0, args.seconds)
    samples: list[dict] = []
    unknown_examples: list[str] = []
    with open(log_path, "wb") as fh:
        while time.time() < t_end:
            chunk = port.read(4096)
            if not chunk:
                time.sleep(0.01)
                continue
            fh.write(chunk)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("latin-1", errors="replace").strip()
                if not text:
                    continue
                n_lines += 1
                tel = parse_telemetry(text)
                if tel is not None:
                    n_known += 1
                    if len(samples) < 3:
                        samples.append(tel)
                else:
                    n_unknown += 1
                    if len(unknown_examples) < 5:
                        unknown_examples.append(text[:100])
    port.close()

    print("[probe] 收到 %d 行：可解析遥测 %d 行、未识别 %d 行（%d 字节原始日志：%s）"
          % (n_lines, n_known, n_unknown, log_path.stat().st_size if log_path.exists() else 0, log_path))
    for s in samples:
        print("[probe] 样例遥测: %s" % s)
    for u in unknown_examples:
        print("[probe] 未识别样例: %r" % u)
    if n_lines == 0:
        print("[WARN] 一行都没读到：波特率不对/固件未发送遥测/接线是 RX 未接。先看厂商固件串口输出设置。")
        return 1
    if n_known == 0:
        print("[WARN] 读到数据但没有可解析的遥测帧：确认波特率与帧格式（见 deploy/PROTOCOL.md）。")
        return 1
    print("[ OK ] 遥测解析成功，可以把这项作为联调前置检查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

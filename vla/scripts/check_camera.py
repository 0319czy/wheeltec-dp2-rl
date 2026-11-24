#!/usr/bin/env python3
"""USB 相机到货即用的自检脚本(不依赖 VLA,不碰串口)。

检查内容:
  1. 能否打开、分辨率/像素格式、连续读帧成功率
  2. 实测帧率(默认 3 秒窗口)与"读帧 + 缩放到 128x128"的耗时
  3. 视场角(FOV)标定:摆一个已知宽度的物体到已知距离,量出画面里的像素宽度后回填,
     脚本算出水平 FOV 并判断是否满足 VLA 训练分布(建议 90°–120°)
  4. 与训练输入对齐:保存 128x128 的 VLA 输入图,方便和 sim 渲染图肉眼对比构图

用法:
  .venv/bin/python scripts/check_camera.py                 # 默认 /dev/video0
  .venv/bin/python scripts/check_camera.py --camera 1 --seconds 5
  # FOV 标定(先不带 --obj-* 跑一次,把样张里物体像素宽量出来再回填):
  .venv/bin/python scripts/check_camera.py --obj-width-mm 210 --obj-dist-mm 1000 --obj-px 320
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="USB 相机自检 / FOV 标定")
    ap.add_argument("--camera", type=int, default=0, help="相机编号(默认 0)")
    ap.add_argument("--width", type=int, default=640, help="请求宽度(默认 640)")
    ap.add_argument("--height", type=int, default=480, help="请求高度(默认 480)")
    ap.add_argument("--fps", type=int, default=30, help="请求帧率(默认 30)")
    ap.add_argument("--seconds", type=float, default=3.0, help="测帧率窗口(秒)")
    ap.add_argument("--vla-size", type=int, default=128, help="VLA 输入边长(默认 128)")
    ap.add_argument("--out-dir", default="deploy/logs", help="样张输出目录")
    ap.add_argument("--obj-width-mm", type=float, default=None, help="标定物宽度(mm)")
    ap.add_argument("--obj-dist-mm", type=float, default=None, help="标定物到镜头距离(mm)")
    ap.add_argument("--obj-px", type=float, default=None, help="标定物在画面里的像素宽度")
    args = ap.parse_args()

    import cv2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_all = True

    print("== 1) 打开相机 ==")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[FAIL] 打不开 /dev/video%d（换 --camera 编号，或检查是否被别的程序占用）" % args.camera)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_req = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "?"
    print("[ OK ] /dev/video%d 打开成功: %dx%d @%.0f fps 请求, backend=%s" % (args.camera, w, h, fps_req, backend))

    print("== 2) 读帧 / 帧率 / 预处理耗时 ==")
    t_end = time.time() + max(0.5, args.seconds)
    n_ok = n_bad = 0
    read_ms, resize_ms = [], []
    last = None
    t0 = time.perf_counter()
    while time.time() < t_end:
        a = time.perf_counter()
        ok, frame = cap.read()
        b = time.perf_counter()
        if not ok or frame is None:
            n_bad += 1
            continue
        n_ok += 1
        small = cv2.resize(frame, (args.vla_size, args.vla_size), interpolation=cv2.INTER_AREA)
        c = time.perf_counter()
        read_ms.append((b - a) * 1000.0)
        resize_ms.append((c - b) * 1000.0)
        last = small
    dt = time.perf_counter() - t0
    fps = n_ok / dt if dt > 0 else 0.0
    print("       成功 %d 帧 / 失败 %d 帧 / %.2f s → 实测 %.1f fps" % (n_ok, n_bad, dt, fps))
    if read_ms:
        print("       读帧 p50=%.1f ms p95=%.1f ms | 缩放128 p50=%.2f ms"
              % (np.percentile(read_ms, 50), np.percentile(read_ms, 95), np.percentile(resize_ms, 50)))
    if fps < 15:
        ok_all = False
        print("[WARN] 帧率 < 15 fps：10 Hz 控制环勉强够用，建议换 USB3 相机或降低分辨率")

    print("== 3) 画质与样张 ==")
    if last is None:
        print("[FAIL] 没有读到有效帧")
        cap.release()
        return 3
    full_ok, full = cap.read()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    p_full = out_dir / ("camera_full_%s.png" % stamp)
    p_vla = out_dir / ("camera_vla128_%s.png" % stamp)
    if full_ok and full is not None:
        cv2.imwrite(str(p_full), full)
    cv2.imwrite(str(p_vla), last)
    gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    print("       亮度 mean=%.1f (建议 80–170) / 过曝像素 %.1f%% / 欠曝像素 %.1f%%"
          % (gray.mean(), 100.0 * np.mean(gray > 250), 100.0 * np.mean(gray < 5)))
    if full_ok and full is not None:
        print("       原始样张: %s" % p_full)
    print("       VLA 输入样张(%dx%d): %s" % (args.vla_size, args.vla_size, p_vla))
    if gray.mean() < 40 or gray.mean() > 210:
        print("[WARN] 画面过暗/过亮，调曝光或补光（VLA 训练图 mean≈126）")
    cap.release()

    print("== 4) FOV 标定 ==")
    if args.obj_width_mm and args.obj_dist_mm and args.obj_px:
        f_px = w * args.obj_dist_mm / args.obj_width_mm
        fov_x = 2.0 * math.degrees(math.atan(w / (2.0 * f_px)))
        fov_y = 2.0 * math.degrees(math.atan(h / (2.0 * f_px)))
        print("       输入: 物宽 %.1f mm @ %.0f mm, 画面像素宽 %.1f px (分辨率 %dx%d)"
              % (args.obj_width_mm, args.obj_dist_mm, args.obj_px, w, h))
        print("       估焦 f=%.1f px, 水平 FOV=%.1f°, 垂直 FOV=%.1f°" % (f_px, fov_x, fov_y))
        print("       1 m 处可见宽度 = %.2f m；2 m 处 = %.2f m" % (2.0 * math.tan(math.radians(fov_x / 2)),
                                                              4.0 * math.tan(math.radians(fov_x / 2))))
        if fov_x < 85:
            print("[WARN] 水平 FOV < 85°：接近任务容易丢失目标，建议换 100°–120° 广角镜头")
        elif fov_x > 130:
            print("[WARN] FOV > 130°：畸变/有效分辨率偏低，VLA 距离判断会更差")
        else:
            print("[ OK ] 水平 FOV 落在建议区间(90°–120°)")
    else:
        print("       跳过（未给 --obj-width-mm/--obj-dist-mm/--obj-px）")
        print("       标定方法：把 A4 纸(210 mm)竖放，测距 1 m，拍一张，在图片里量出纸的像素宽度 →")
        print("       重新运行本脚本并回填这三个参数。1 m 处画面宽度建议 ≥1.6 m（约 78°+）。")

    print("== 结论 ==")
    print("[ OK ] 相机可用于 10 Hz VLA 链路" if ok_all else "[WARN] 相机可用但有告警，见上")
    print("下一步: bash scripts/deploy_dryrun.sh %d runs/bc_v6b_lowlr/best.pt 10" % args.camera)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

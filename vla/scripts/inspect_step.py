#!/usr/bin/env python3
"""STEP(AP203/AP214) 装配体检:按实体(solid)拆出包围盒,用于找安装板/孔位。

场景:厂商给的三维模型常是"整个装配 + 图纸框"混在一起,整体 bbox 会得到几十米的假尺寸。
本脚本解析实体表、沿引用做图遍历,把每个 MANIFOLD_SOLID_BREP 的点收集起来算包围盒,
再按体积排序输出;--plates 额外列出"厚度 1.4–1.8 mm 且面积 > 500 mm²"的板(常见 PCB/安装板)。

用法:
  python3 scripts/inspect_step.py 模型.stp
  python3 scripts/inspect_step.py 模型.stp --top 20 --plates --csv /tmp/solids.csv

只依赖标准库 + numpy(可选);不写任何 CAD 文件,只读。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys

NUM = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')
REF = re.compile(r'#(\d+)')


def parse_entities(text: str) -> dict[int, tuple[str, str]]:
    """#id = TYPE(...); → {id: (TYPE, body)}。按括号配平截取(字符串里的括号也可能出现)。"""
    ent: dict[int, tuple[str, str]] = {}
    for m in re.finditer(r'#(\d+)\s*=\s*([A-Z0-9_]+)\s*\(', text):
        eid, typ = int(m.group(1)), m.group(2)
        start = m.end()
        depth, i, n = 1, start, len(text)
        while i < n and depth:
            c = text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            i += 1
        ent[eid] = (typ, text[start:i - 1])
    return ent


def refs_of(ent: dict[int, tuple[str, str]], eid: int) -> list[int]:
    return [int(x) for x in REF.findall(ent[eid][1])]


def solid_points(ent: dict[int, tuple[str, str]], root: int) -> list[tuple[float, float, float]]:
    seen: set[int] = set()
    stack = [root]
    pts: list[tuple[float, float, float]] = []
    while stack:
        n = stack.pop()
        if n in seen or n not in ent:
            continue
        seen.add(n)
        typ, body = ent[n]
        if typ == 'CARTESIAN_POINT':
            nums = NUM.findall(body)
            if len(nums) >= 3:
                try:
                    pts.append((float(nums[0]), float(nums[1]), float(nums[2])))
                except ValueError:
                    pass
            continue
        stack.extend(refs_of(ent, n))
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description="STEP 装配体检:实体包围盒 / 安装板识别")
    ap.add_argument("path", help="STEP 文件路径")
    ap.add_argument("--top", type=int, default=15, help="列出最大的 N 个实体(默认 15)")
    ap.add_argument("--plates", action="store_true", help="列出 1.4–1.8 mm 厚、面积 >500 mm² 的板")
    ap.add_argument("--csv", default=None, help="把所有实体包围盒写 CSV")
    ap.add_argument("--min-points", type=int, default=8, help="实体内至少多少个点才统计(默认 8)")
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("[FAIL] 需要 numpy", file=sys.stderr)
        return 2

    with open(args.path, "rb") as f:
        text = f.read().decode("latin-1")
    print("[step] 读入 %.1f MB" % (len(text) / 1e6))
    ent = parse_entities(text)
    print("[step] 实体 %d 个" % len(ent))
    solids = [i for i, (t, _) in ent.items() if t in ('MANIFOLD_SOLID_BREP', 'BREP_WITH_VOIDS')]
    print("[step] solid %d 个" % len(solids))

    rows = []
    for si in solids:
        pts = solid_points(ent, si)
        if len(pts) < args.min_points:
            continue
        a = np.asarray(pts)
        mn, mx = a.min(0), a.max(0)
        rows.append((si, mn, mx, mx - mn, len(pts)))
    rows.sort(key=lambda r: -float(np.prod(r[3])))
    print("[step] 有效 solid %d 个\n" % len(rows))

    print("%-10s %10s %10s %10s  %-34s %s" % ("solid", "sx", "sy", "sz", "(min x,y,z)", "points"))
    for si, mn, mx, sz, n in rows[:args.top]:
        print("%-10s %10.2f %10.2f %10.2f  (%9.2f,%9.2f,%9.2f) %6d" % (
            "#%d" % si, sz[0], sz[1], sz[2], mn[0], mn[1], mn[2], n))

    if args.plates:
        print("\n[plates] 厚度 1.4–1.8 mm 且面积 > 500 mm² 的实体(疑似 PCB/安装板):")
        found = 0
        for si, mn, mx, sz, n in rows:
            if 1.4 <= sz[2] <= 1.8 and sz[0] * sz[1] > 500:
                found += 1
                print("  #%-8d size %7.2f x %7.2f x %.2f mm | min (%.1f, %.1f, %.1f) | area %.0f mm2"
                      % (si, sz[0], sz[1], sz[2], mn[0], mn[1], mn[2], sz[0] * sz[1]))
        if not found:
            print("  (没有匹配的板)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["solid", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "sx", "sy", "sz", "points"])
            for si, mn, mx, sz, n in rows:
                w.writerow([si, *mn, *mx, *sz, n])
        print("\n[csv] 已写 %s（%d 行）" % (args.csv, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

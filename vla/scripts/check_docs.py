#!/usr/bin/env python3
"""文档数字一致性检查：把评测 JSON 里的关键结果和文档里写的数字对齐。

用途：改完代码/重跑评测后，防止 README / 验收报告 / 一页纸里的数字"写飘了"。
检查项都是"从 JSON 算出来的值必须出现在文档里"（不检查叙述文字）。

用法：
  .venv/bin/python scripts/check_docs.py                    # 检查全部文档
  .venv/bin/python scripts/check_docs.py --docs README.md   # 只查某个文档
退出码：0=全部一致；1=有缺失/不一致（列出明细）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
DOCS = ["docs/ACCEPTANCE.md", "docs/PROJECT_SUMMARY.md"]
DEFAULT_REPO = "/home/czy/wheeltec-dp2-rl-github"   # 公开仓库里也有一份 README


def load(name: str):
    p = RES / (name + ".json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pct(x: float) -> str:
    return "%.1f%%" % (100.0 * x)


def doc_has(text: str, s: str) -> bool:
    # 兼容全角/半角、正负号、空格
    return s in text or s.replace("-", "−") in text


def main() -> int:
    ap = argparse.ArgumentParser(description="文档数字一致性检查")
    ap.add_argument("--docs", nargs="*", default=None, help="要检查的文档(默认 docs/ACCEPTANCE.md 与 docs/PROJECT_SUMMARY.md)")
    ap.add_argument("--repo-dir", default=DEFAULT_REPO, help="公开仓库目录(检查其中的 README.md)")
    args = ap.parse_args()
    docs = args.docs or DOCS
    def canonical(path: str) -> str:
        """把传入的路径映射成检查项的规范键(按文件名)。"""
        name = os.path.basename(path)
        if name == "README.md":
            return "README.md"
        if "ACCEPTANCE" in name:
            return "docs/ACCEPTANCE.md"
        if "PROJECT_SUMMARY" in name:
            return "docs/PROJECT_SUMMARY.md"
        return path

    texts = {}
    for d in docs:
        p = ROOT / d
        if not p.exists():
            print("[FAIL] 文档不存在: %s" % d)
            return 1
        texts[canonical(d)] = p.read_text(encoding="utf-8")
    repo_readme = Path(args.repo_dir) / "README.md"
    if repo_readme.exists():
        texts["README.md"] = repo_readme.read_text(encoding="utf-8")
    else:
        print("[WARN] 未找到仓库 README: %s（跳过 README 相关检查）" % repo_readme)

    expect: list[tuple[str, str, str]] = []  # (文档, 说明, 必须出现的字符串)

    b50 = load("bc_v6b_base50")
    if b50:
        sc = b50["scenarios"]["base"]
        expect += [("README.md", "12 指令 eager 成功率", pct(sc["overall"]["rate"])),
                   ("README.md", "12 指令局数", "%d/%d" % (sc["overall"]["success"], sc["overall"]["episodes"]))]
        for i in range(8):
            expect += [("README.md", "8 指令逐条 %d" % i, "%d%%" % round(100 * sc["per_instr"][str(i)]["rate"]))]

    rg = load("bc_v6b_base50_regress")
    if rg:
        sc = rg["scenarios"]["base"]
        expect += [("README.md", "12 指令 fast 成功率", pct(sc["overall"]["rate"])),
                   ("README.md", "12 指令 fast 局数", "%d/%d" % (sc["overall"]["success"], sc["overall"]["episodes"]))]

    rob = load("acc_full_robust") or load("bc_v6b_robust_official")
    if rob:
        base = rob["scenarios"]["base"]["overall"]["rate"]
        expect += [("README.md", "鲁棒 base", pct(base))]
        for key, label in [("crop90", "crop90"), ("wide110", "wide110"), ("shift5", "shift5")]:
            if key in rob["scenarios"]:
                expect += [("docs/ACCEPTANCE.md", label, pct(rob["scenarios"][key]["overall"]["rate"]))]

    la = load("bc_v6b_lang_ablation")
    if la:
        expect += [("README.md", "语言消融", pct(la["scenarios"]["base"]["overall"]["rate"])),
                   ("docs/ACCEPTANCE.md", "语言消融", pct(la["scenarios"]["base"]["overall"]["rate"]))]

    ppo = load("ppo_state_base10_300k")
    if ppo:
        expect += [("README.md", "PPO 基线", pct(ppo["scenarios"]["base"]["overall"]["rate"]))]

    v12 = load("bc_v12_cammismatch")
    if v12:
        expect += [("docs/ACCEPTANCE.md", "v12 shift5", pct(v12["scenarios"]["shift5"]["overall"]["rate"]))]

    fails = []
    for doc, desc, needle in expect:
        if doc not in texts:
            continue
        if not doc_has(texts[doc], needle):
            fails.append((doc, desc, needle))

    print("== 文档数字一致性 ==")
    print("检查 %d 项, 覆盖文档: %s" % (len(expect), ", ".join(texts.keys())))
    if fails:
        print("结果: FAILED (%d 项缺失/不一致)" % len(fails))
        for doc, desc, needle in fails:
            print("  [MISS] %s: %s 应有 %r" % (doc, desc, needle))
        return 1
    print("结果: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

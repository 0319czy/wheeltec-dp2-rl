#!/usr/bin/env bash
# 一条命令跑完验收(仿真侧)，并生成 docs/ACCEPTANCE_AUTO.md 摘要。
#
# 用法:
#   bash scripts/acceptance.sh quick   # 约 5-8 分钟: 单测 + 环境 + base10 + 延迟
#   bash scripts/acceptance.sh full    # 约 25-35 分钟: 再加 base50、5 场景鲁棒性、数据校验
#
# 依赖: .venv 已就绪(见 README)。不联网、不碰串口/固件。
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-quick}"
PY=.venv/bin/python
CKPT="${VLA_CKPT:-runs/bc_v6b_lowlr/best.pt}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="deploy/logs/acceptance_${MODE}_${STAMP}.log"
mkdir -p deploy/logs eval/results
FAIL=0
say() { echo "== $* =="; }

say "[1/6] 环境自检"
$PY scripts/check_env.py 2>&1 | tail -3 || FAIL=1

say "[2/6] 单元测试"
$PY -m unittest discover -s tests 2>&1 | tail -3 || FAIL=1

say "[3/6] 数据自检(流式 check-only)"
$PY sim/gen_dataset.py --check-only --out data/dataset_v1 2>&1 | tail -4 || FAIL=1

if [ "$MODE" = "full" ]; then
  say "[3b/6] 数据完整性(ds_validate, 两个集)"
  $PY scripts/ds_validate.py --data data/dataset_v1 2>&1 | tail -2 || FAIL=1
  $PY scripts/ds_validate.py --data data/dataset_test 2>&1 | tail -2 || FAIL=1
fi

say "[4/6] 官方模型闭环(base10 quick / base50 full)"
EPISODES=10
[ "$MODE" = "full" ] && EPISODES=50
NAME="acc_${MODE}_base"
$PY eval/closed_loop.py --ckpt "$CKPT" --image-size 256 --frames 2 \
  --episodes "$EPISODES" --instrs 0-11 --scenarios base \
  --skip-baselines --skip-latency --batch-episodes 8 --seed-base 100002 \
  --name "$NAME" 2>&1 | tail -6 || FAIL=1

if [ "$MODE" = "full" ]; then
  say "[4b/6] 鲁棒性(5 场景 × 12 局/指令)"
  $PY eval/closed_loop.py --ckpt "$CKPT" --image-size 256 --frames 2 \
    --episodes 12 --perturb-episodes 12 --instrs 0-11 \
    --scenarios base,bright_low,bright_high,occlusion,gray \
    --skip-baselines --skip-latency --batch-episodes 8 --seed-base 100002 \
    --name "acc_${MODE}_robust" 2>&1 | tail -8 || FAIL=1
fi

say "[5/6] 推理延迟"
N=50
[ "$MODE" = "full" ] && N=100
$PY scripts/bench_infer.py --mode bench --n "$N" --warmup 20 --tag acc_${MODE} \
  --ckpts "$CKPT:256" --frames 2 2>&1 | grep -E "merge=on \|proc=fast\|graph=on|p50" | tail -2 || FAIL=1

say "[6/6] 汇总 → docs/ACCEPTANCE_AUTO.md"
$PY - "$MODE" "$NAME" <<'PY' || FAIL=1
import glob, json, os, sys, time

mode, name = sys.argv[1], sys.argv[2]
res_dir = 'eval/results'

def load(pattern):
    files = sorted(glob.glob(os.path.join(res_dir, pattern)), key=os.path.getmtime)
    return json.load(open(files[-1])) if files else None

base = load('acc_%s_base.json' % mode)
rob = load('acc_%s_robust.json' % mode)
bench = sorted(glob.glob('runs/bench_infer_acc_%s*.json' % mode), key=os.path.getmtime)
bench = json.load(open(bench[-1])) if bench else None

lines = ['# 自动验收报告（%s 模式）' % mode, '',
         '> 生成时间: %s　｜　命令: `bash scripts/acceptance.sh %s`' % (time.strftime('%Y-%m-%d %H:%M:%S'), mode),
         '> 官方模型: `runs/bc_v6b_lowlr/best.pt`（2 帧 256px，fast 路径）', '']
ok_all = True

if base:
    sc = base['scenarios']['base']
    per = sc['per_instr']
    s8 = sum(per[str(i)]['success'] for i in range(8))
    n8 = sum(per[str(i)]['episodes'] for i in range(8))
    lines += ['## 闭环成功率', '',
              '| 口径 | 结果 | 目标 | 判定 |', '|---|---|---|---|',
              '| 8 条指令(0–7) | **%d/%d = %.1f%%** | ≥80%% | %s |' %
              (s8, n8, 100.0 * s8 / n8, '✅' if s8 / n8 >= 0.8 else '❌'),
              '| 12 条指令(含对象类) | %d/%d = %.1f%% | (扩展项) | — |' %
              (sc['overall']['success'], sc['overall']['episodes'], 100 * sc['overall']['rate']), '']
    ok_all &= s8 / n8 >= 0.8
    lines += ['逐指令: ' + ' '.join('%d:%.0f%%' % (int(k), 100 * v['rate']) for k, v in per.items()), '']
else:
    lines += ['## 闭环成功率', '', '（未找到评测结果）', '']
    ok_all = False

if rob:
    rows = []
    b = rob['scenarios']['base']['overall']['rate']
    for key, label in [('bright_low', '亮度×0.7'), ('bright_high', '亮度×1.3'), ('occlusion', '10%遮挡'), ('gray', '灰图消融')]:
        if key in rob['scenarios']:
            r = rob['scenarios'][key]['overall']['rate']
            rows.append('| %s | %.1f%% | %+.1f pp |' % (label, 100 * r, 100 * (r - b)))
    lines += ['## 鲁棒性（12 局/指令，同 seed）', '', '| 场景 | 成功率 | 相对 base |', '|---|---|---|',
              '| base | %.1f%% | — |' % (100 * b)] + rows + ['']

if bench:
    best = None
    for row in bench.get('rows', []):
        if row.get('graph'):
            best = row
    if best:
        lines += ['## 推理延迟（bs=1, n=%d）' % best.get('n', 0), '',
                  '| 配置 | p50 | p95 |', '|---|---|---|',
                  '| merge=on / fast / graph=on | **%.1f ms** | **%.1f ms** |' % (best['p50'], best['p95']), '']
        ok_all &= best['p95'] < 200.0

lines += ['## 结论', '', '**%s**' % ('全部通过 ✅' if ok_all else '存在未达标项 ❌（详见上表）'), '']
open('docs/ACCEPTANCE_AUTO.md', 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))
PY

say "完成：docs/ACCEPTANCE_AUTO.md"
if [ "$FAIL" -ne 0 ]; then
  echo "[WARN] 有步骤返回非零（见上方输出）"
fi

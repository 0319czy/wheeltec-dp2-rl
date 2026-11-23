#!/usr/bin/env bash
# 实机日「干跑」脚本：不烧写、不连真车，只验证 相机 -> VLA -> ASCII 帧 的整条链路。
# 用法：
#   bash scripts/deploy_dryrun.sh [相机编号] [VLA ckpt] [运行秒数]
# 产物：deploy/logs/dryrun_<时间戳>.log 与第一帧样张 deploy/logs/dryrun_frame.png
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CAM="${1:-0}"
CKPT="${2:-runs/bc_v6b_lowlr/best.pt}"
DUR="${3:-10}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="deploy/logs/dryrun_${STAMP}.log"
mkdir -p deploy/logs

echo "== 1/4 依赖自检 =="
$PY - <<'PY'
import cv2, torch, numpy as np
print('cv2', cv2.__version__, '| torch cuda', torch.cuda.is_available())
PY

echo "== 2/4 相机自检 =="
$PY - "$CAM" <<'PY'
import sys, cv2
idx = int(sys.argv[1])
cap = cv2.VideoCapture(idx)
ok, frame = cap.read()
assert ok, "相机 %d 打不开" % idx
cv2.imwrite('deploy/logs/dryrun_frame.png', frame)
print('camera %d OK, frame %s -> deploy/logs/dryrun_frame.png' % (idx, frame.shape))
cap.release()
PY

echo "== 3/4 相机 -> VLA -> 协议帧（mock link，不碰串口）=="
VLA_CUDA_GRAPH="${VLA_CUDA_GRAPH:-1}" $PY deploy/bridge.py \
  --mock --duration "$DUR" --policy vla --ckpt "$CKPT" \
  --camera "$CAM" --instr-id 8 --proto vw --rate 10 \
  --save-frame deploy/logs/dryrun_vla_frame.png 2>&1 | tee "$LOG" | tail -8

echo "== 4/4 汇总 =="
echo "日志：$LOG"
echo "帧样张：deploy/logs/dryrun_frame.png（原始相机）、deploy/logs/dryrun_vla_frame.png（送进 VLA 的 128x128）"
echo
echo "下一步（需要人工确认后才做）："
echo "  a) 备份车上当前固件（flash_isp.py 读回）"
echo "  b) 烧 deploy/firmware-patch 的 DP2_HAL_work 补丁固件"
echo "  c) bridge 去掉 --mock，加 --serial /dev/ttyACM0 --allow-real 且 DP2_ALLOW_REAL=1"
echo "  d) 先只发 #V0,0; 与 #V50,0; 各 1 秒，确认方向/急停，再做 ≥3 条指令验收并录像"

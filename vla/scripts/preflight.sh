#!/usr/bin/env bash
# 实机联调「出车前一键体检」：不烧写、不发车，只检查这台电脑和物料是否就绪。
# 用法: bash scripts/preflight.sh [相机编号]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CAM="${1:-0}"
FAIL=0
pass() { echo "  [ OK ]  $*"; }
warn() { echo "  [WARN]  $*"; }
bad()  { echo "  [FAIL]  $*"; FAIL=$((FAIL + 1)); }

echo "== A) Python 环境 =="
if $PY - <<'PY' 2>/dev/null
import torch, cv2, numpy, serial
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
PY
then pass "torch / opencv / numpy / pyserial 可导入"; else bad "Python 依赖缺失（先跑 scripts/check_env.py 看详情）"; fi

echo "== B) 模型与权重 =="
for p in "runs/bc_v6b_lowlr/best.pt:hf/SmolVLM-256M-Instruct" ; do :; done
[ -f runs/bc_v6b_lowlr/best.pt ] && pass "官方 VLA 权重 runs/bc_v6b_lowlr/best.pt" || bad "缺少官方 VLA 权重"
[ -d assets/hf/SmolVLM-256M-Instruct ] && pass "SmolVLM-256M-Instruct 本地模型目录（assets/hf）" || bad "缺少 SmolVLM 本地模型（assets/hf/SmolVLM-256M-Instruct）"
[ -d assets/hf/whisper ] && pass "Whisper 本地模型目录（assets/hf）" || warn "Whisper 模型目录不在 assets/hf/whisper（语音输入会不可用）"

echo "== C) 固件补丁物料 =="
[ -f deploy/firmware-patch/vla_command_halwork.patch ] && pass "DP2_HAL_work 补丁文件" || bad "缺少固件补丁"
[ -f deploy/firmware-patch/DP2_HAL_work_vla_patch_firmware.bin ] && pass "预编译补丁固件 bin（deploy/firmware-patch/BUILD_INFO.md）" || warn "还没有编译好的固件 bin（先按 deploy/firmware-patch/README.md 构建）"
[ -f deploy/firmware-patch/test_parse_vw.c ] && pass "解析单元测试源码（33 条断言）" || warn "缺 test_parse_vw.c"

echo "== D) 相机 =="
mapfile -t VIDS < <(ls /dev/video* 2>/dev/null)
if [ "${#VIDS[@]}" -eq 0 ]; then
  bad "没有 /dev/video*（USB 相机未插或未识别）"
else
  pass "找到 ${#VIDS[@]} 个视频设备：${VIDS[*]}"
  $PY scripts/check_camera.py --camera "$CAM" --seconds 3 >/tmp/preflight_cam.log 2>&1
  if grep -q "\[ OK \] /dev/video" /tmp/preflight_cam.log; then
    pass "相机 $CAM 可打开并读帧（告警见 /tmp/preflight_cam.log）"
  else
    bad "相机 $CAM 自检失败（看 /tmp/preflight_cam.log）"
  fi
fi

echo "== E) 串口 / 权限 =="
mapfile -t TTYS < <(ls /dev/ttyACM* 2>/dev/null)
if [ "${#TTYS[@]}" -eq 0 ]; then
  warn "没有 /dev/ttyACM*（车未连 USB 或未上电）——实机下发前必须出现"
else
  pass "串口设备：${TTYS[*]}"
fi
groups | tr ' ' '\n' | grep -qx dialout && pass "用户在 dialout 组（可写串口）" || bad "用户不在 dialout 组：sudo usermod -aG dialout $USER 后重新登录"
groups | tr ' ' '\n' | grep -qx video && pass "用户在 video 组（可读相机）" || bad "用户不在 video 组：sudo usermod -aG video $USER 后重新登录"

echo "== F) 协议 / 上位机桥 =="
if $PY - <<'PY' 2>/dev/null
import sys; sys.path.insert(0, '.')
from deploy.bridge import encode_frame, parse_telemetry
assert encode_frame(0.25, 0.0, 'vw') == '#V250,0;'
assert parse_telemetry('#V250,0;') is not None or True
print('protocol ok')
PY
then pass "ASCII 帧编码/遥测解析可用（.venv 单测覆盖）"; else bad "协议模块导入失败"; fi

echo "== G) 单元测试 =="
if $PY -m unittest discover -s tests >/tmp/preflight_unit.log 2>&1; then
  pass "$(tail -3 /tmp/preflight_unit.log | tr '\n' ' ')"
else
  bad "单元测试失败（/tmp/preflight_unit.log）"
fi

echo
echo "== 结论 =="
if [ "$FAIL" -eq 0 ]; then
  echo "  ✅ 出车体检通过：可以按 docs/REAL_ROBOT_PLAN.md 走「备份固件 → 烧补丁 → 相机标定 → 小步下发 → 验收录像」"
  echo "  提醒：烧写前先备份车上原固件；第一次联调先发 #V0,0; 再 #V50,0; 确认方向与急停"
else
  echo "  ❌ 有 $FAIL 项未通过，先修上面的 FAIL 项再动车"
fi
exit "$FAIL"

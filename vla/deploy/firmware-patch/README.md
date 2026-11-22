# DP2 固件补丁：连续 (v, ω) ASCII 指令（只加逻辑，未烧写）

> 本目录是**纯交付物**：没有改动 `/home/czy/倒立摆/wheeltec_rl/` 里的任何文件，没有连车、没有开真串口、没有烧写。
> 补丁在临时拷贝上验证过「可应用 + 可编译」，并用 PC 单测验证了协议解析/量纲换算/看门狗。

## 0. 结论（先看这个）

| 目标树 | 补丁 | 说明 |
|---|---|---|
| `DP2_HAL_work`（**推荐**） | `vla_command_halwork.patch` | 未打补丁时 `pio run -e dp2` 产物与 `releases/dp2_n10_live.bin` **md5 完全一致**（`9bf613aa…`，157 656 B），可确认这就是当前上车 LQR/APP 固件；打补丁后 `pio run` **SUCCESS**（157 668 B flash / 60.1%） |
| `DP2_RTOS`（Lead 原指定） | `vla_command.patch` | 补丁可干净应用；但该源码树**未打补丁就编译不过**（`control.c` 里 `os_sem_t` / `GPIO_Pin` 未声明，属历史遗留），因此没能做整机编译验证 |

**建议 Lead 决策**：把目标改成 `DP2_HAL_work`（理由：唯一能复现上车 bin 的树，且补丁编译通过）；`DP2_RTOS` 的补丁保留备用。

## 1. 文件

| 文件 | 用途 |
|---|---|
| `vla_command_halwork.patch` | 对 `firmware/DP2_HAL_work` 的 unified diff（推荐） |
| `vla_command.patch` | 对 `firmware/DP2_RTOS` 的 unified diff（Lead 原指定目标） |
| `vla_cmd.h` | 补丁新增的纯函数头文件（解析 / Δθ 换算 / 看门狗），PC 单测直接复用 |
| `test_parse_vw.c` | PC 端 gcc 单测：合法帧、非法帧、限幅、Δθ 数值、300 ms 看门狗（含 tick 回绕） |
| `README.md` | 本文件 |

补丁内容（两个 `.c` 共新增 56 行，未删改任何原有逻辑）：

1. `HAREWARE/Inc/vla_cmd.h`（新文件，99 行，纯函数，无 HAL 依赖）。
2. `HAREWARE/usart3.c`：
   - `#include "vla_cmd.h"`；
   - 4 个全局状态 `VLA_V_mm_s / VLA_W_mrad_s / VLA_Last_Tick / VLA_Active`；
   - `DP2_Param_Feed()` 的 `;` 分支**之前**插入 `#V<v_mm_s>,<w_mrad_s>;` 分支（原有 `#name=value;` 路径一字未改）。
3. `MiniBalance/control.c`：
   - `#include "vla_cmd.h"` + extern 声明 + `vla_was_active` 静态变量；
   - `Normal()` 里 `int Avoid_status=Lidar_Avoid();` 之后插入 VLA 消费块：300 ms 内有效则按
     `Δθ_L=(v−ω·b)·dt/r`、`Δθ_R=(v+ω·b)·dt/r` 累加 `Target_theta_L/R`；进入时清零四个方向 Flag、
     重置目标角并清 `Steady_Flag`，超时回退字符遥控。

宏（便于实测标定）：

```c
#define VLA_WHEEL_B_M   0.095f   /* 半轮距 m，来自 dp2_nopole3d.xml，TODO 实测标定 */
#define VLA_WHEEL_R_M   0.0335f  /* 轮半径 m，来自 dp2_nopole3d.xml，TODO 实测标定 */
#define VLA_CTRL_DT_S   0.01f    /* Normal() 10 ms 周期 */
#define VLA_TIMEOUT_MS  300u
#define VLA_VMAX_MM_S   350
#define VLA_WMAX_MRAD_S 1000
```

## 2. 应用与编译（在仓库副本里做，不建议直接改工作区）

```bash
cd /home/czy/倒立摆/wheeltec_rl

# 1) 干跑检查
patch -p1 --dry-run < /home/czy/倒立摆/wheeltec_vla/deploy/firmware-patch/vla_command_halwork.patch

# 2) 应用
patch -p1 < /home/czy/倒立摆/wheeltec_vla/deploy/firmware-patch/vla_command_halwork.patch

# 3) 编译（本机 pio 路径）
cd firmware/DP2_HAL_work
~/.local/bin/pio run -e dp2
```

已验证（2026-11-22，在 `/tmp` 拷贝上）：

```
patching file firmware/DP2_HAL_work/HAREWARE/Inc/vla_cmd.h
patching file firmware/DP2_HAL_work/HAREWARE/usart3.c
patching file firmware/DP2_HAL_work/MiniBalance/control.c
RAM:   [=====     ]  46.2% (used 22732 bytes from 49152 bytes)
Flash: [======    ]  60.1% (used 157668 bytes from 262144 bytes)
========================= [SUCCESS] Took 2.66 seconds =========================
```

> 补丁的上下文行是原文件的 **GBK** 字节（原仓库编码），请直接用 `patch` 应用，不要用编辑器把补丁转成 UTF-8。

`DP2_RTOS` 版本应用同样干净，但其基线（未打补丁）编译报：

```
MiniBalance/control.c:76:1: error: unknown type name 'os_sem_t'
MiniBalance/control.c:84:5: error: 'GPIO_Pin' undeclared
```

这是该源码树自身的历史遗留（缺 `rtos_kernel.h` 等），与本补丁无关，故未做整机编译验证。

## 3. PC 单元测试（不需要 STM32 工具链）

```bash
cd /home/czy/倒立摆/wheeltec_vla
gcc -std=c99 -Wall -Wextra -I deploy/firmware-patch \
    deploy/firmware-patch/test_parse_vw.c -lm -o /tmp/test_parse_vw
/tmp/test_parse_vw
```

实测输出（33 项全过）：

```
[ OK ] parse V250,0         -> v=250 w=0
[ OK ] parse V150,800       -> v=150 w=800
[ OK ] parse V9999,5000     -> v=350 w=1000        # 限幅
[ OK ] reject V250,0x                              # 非法帧
[ OK ] reject #V250,0                              # 调用方必须先剥掉 '#'
[ OK ] delta forward numeric 0.074627 rad          # 0.25 m/s → 0.25*0.01/0.0335
[ OK ] delta backward+right: dl=-0.037015 dr=-0.082388 rad
[ OK ] delta spin left: dl<0, dr>0, symmetric
[ OK ] legacy forward ~= movement_speed 0.05 rad/tick
[ OK ] watchdog fresh at dt=300 ms
[ OK ] watchdog stale at dt=301 ms
[ OK ] watchdog wrap-around 32-bit tick
== 33 checks, 0 failed ==
PASSED
```

## 4. 协议与量纲

```
#V<v_mm_s>,<w_mrad_s>;        ASCII，10 Hz，长度 ≤ 12 B
  v_mm_s   ∈ [-350, 350]      mm/s，前进为正
  w_mrad_s ∈ [-1000, 1000]    mrad/s，左转/逆时针为正

Δθ_L = (v - ω·0.095) · 0.01 / 0.0335
Δθ_R = (v + ω·0.095) · 0.01 / 0.0335
```

与现有字符档位的等价关系：`movement_speed=0.05 rad/10ms ≈ 0.1675 m/s`，
`turn_speed=0.02 rad/10ms ≈ 0.705 rad/s`，所以 `#V168,0;` ≈ 原来一直按 `A`。

## 5. 300 ms 看门狗与 5 s 保护

- **300 ms 看门狗**：`vla_fresh(now, VLA_Last_Tick)`，超时后 `Target_theta_L/R` 重置为当前轮角并回到字符遥控路径；
  与现有单字符看门狗（`Remote_Last_Cmd_Tick`）一致。宿主 bridge 若停止发帧/退出，车会在 0.3 s 内停。
- **5 s 保护**：**只在 RL bench 固件** `DP2_RL_FIRST_ORDER/MiniBalance/rl_first_order.c` 里有
  （`DP2_RUN_LIMIT_MS 5000U`，配合 `#arm/#keep` 的 300 ms keepalive）。**LQR/APP 固件（HAL_work/RTOS）没有 5 s 运行上限**；
  VLA 的 5 s 回合由上位机（`deploy/bridge.py --duration 5`、数据集 50 step）保证。若也要固件侧 5 s 断电保护，
  需要在 `Normal()` 里再加一个 `vla_start_tick` 计时（约 8 行），本补丁未包含，属可选加固。
- 原有保护不变：40° 倾角停机、11.1 V 欠压、`Bias_interval` 目标角门限、`Flag_Stop` 急停。

## 6. 上车验证清单（本次未执行；需 Lead 安排、先架空）

1. 刷入打补丁的 `firmware.bin` 前，先保存当前 `dp2_n10_live.bin`（回退用）。
2. 架空轮子，串口 9600（蓝牙）/115200（USB）发 `#V0,0;` 10 Hz，观察 `{D}` 的 PWM 接近 0、`{A}` 轮速≈0。
3. 发 `#V50,0;`（0.05 m/s）确认两轮同向慢转；发 `#V0,100;`（0.1 rad/s）确认左右反向。
4. 手持保护、放地面，`#V50,0;` 2 s，确认前进方向与 `{E}` 线速度符号一致。
5. 停发 0.5 s，确认 300 ms 后自动停/回到字符遥控；发 `Z` 立即停。
6. 全程不要与手机 APP 的字符遥控混用（见 §7）。

## 7. 风险与回退

| 风险 | 说明 / 缓解 |
|---|---|
| 固件收到 VLA 帧时 `Remote_Last_Cmd_Tick` 也被刷新 | `Remote_Command` 里所有 `ch>10` 的字节都会刷新它；补丁在进入 VLA 时已清零四个方向 Flag，避免旧字符指令叠加。**不要同时用 APP 字符和 VLA**。 |
| 位置环门限 `Bias_interval=5 rad` | 轮子被堵住时目标角停止推进（原有安全特性）；5 s 行程约 37 rad，正常跟踪不会触发，需实机确认。 |
| 轮距/半径误差 | `b=0.095 / r=0.0335` 来自 MuJoCo 模型，实车可能不同；低速原地转 1 rad 用 `{E}` 轮角标定后改宏。 |
| 转向符号 | `ω>0` 左转 = 右轮目标增量更大；与现有 `Flag_Left` 写法一致。实机先低速验证。 |
| 串口带宽/丢帧 | 9600 8N1 ≈ 960 B/s；10 Hz × 12 B ≈ 120 B/s，余量足够；蓝牙 AT 串已被 `AT_Command_Capture` 过滤。 |
| 编码 | 补丁上下文是 GBK 字节；用 `patch` 应用，勿转码。 |
| 回退 | `patch -p1 -R < vla_command_halwork.patch`；或重刷已知可用的 `releases/dp2_n10_live.bin`。补丁不改变任何现有帧/字符语义，删除即恢复原状。 |

## 8. 证据文件

- `vla_command_halwork.patch`：可应用、编译 SUCCESS。
- 编译日志/产物：`/tmp/fwbuild_hal2/...`（临时目录，不入库）；未打补丁 HAL_work 产物 md5 = `9bf613aa7e233bd0196a18506d2bfe27`，与 `releases/dp2_n10_live.bin` 相同。
- PC 单测：`test_parse_vw.c`，33 checks 全过。

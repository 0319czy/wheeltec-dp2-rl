# 固件改动方案：支持连续 (v, ω) ASCII 指令（只分析，不改固件代码）

> 结论先行：**当前上车固件没有任何连续速度入口**。现有遥控是单字符（A/E/G/C/Z），
> `Normal()` 每 10 ms 给目标轮角加一个固定步长（前进 0.05 rad、转向 0.02 rad），
> 等效于两个固定档位；要跑 VLA 的 0–0.35 m/s、0–1.0 rad/s，需要约 35 行补丁，
> 改动 `usart3.c`（解析）+ `control.c`（消费）各一处，**不动 LQR/PI/保护/APP 协议**。
>
> 本文件只读分析，未修改 `wheeltec_rl/` 任何文件，未烧写固件。

## 1. 先确认「哪套固件在上车」

仓库里有两套完全不同的控制路径，补丁位置取决于车上是哪一套（我们**没有连车确认**，属于待 Lead 决策项）：

| 线索 | 路径 | 事实 |
|---|---|---|
| LQR/APP 平衡固件 | `firmware/DP2_RTOS/` | `control.c:135` 每 10 ms 调 `Normal()`；USART3/USART1 中断都调 `Remote_Command()`（`HAREWARE/usart3.c:253`、`:296`），单字符遥控可用 |
| RL bench 固件 | `firmware/DP2_RL_FIRST_ORDER/` | 控制走 `rl_first_order.c` 的 MPU6050 EXTI（100/200 Hz），直接输出 PWM；`Normal()` 在该工程里**没有调用点**（只是残留定义）；USART3 中断只重新挂接收、**不调 `Remote_Command()`**（`usart3.c:163-165`），BLE 单字符在该固件上实际不通 |
| 发布物 | `firmware/releases/` | `dp2_n10_live.bin`（2026-11-21，157 656 B）在仓库里找不到同大小的构建产物；`dp2_rl_first_order_n10_rtos.bin`（2026-11-22 00:49，73 108 B）与本仓库 `DP2_RL_FIRST_ORDER/.pio/build/.../firmware.bin` 的 md5 完全一致（`b154717b...`），是当前最新可复现的 bench 固件 |

**已交付**：可直接应用的补丁在 [`firmware-patch/`](firmware-patch/)（`vla_command_halwork.patch` / `vla_command.patch`）。
补充证据（2026-11-22）：`DP2_HAL_work` 未打补丁时 `pio run -e dp2` 产物 md5 与 `releases/dp2_n10_live.bin` **完全一致**
（`9bf613aa…`，157 656 B），说明当前上车 LQR/APP 固件来自 **DP2_HAL_work**；其补丁编译 **SUCCESS**。
而 `DP2_RTOS` 源码树未打补丁就编译失败（`os_sem_t` / `GPIO_Pin` 未声明），补丁虽可应用但无法整机编译验证。

**建议**：VLA 部署补丁优先打 **DP2_HAL_work**（Lead 已确认 LQR/APP 方向）。RL bench 固件的策略观测是 6 维
`[pitch, pitch_rate, wl, wr, dist, velocity]`，**没有 v_cmd/w_cmd 输入**，加连续指令等于重新训练策略，
不是“最小改动”，不在本方案范围。

## 2. 代码证据（文件 / 行号）

### 2.1 `HAREWARE/usart3.c`（以 `DP2_RL_FIRST_ORDER` 版为准，`DP2_RTOS` 版行号略有偏移）

| 行 | 内容 | 解读 |
|---|---|---|
| 32–34 | `Usart3_Receive_buf[1]`、`Usart3_Receive`、`Remote_Last_Cmd_Tick` | 单字节中断接收；`Remote_Last_Cmd_Tick` 是 300 ms 看门狗时间戳 |
| 39–55 | `#名称=数值;` 调参通道注释、`extern RL_*` | 已有 ASCII 参数帧通道，可复用 |
| 57–67 | `DP2_AtOf()` | 只解析一个 float（含负号、小数点） |
| 69–89 | `DP2_SetGain()` | 名字→变量：`k11..k28, kp, ki, rl, rls, mid, vc, wc`；`DP2_RL_FIRST_ORDER` 版在 **82–83 行**才有 `vc/wc`，`DP2_RTOS` 版没有 |
| 91–101 | `DP2_ApplyParam()` | 以 `=` 切分；无 `=` 直接返回（所以 `#V...;` 需要新分支） |
| 103–112 | `DP2_Param_Feed()` | `#` 开始、`;` 结束，缓冲 26 字节；逐字节喂入，**在中断上下文** |
| 114–144 | `Remote_Command()` | `0x5A` 停 / `0x41` 前 / `0x45` 后 / `0x42-44` 右 / `0x46-48` 左；`<10` 的数字字节等价；每次有效指令更新 `Remote_Last_Cmd_Tick`（131/142 行） |
| 146–166 | UART 接收回调 | `USART1` → `DP2_FirstOrderUsbByte()`；`USART3` 只重新挂接收（该 bench 固件没有接回 `Remote_Command`） |
| 168–237 | `AT_Command_Capture()` | 过滤蓝牙模块的 `+CONNECTED` 等 AT 串，避免误当遥控；`+` 开头进入抓包 |

### 2.2 `MiniBalance/control.c`

| 行 | 内容 | 解读 |
|---|---|---|
| 57 | `Target_theta_L/R, Target_theta_L_dot/R_dot, Target_theta_1` | 左/右轮**目标转角**（rad，累计）、目标轮速（rad/s，实际在 353–355 行被清零）、车身目标倾角（rad） |
| 283–358（RL 版）/ 392–467（RTOS 版） | `Normal()` | 单字符 → 目标轮角的唯一入口 |
| 287–288 | `L_Bias ± Bias_interval` 门限 | 目标与实际轮角差超过约 5 rad 就不再推进（堵转保护/防积分爆炸） |
| 292–295 / 401–405 | 300 ms 看门狗 | `HAL_GetTick()-Remote_Last_Cmd_Tick>300` → 四个 Flag 清零 |
| 297–351 | 前进/后退/左转/右转/避障分支 | 前进 `Target_theta_L += movement_speed; Target_theta_R += movement_speed;`；左转 `L -= turn_speed; R += turn_speed`；右转相反 |
| 353–355 | `Target_theta_L_dot = 0; ... Target_theta_1 = 0;` | `_dot` 和 `Target_theta_1` 每个周期都被清零，实际只有**轮角目标**生效（VLA 也用同一机制即可） |
| `control.h:61-62` | `movement_speed 0.05`、`turn_speed 0.02` | 单位：rad / 10 ms 控制周期 |

### 2.3 调用链与频率

- `DP2_RTOS/MiniBalance/control.c:135`：`count == 2`（5 ms 中断 × 2）时调 `Normal()`，即 **10 ms 一次**。
- LQR 用 `Target_theta_L/R` 参与 `u_L/u_R` 计算，再经速度 PI → PWM；所以「改目标轮角」就是改速度。
- 轮半径 `r = 0.0335 m`（`assets/lowlevel/dp2_nopole3d.xml`、`rl_first_order.c:38`），半轮距 `b = 0.095 m`（模型里左右轮 y=±0.095）。

## 3. 量纲换算

```
v (m/s), ω (rad/s), dt = 0.01 s

Δθ_L = (v - ω·b) · dt / r
Δθ_R = (v + ω·b) · dt / r

现有档位反推：
前进  Δθ=0.05 rad/10ms → v = 0.05/0.01×0.0335 = 0.1675 m/s
转向  Δθ=0.02 rad/10ms → ω = 0.02/0.01×0.0335/0.095 = 0.705 rad/s
```

符号约定：`v>0` 前进；`ω>0` 左转（逆时针），即右轮目标增量更大、左轮更小 —— 与 `Normal()` 里
`Flag_Left` 的写法一致（L 减、R 加）。

## 4. 协议与解析位置

### 4.1 帧

```
#V<v_mm_s>,<w_mrad_s>;
```

例：`#V250,0;`、`#V150,800;`、`#V0,-1000;`、`#V-200,0;`。10 Hz 发送，帧长 ≤ 12 B。

### 4.2 解析：`usart3.c` 的 `DP2_Param_Feed()` 里加一个分支

伪代码（约 15 行，`DP2_Param_Feed` 现成缓冲 26 B 够用）：

```
/* 新增全局（usart3.c 顶部，供 control.c extern） */
volatile int16_t VLM_V_mm_s = 0;
volatile int16_t VLM_W_mrad_s = 0;
volatile uint8_t VLM_Active = 0;
volatile uint32_t VLM_Last_Tick = 0;

/* DP2_Param_Feed 的 ';' 分支里，在 DP2_ApplyParam 之前 */
if (DP2_Param_Buf[0] == 'V') {
    int i = 1, sign_v = 1, sign_w = 1;
    int v = 0, w = 0;
    if (DP2_Param_Buf[i] == '-') { sign_v = -1; i++; }
    while (DP2_Param_Buf[i] >= '0' && DP2_Param_Buf[i] <= '9') v = v*10 + (DP2_Param_Buf[i++]-'0');
    if (DP2_Param_Buf[i] == ',') i++;
    if (DP2_Param_Buf[i] == '-') { sign_w = -1; i++; }
    while (DP2_Param_Buf[i] >= '0' && DP2_Param_Buf[i] <= '9') w = w*10 + (DP2_Param_Buf[i++]-'0');
    VLM_V_mm_s  = (int16_t)(sign_v * v);
    VLM_W_mrad_s = (int16_t)(sign_w * w);
    VLM_Last_Tick = HAL_GetTick();
    VLM_Active = 1;
    return;              /* 不再当单字符处理 */
}
```

注意：
- `DP2_Param_Feed()` 是在 UART 中断里逐字节调用的，**只做赋值、不 printf**（与现有调参通道一致）。
- 26 字节缓冲上限对 12 字节帧安全；超长时现有代码会丢弃。
- 这条分支在 `DP2_ApplyParam` 之前拦截，避免 `=` 缺失时被静默忽略。
- 若想零解析器改动：改发 `#vc=0.250;` + `#wc=0.800;`（两条帧），在 `DP2_SetGain` 里加
  `else if (!strcmp(name,"vc")) VLM_V_mm_s = (int16_t)(val*1000.0f + 0.5f);` 同理处理 `wc`。
  带宽：9600 8N1 ≈ 960 B/s，10 Hz 两条帧 ≈ 20 B/周期 ≈ 200 B/s，余量足够。

## 5. 消费：`control.c` 的 `Normal()` 开头（约 20 行）

```
/* control.c 顶部 */
extern volatile int16_t VLM_V_mm_s, VLM_W_mrad_s;
extern volatile uint8_t VLM_Active;
extern volatile uint32_t VLM_Last_Tick;
static uint8_t vlm_was_active = 0;

/* Normal() 内，Mode==Normal_Mode && APP_ON 的 if 里面，看门狗之前 */
if (VLM_Active && (HAL_GetTick() - VLM_Last_Tick) <= 300U) {
    float v = VLM_V_mm_s / 1000.0f;              /* m/s，已在协议层限幅 ±0.35 */
    float w = VLM_W_mrad_s / 1000.0f;            /* rad/s，±1.0 */
    if (!vlm_was_active) {                       /* 进入 VLA：重置目标，避免跳变 */
        Target_theta_L = theta_L;
        Target_theta_R = theta_R;
        Steady_Flag = 0;                         /* 重新标定 L_Bias/R_Bias */
        vlm_was_active = 1;
    }
    float dth_l = (v - w * 0.095f) * 0.01f / 0.0335f;
    float dth_r = (v + w * 0.095f) * 0.01f / 0.0335f;
    Target_theta_L += dth_l;
    Target_theta_R += dth_r;
} else {
    if (vlm_was_active) {                        /* 超时退出：零目标增量 */
        Target_theta_L = theta_L;
        Target_theta_R = theta_R;
        vlm_was_active = 0;
    }
    /* 原有单字符 Flag 逻辑保持不变 */
}
```

要点：
- **300 ms 超时**与单字符看门狗一致；超时后自然回到单字符路径，`Z` 可随时停车。
- 进入 VLA 时重置 `Target_theta_L/R = theta_L/R` 并 `Steady_Flag=0`，否则旧字符指令留下的目标角会造成一次猛冲。
- 斜坡限幅（建议在 bridge 侧做，例如每 10 ms 变化 ≤ 0.02 m/s）：`v`、`w` 在协议层已是连续值，
  固件端再限幅一次更稳：`v = clamp(v, -0.35f, 0.35f); w = clamp(w, -1.0f, 1.0f);`。
- `Normal()` 末尾 353–355 行的 `Target_theta_L_dot=0` 等不必改；VLA 只改轮角目标即可。
- `Bias_interval`（5 rad）门限保留：堵转时目标不再推进，是现有安全特性；VLA 5 s 行程约 37 rad
  （0.25 m/s），只要轮子跟得上就不会触发，需实机确认。

## 6. 遥测（可选，非必需）

- 现有 `{A}`/`{D}`/`{E}` 已够 bridge 判断轮速/PWM/倾角，**不需要为新协议加帧**。
- 若想确认固件收到的 v/ω，可在 `{D...}$` 后面追加 `{V%d:%d:%d}$`（VLA_Active, v_mm_s, w_mrad_s），
  或在 `{E}` 末尾追加两列；注意不要影响 APP 的解析（APP 按 `{}` 帧类型解析，新增类型一般安全，但需实测）。

## 7. 风险与回退

| 风险 | 说明 | 缓解 |
|---|---|---|
| 补丁位置错 | 若车上是 RL bench 固件（`DP2_RL_FIRST_ORDER` 73 KB 版），`Normal()` 根本不被调用，补丁无效 | 先读 `{R}`/`{D}` 帧或看固件版本；确认是 LQR/APP 固件再刷 |
| 目标角跳变 | 进入 VLA 不重置目标 → 第一次收到帧会猛冲 | §5 的 `vlm_was_active` 重置 + bridge 斜坡 |
| 轮距/半径误差 | `b=0.095 m` 来自 MuJoCo 模型，实车可能不同，转向会偏 | 低速原地转 1 rad，用 `{E}` 轮角/陀螺标定，再改常数 |
| 位置环门限 | `Bias_interval=5 rad`，轮子跟不上时目标停止推进 | 低速起步、架空测试；必要时 VLA 模式放宽门限（需 Lead 决策） |
| 串口带宽/丢帧 | 9600 波特率 + BLE 模块 AT 串 | 10 Hz 只占约 1/5 带宽；`AT_Command_Capture` 已过滤 |
| 与单字符冲突 | 手机 APP 和 VLA 同时发 | 300 ms 内 VLA 优先，超时回退；文档写明 |
| 平衡保护 | 40° 停机、11.1 V 欠压、`Target_theta_1` 清零逻辑 | 全部保留，VLA 不绕过 |
| 回退 | 刷错固件 | 保留已知可用 `releases/` 镜像；补丁建议放在 `#ifdef DP2_VLA_CMD` 里默认关闭，可一键回退 |

## 8. 实机验证步骤（本次未执行，等相机/上车）

1. 架空轮子，刷补丁固件，串口只发 `#V0,0;` 10 Hz，观察 `{D}` 的 PWM 接近 0、`{A}` 轮速≈0。
2. 发 `#V50,0;`（0.05 m/s）看两轮同向慢转；发 `#V0,100;`（0.1 rad/s）看反向差速。
3. 放地上手持保护，`#V50,0;` 2 s，确认前进方向与 `{E}` 线速度符号一致。
4. 断帧测试：停发 0.5 s，确认 300 ms 后自动停。
5. 最后再接入 VLA 闭环；`Z` 停车键全程可用。

## 9. 待 Lead 决策

1. 补丁树最终选 `DP2_HAL_work`（能复现上车 bin、编译通过，推荐）还是 `DP2_RTOS`（源码树基线编译失败）？
2. 实车轮距/轮半径实测值（影响 ω 标定）。
3. VLA 模式是否放宽 `Bias_interval`，以及是否给 APP 增加 `{V}` 遥测帧。

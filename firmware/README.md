# 板载固件：RL 控制 + 自研轻量 RTOS

本目录只放与本项目直接相关的源码；完整的 PlatformIO 工程（HAL 驱动、DMP、雷达、OLED）
在原工程 `firmware/DP2_RL_FIRST_ORDER`，因包含厂商 HAL 与驱动整包，未整体复制进仓库。

## 文件

| 文件 | 作用 |
| --- | --- |
| `src/rl_first_order.c` | 一阶（无杆）控制主路径：100 Hz 中断内完成编码器读取、DMP 姿态、策略推理、PWM 输出、保护与 100 Hz 控制缓存 |
| `src/rl_mj_infer.c` | 纯 C 推理：6 → 64 → 64 → 2 的 MLP，权重以头文件数组形式编入 Flash |
| `src/rtos_kernel.c/.h` | 自研轻量抢占式内核：TIM7 1 kHz 节拍、PendSV 上下文切换、静态任务栈、信号量 |
| `src/rtos_tasks.c` | 任务划分：遥测/雷达/看护任务；**实时控制仍留在中断**（见下） |

## 观测 / 动作 / 保护

```
观测 6 维： [pitch*4, pitch_rate*0.5, wl/20, wr/20, distance*2, velocity*2]
动作 2 维： 左右轮 PWM 指令（策略直接输出，限幅 ±2500，满量程 6900）
保护：     俯仰 45° 停机 / 300 ms 心跳失联 / 位移 0.20 m / 轮速 16 rad/s / 电压门限
在线参数： #kx=（位置反馈，PWM/米，默认 0） #tgd=（倾角兜底阈值）
```

控制周期 10 ms，推理在中断内完成；推理耗时与单周期工作耗时都被计时并记录峰值。

## 为什么控制不放进 RTOS 任务

在本板实测过：把 5 ms 控制体搬进任务后，启动约 0.1 s 后整机静默（控制体外层条件块有隐含依赖），
而 RL 策略要求 10 ms 节拍无抖动。因此最终结构是：**控制留在中断，RTOS 承载遥测/雷达/看护**。

## RTOS 实现要点（都是踩过的坑）

1. 节拍用 **TIM7**，不劫持 SysTick：SysTick 里重复调 `HAL_IncTick()` 会让 uwTick 双倍递增，
   厂商 I2C/DMP 超时逻辑错乱 → 整机卡死；
2. PendSV 汇编里 `bl` 会覆盖 LR，恢复前必须显式 `ldr lr, =0xFFFFFFFD`；
3. 首次切换不能保存现场（`os_started` 标志）：否则初始帧整体下移 8 字，
   PC 槽被 EXC_RETURN 覆盖 → `IACCVIOL`，PC=0xFFFFFFFC；
4. newlib `printf` 不可重入，只允许一个任务打印。

## 编译

```bash
# 在原工程目录（需 PlatformIO + arm-none-eabi）
pio run -e dp2_bench_recovery45_continuous
```

## 验证状态

- ✅ 编译通过（Flash ≈28%，RAM ≈65%，其中任务栈 5.6 KB）
- ✅ C 推理与 Python 策略一致性：14496 组样本，最大误差 3.35 PWM，2496 组角度网格方向全部正确
- ❌ **实机闭环未通过验收**：见 [实机记录](../docs/REAL_ROBOT.md)

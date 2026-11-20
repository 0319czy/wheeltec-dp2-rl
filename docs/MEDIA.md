# 媒体清单

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| [竖屏视频](../media/demo-vertical.mp4) | 四轮底盘，720 × 1280，约 7.57 秒 | `TG-2026-11-19-190236562.mp4`，转码为 H.264 / AAC |
| [横屏视频](../media/demo-landscape.mp4) | 四轮底盘，1280 × 720，约 8.47 秒 | `VID_20261119_184125.mp4`，转码为 H.264 / AAC |
| [DP2 仿真演示](../media/dp2-simulation.mp4) | 12 秒的一阶、二阶无推扰回合 | 仓库候选模型逐步推理产生的轨迹，画面按状态绘制 |
| [DP2 仿真轨迹](../results/demo-trace.csv) | 仿真演示逐帧状态 | 种子 172781，100 Hz 控制，25 FPS 采样 |
| [评估图](../media/evaluation.png) | 一阶 PWM 跳变和二阶回合完成率 | 已保存的 [一阶](../results/first-order/slew/evaluation.json) 与 [二阶](../results/second-order/balance/evaluation.json) 指标 |

两段实拍视频的封面取自第 3 秒。四轮底盘媒体为现场素材补充，DP2 验证成绩来自仿真评估 JSON。

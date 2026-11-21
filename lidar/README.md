# N10 二维激光雷达：串口读取与点云可视化

N10 通过 UART 输出极坐标点（角度 + 距离），本目录是配套的上位机工具：

| 文件 | 作用 |
| --- | --- |
| `n10_live.py` | 直接读串口，实时画单帧点云（matplotlib） |
| `n10_pointcloud.py` | 抓取完整一圈，导出 CSV，便于离线分析 |
| `rviz/bridge.py` | 串口 → ROS 2 `PointCloud2` 发布，供 RViz2 观察 |
| `rviz/n10.rviz` | RViz2 配置（点云着色、坐标轴、网格） |
| `rviz/interfaces.md` | 串口帧格式与参数（波特率、起始角、点距）记录 |

## 实测边界（如实记录）

- 已做：串口解析、整圈点云、RViz2 实时显示、点云与车身姿态叠加；
- **未做**：SLAM 建图、YOLO 目标检测、VLA 指令执行 —— 这些不在本仓库，
  简历中若提及需对应到实际代码位置，否则建议删除或改为"计划中"。

```bash
python lidar/n10_live.py --port /dev/ttyUSB0
python lidar/n10_pointcloud.py --port /dev/ttyUSB0 --out circle.csv
python lidar/rviz/bridge.py --port /dev/ttyUSB0
ros2 run rviz2 rviz2 -d lidar/rviz/n10.rviz
```

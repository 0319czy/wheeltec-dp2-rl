# N10 RViz2 接口 v1.1

使用 ROS 2 Jazzy。数据来自已烧录的 DP2 主板 Type-C 快读接口。串口参数由 CLI --port 指定，轨迹目录由 --out 指定。

| 发布端 | 消息类型 | QoS |
|---|---|---|
| /n10/points | sensor_msgs/msg/PointCloud2 | Best Effort、Volatile、Keep Last 1 |
| /n10/path | nav_msgs/msg/Path | Reliable、Volatile、Keep Last 1 |
| /tf | tf2_msgs/msg/TFMessage | tf2_ros 动态默认 QoS |

节点提供 /n10/reset_origin（std_srvs/srv/Trigger）。下一个扫描建立新原点，路径从新段重新绘制，CSV 的 segment 递增。服务在默认互斥回调组，由采集循环非阻塞处理。

n10_usb 节点发布 n10_odom 到 n10_laser 的动态变换。变换与点云共用主机接收时间戳，路径使用同一坐标原点。点云由雷达角度和距离转换为 XY 平面上的 XYZ 点。估计失配时保留上一个通过匹配的位姿，CSV 的 accepted 写 0。

脚本直接运行，不需要构建 ROS 包。串口、匹配、发布与 CSV 同步在采集线程。依赖 rclpy、geometry_msgs、nav_msgs、sensor_msgs、sensor_msgs_py、std_msgs、std_srvs、tf2_ros、NumPy、SciPy、pyserial。RViz2 使用 n10_odom 作为固定坐标系。
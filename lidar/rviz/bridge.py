#!/usr/bin/env python3
"""Publish DP2 N10 point clouds and estimated paths to ROS 2 for RViz."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import math
from pathlib import Path as FilePath
import sys
import threading

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

sys.path.insert(0, str(FilePath(__file__).resolve().parents[1]))
from n10_live import LidarOdometry, scan_stream


ODOM_FRAME = "n10_odom"
LIDAR_FRAME = "n10_laser"


class N10Bridge(Node):
    def __init__(self):
        super().__init__("n10_usb")
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.points = self.create_publisher(PointCloud2, "/n10/points", sensor_qos)
        self.path = self.create_publisher(Path, "/n10/path", QoSProfile(depth=1))
        self.tf = TransformBroadcaster(self)
        self.history = deque(maxlen=4000)
        self.reset_pending = False
        self.reset_service = self.create_service(Trigger, "/n10/reset_origin", self.reset_origin)

    def reset_origin(self, _request, response):
        self.reset_pending = True
        response.success = True
        response.message = "The next scan establishes a new trajectory segment."
        return response

    def publish_scan(self, xy, pose, accepted):
        stamp = self.get_clock().now().to_msg()
        odom_header = Header(stamp=stamp, frame_id=ODOM_FRAME)
        laser_header = Header(stamp=stamp, frame_id=LIDAR_FRAME)
        x, y, yaw = map(float, pose)
        tf = TransformStamped(header=odom_header, child_frame_id=LIDAR_FRAME)
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.z = math.sin(yaw / 2)
        tf.transform.rotation.w = math.cos(yaw / 2)
        self.tf.sendTransform(tf)

        xyz = np.zeros((len(xy), 3), dtype=np.float32)
        xyz[:, :2] = xy
        self.points.publish(create_cloud_xyz32(laser_header, xyz))

        if accepted:
            endpoint = self.history[-1].pose if self.history else None
            if endpoint is None or (
                math.hypot(x - endpoint.position.x, y - endpoint.position.y) >= 0.004
                or abs(tf.transform.rotation.z - endpoint.orientation.z) >= 0.003
            ):
                point = PoseStamped(header=odom_header)
                point.pose.position.x = x
                point.pose.position.y = y
                point.pose.orientation = tf.transform.rotation
                self.history.append(point)
        self.path.publish(Path(header=odom_header, poses=list(self.history)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--out", required=True, type=FilePath)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=[])
    node = N10Bridge()
    odometry = LidarOdometry()
    stop = threading.Event()
    start = None
    segment = 0
    scans = scan_stream(args.port, stop)
    try:
        with (args.out / "trajectory.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("utc", "time_s", "segment", "x_m", "y_m", "yaw_deg", "accepted",
                             "overlap", "rmse_m", "points", "samples"))
            stream.flush()
            for tick, xy, measured in scans:
                if not rclpy.ok():
                    break
                rclpy.spin_once(node, timeout_sec=0.0)
                if node.reset_pending:
                    node.reset_pending = False
                    node.history.clear()
                    odometry = LidarOdometry()
                    segment += 1
                start = tick if start is None else start
                pose, accepted, overlap, error = odometry.update(xy)
                node.publish_scan(xy, pose, accepted)
                writer.writerow((datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                 f"{tick - start:.3f}", segment, f"{pose[0]:.6f}", f"{pose[1]:.6f}",
                                 f"{math.degrees(pose[2]):.4f}", int(accepted),
                                 f"{overlap:.3f}", f"{error:.4f}", len(xy), measured))
                stream.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        scans.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

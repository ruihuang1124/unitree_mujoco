#!/usr/bin/python3
import argparse
import json
import select
import socket
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish Unitree MuJoCo debug UDP frames as ROS2 TF."
    )
    parser.add_argument("--bind", default="127.0.0.1", help="UDP bind host, default 127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=16001, help="UDP port from unitree_mujoco --debug_rviz")
    parser.add_argument("--root-frame", default="mujoco_world", help="RViz fixed/root frame")
    parser.add_argument("--frame-prefix", default="", help="Prefix added to dynamic MuJoCo TF frame names")
    parser.add_argument("--status-rate", type=float, default=1.0, help="Console status print rate in Hz")
    return parser.parse_args()


def sanitize_frame_id(name):
    cleaned = []
    for char in name:
        if char.isalnum() or char in ("_", "-", "/"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("/") or "frame"


def make_transform(parent, child, position, quaternion_xyzw, stamp):
    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = parent
    transform.child_frame_id = child
    transform.transform.translation.x = float(position[0])
    transform.transform.translation.y = float(position[1])
    transform.transform.translation.z = float(position[2])
    transform.transform.rotation.x = float(quaternion_xyzw[0])
    transform.transform.rotation.y = float(quaternion_xyzw[1])
    transform.transform.rotation.z = float(quaternion_xyzw[2])
    transform.transform.rotation.w = float(quaternion_xyzw[3])
    return transform


def make_udp_socket(bind_host, udp_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_host, udp_port))
    sock.setblocking(False)
    return sock


def main():
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("mujoco_debug_tf")
    tf_broadcaster = TransformBroadcaster(node)
    static_broadcaster = StaticTransformBroadcaster(node)

    static_broadcaster.sendTransform(
        make_transform(
            args.root_frame,
            "world",
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            node.get_clock().now().to_msg(),
        )
    )

    sock = make_udp_socket(args.bind, args.udp_port)
    node.get_logger().info(
        f"Listening on {args.bind}:{args.udp_port}, publishing MuJoCo TF under {args.root_frame}"
    )

    frame_counts = {}
    last_status_wall = 0.0
    last_packet_wall = 0.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            readable, _, _ = select.select([sock], [], [], 0.05)
            if not readable:
                continue

            data, _ = sock.recvfrom(65535)
            try:
                packet = json.loads(data.decode("utf-8"))
                frames = packet["frames"]
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                node.get_logger().warn(f"Skipping malformed UDP packet: {exc}")
                continue

            stamp = node.get_clock().now().to_msg()
            transforms = []
            for frame in frames:
                raw_name = frame["name"]
                if raw_name == "world":
                    continue
                name = sanitize_frame_id(args.frame_prefix + raw_name)
                transforms.append(
                    make_transform("world", name, frame["p"], frame["q_xyzw"], stamp)
                )
                frame_counts[name] = frame_counts.get(name, 0) + 1
            if transforms:
                tf_broadcaster.sendTransform(transforms)

            now_wall = time.monotonic()
            last_packet_wall = now_wall
            if args.status_rate > 0.0 and now_wall - last_status_wall >= 1.0 / args.status_rate:
                last_status_wall = now_wall
                status = ", ".join(f"{name}:count={count}" for name, count in sorted(frame_counts.items()))
                node.get_logger().info(f"sim_time={packet.get('sim_time_s', 0.0):.3f} {status}")
    except KeyboardInterrupt:
        pass
    finally:
        _ = last_packet_wall
        sock.close()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

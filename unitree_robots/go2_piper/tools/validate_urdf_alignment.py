#!/usr/bin/env python3
"""Validate the canonical Go2-Piper MJCF against its training URDF."""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_URDF = Path(
    "/home/ray/colcon_ws/src/quadruped_control_ros2/robot_description/"
    "go2_arm_description/xacro/go2_arm_front_nuc_low.urdf"
)
DEFAULT_MJCF = Path(__file__).resolve().parents[1] / "go2_piper_fake_ee.xml"

URDF_TO_MJCF_BODY = {
    "trunk": "base_link",
    "base_link": "arm_base_link",
}


def numbers(value: str | None, count: int, default: float = 0.0) -> list[float]:
    if value is None:
        return [default] * count
    result = [float(item) for item in value.split()]
    if len(result) != count:
        raise ValueError(f"expected {count} values, got {value!r}")
    return result


def quaternion_from_rpy(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def close_vector(actual: list[float], expected: list[float], tolerance: float) -> bool:
    return len(actual) == len(expected) and max(
        (abs(a - b) for a, b in zip(actual, expected)), default=0.0
    ) <= tolerance


def same_rotation(actual: list[float], expected: list[float], tolerance: float) -> bool:
    norm_a = math.sqrt(sum(value * value for value in actual))
    norm_b = math.sqrt(sum(value * value for value in expected))
    dot = abs(sum(a * b for a, b in zip(actual, expected)) / (norm_a * norm_b))
    return 1.0 - min(dot, 1.0) <= tolerance


def collect_defaults(root: ET.Element) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {}

    def visit(
        node: ET.Element,
        inherited_joint: dict[str, str],
        inherited_motor: dict[str, str],
        inherited_geom: dict[str, str],
    ) -> None:
        joint = dict(inherited_joint)
        motor = dict(inherited_motor)
        geom = dict(inherited_geom)
        joint_node = node.find("joint")
        motor_node = node.find("motor")
        geom_node = node.find("geom")
        if joint_node is not None:
            joint.update(joint_node.attrib)
        if motor_node is not None:
            motor.update(motor_node.attrib)
        if geom_node is not None:
            geom.update(geom_node.attrib)
        class_name = node.get("class")
        if class_name:
            result[class_name] = {
                "joint": dict(joint),
                "motor": dict(motor),
                "geom": dict(geom),
            }
        for child in node.findall("default"):
            visit(child, joint, motor, geom)

    default_root = root.find("default")
    if default_root is not None:
        visit(default_root, {}, {}, {})
    return result


def resolved_attributes(
    node: ET.Element,
    kind: str,
    defaults: dict[str, dict[str, dict[str, str]]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    class_name = node.get("class")
    if class_name in defaults:
        result.update(defaults[class_name][kind])
    result.update(node.attrib)
    return result


def primitive_collision(
    collision: ET.Element,
) -> tuple[str, list[float], list[float], list[float]] | None:
    origin = collision.find("origin")
    pos = numbers(origin.get("xyz") if origin is not None else None, 3)
    quat = quaternion_from_rpy(
        numbers(origin.get("rpy") if origin is not None else None, 3)
    )
    geometry = collision.find("geometry")
    box = geometry.find("box")
    if box is not None:
        return "box", pos, quat, [value / 2.0 for value in numbers(box.get("size"), 3)]
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return "cylinder", pos, quat, [
            float(cylinder.get("radius")),
            float(cylinder.get("length")) / 2.0,
        ]
    sphere = geometry.find("sphere")
    if sphere is not None:
        return "sphere", pos, quat, [float(sphere.get("radius"))]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    urdf_root = ET.parse(args.urdf).getroot()
    mjcf_root = ET.parse(args.mjcf).getroot()
    urdf_links = {node.get("name"): node for node in urdf_root.findall("link")}
    urdf_joints = {node.get("name"): node for node in urdf_root.findall("joint")}
    mjcf_bodies = {node.get("name"): node for node in mjcf_root.iter("body")}
    mjcf_joints = {node.get("name"): node for node in mjcf_root.iter("joint") if node.get("name")}
    defaults = collect_defaults(mjcf_root)
    failures: list[str] = []

    mapped_links: list[tuple[str, str]] = []
    for urdf_name, urdf_link in urdf_links.items():
        inertial = urdf_link.find("inertial")
        if inertial is None:
            continue
        mjcf_name = URDF_TO_MJCF_BODY.get(urdf_name, urdf_name)
        mapped_links.append((urdf_name, mjcf_name))
        body = mjcf_bodies.get(mjcf_name)
        if body is None:
            failures.append(f"missing MJCF body for URDF link {urdf_name}: {mjcf_name}")
            continue
        mjcf_inertial = body.find("inertial")
        if mjcf_inertial is None:
            failures.append(f"{mjcf_name}: missing inertial")
            continue

        origin = inertial.find("origin")
        expected_pos = numbers(origin.get("xyz") if origin is not None else None, 3)
        expected_rpy = numbers(origin.get("rpy") if origin is not None else None, 3)
        if not close_vector(expected_rpy, [0.0, 0.0, 0.0], args.tolerance):
            failures.append(f"{urdf_name}: nonzero inertial RPY is not represented by this audit")
        expected_mass = float(inertial.find("mass").get("value"))
        tensor = inertial.find("inertia")
        expected_inertia = [
            float(tensor.get(name))
            for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ]
        actual_mass = float(mjcf_inertial.get("mass"))
        actual_pos = numbers(mjcf_inertial.get("pos"), 3)
        actual_inertia = numbers(mjcf_inertial.get("fullinertia"), 6)
        if abs(actual_mass - expected_mass) > args.tolerance:
            failures.append(f"{mjcf_name}: mass {actual_mass} != {expected_mass}")
        if not close_vector(actual_pos, expected_pos, args.tolerance):
            failures.append(f"{mjcf_name}: inertial position mismatch")
        if not close_vector(actual_inertia, expected_inertia, args.tolerance):
            failures.append(f"{mjcf_name}: inertia tensor mismatch")

    for joint_name, urdf_joint in urdf_joints.items():
        joint_type = urdf_joint.get("type")
        if joint_type not in {"revolute", "prismatic", "fixed"}:
            continue
        child_name = urdf_joint.find("child").get("link")
        if child_name == "trunk":
            continue
        mjcf_body_name = URDF_TO_MJCF_BODY.get(child_name, child_name)
        body = mjcf_bodies.get(mjcf_body_name)
        if body is None:
            continue
        origin = urdf_joint.find("origin")
        expected_pos = numbers(origin.get("xyz") if origin is not None else None, 3)
        expected_quat = quaternion_from_rpy(
            numbers(origin.get("rpy") if origin is not None else None, 3)
        )
        if not close_vector(numbers(body.get("pos"), 3), expected_pos, args.tolerance):
            failures.append(f"{joint_name}: body position does not match URDF origin")
        actual_quat = (
            numbers(body.get("quat"), 4)
            if body.get("quat") is not None
            else [1.0, 0.0, 0.0, 0.0]
        )
        if not same_rotation(actual_quat, expected_quat, 1.0e-10):
            failures.append(f"{joint_name}: body rotation does not match URDF origin")
        if joint_type == "fixed":
            continue
        mjcf_joint = mjcf_joints.get(joint_name)
        if mjcf_joint is None:
            failures.append(f"missing MJCF joint {joint_name}")
            continue
        attrs = resolved_attributes(mjcf_joint, "joint", defaults)
        expected_axis = numbers(urdf_joint.find("axis").get("xyz"), 3)
        expected_range = [
            float(urdf_joint.find("limit").get("lower")),
            float(urdf_joint.find("limit").get("upper")),
        ]
        if not close_vector(numbers(attrs.get("axis"), 3), expected_axis, args.tolerance):
            failures.append(f"{joint_name}: axis mismatch")
        if not close_vector(numbers(attrs.get("range"), 2), expected_range, args.tolerance):
            failures.append(f"{joint_name}: limit mismatch")

    actuators = {
        node.get("joint"): node for node in mjcf_root.findall("actuator/motor")
    }
    for joint_name, actuator in actuators.items():
        urdf_joint = urdf_joints[joint_name]
        expected_effort = float(urdf_joint.find("limit").get("effort"))
        attrs = resolved_attributes(actuator, "motor", defaults)
        control_range = numbers(attrs.get("ctrlrange"), 2)
        if not close_vector(control_range, [-expected_effort, expected_effort], args.tolerance):
            failures.append(f"{joint_name}: actuator effort range mismatch")

    collision_count = 0
    for urdf_name, urdf_link in urdf_links.items():
        expected_collisions = [
            collision
            for node in urdf_link.findall("collision")
            if (collision := primitive_collision(node)) is not None
        ]
        if not expected_collisions:
            continue
        mjcf_name = URDF_TO_MJCF_BODY.get(urdf_name, urdf_name)
        body = mjcf_bodies.get(mjcf_name)
        if body is None:
            continue
        actual_collisions = []
        for geom in body.findall("geom"):
            if geom.get("class") not in {"collision", "foot"}:
                continue
            attrs = resolved_attributes(geom, "geom", defaults)
            geom_type = attrs.get("type", "sphere")
            size_count = {"box": 3, "cylinder": 2, "sphere": 1}.get(geom_type)
            if size_count is None:
                continue
            actual_collisions.append(
                (
                    geom_type,
                    numbers(attrs.get("pos"), 3),
                    numbers(attrs.get("quat"), 4)
                    if attrs.get("quat") is not None
                    else [1.0, 0.0, 0.0, 0.0],
                    numbers(attrs.get("size"), size_count),
                )
            )
        unmatched = list(actual_collisions)
        for expected_type, expected_pos, expected_quat, expected_size in expected_collisions:
            match_index = next(
                (
                    index
                    for index, (actual_type, actual_pos, actual_quat, actual_size) in enumerate(unmatched)
                    if actual_type == expected_type
                    and close_vector(actual_pos, expected_pos, args.tolerance)
                    and same_rotation(actual_quat, expected_quat, 1.0e-10)
                    and close_vector(actual_size, expected_size, args.tolerance)
                ),
                None,
            )
            if match_index is None:
                failures.append(
                    f"{mjcf_name}: missing or mismatched {expected_type} collision"
                )
            else:
                unmatched.pop(match_index)
                collision_count += 1
        if unmatched:
            failures.append(
                f"{mjcf_name}: {len(unmatched)} extra primitive collision geom(s)"
            )

    fake_ee = mjcf_bodies.get("fake_ee")
    if fake_ee is None or fake_ee.find("inertial") is not None:
        failures.append("fake_ee must exist as a massless tracking frame")
    elif any(float(geom.get("density", "1000")) != 0.0 for geom in fake_ee.findall("geom")):
        failures.append("all fake_ee visualization geoms must have zero density")

    urdf_mass = sum(
        float(urdf_links[name].find("inertial/mass").get("value"))
        for name, _ in mapped_links
    )
    mjcf_mass = sum(
        float(mjcf_bodies[name].find("inertial").get("mass"))
        for _, name in mapped_links
    )
    if abs(urdf_mass - mjcf_mass) > args.tolerance:
        failures.append(f"total mass {mjcf_mass} != URDF {urdf_mass}")

    if failures:
        print("Go2-Piper URDF/MJCF alignment FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(
        f"Go2-Piper URDF/MJCF alignment passed: {len(mapped_links)} rigid bodies, "
        f"20 joints, {collision_count} primitive collisions, "
        f"total mass {mjcf_mass:.12f} kg."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

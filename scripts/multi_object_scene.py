"""로봇 앞 최대 5개 낙하 배치와 한 종류 도형의 카메라 순차 운반을 확인한다."""

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image

from contact_grasp_probe import CONFIG, Probe, grasp_plan
from rgbd_camera import camera_for_angle, capture, components, depth_metres, estimate_many, rgb_crop, save_observation, world_points


LAYOUT = {
    "version": "multi-scene-v3-sequential-pick",
    "region": {"x": [0.38, 0.65], "y": [-0.23, 0.23]},
    "gap_m": 0.03,
    "table_bounds": {"x": [-0.60, 0.95], "y": [-0.60, 0.60]},
    "bin_centers": {"cuboid": [-0.28, 0.30], "cylinder": [-0.28, -0.30]},
    "bin_inner_size_m": [0.30, 0.24],
    "drop_clearance_m": 0.05,
    "max_placement_attempts": 20,
    "max_scene_attempts": 20,
}


def validate_counts(cuboids, cylinders):
    # 잘못된 개수로 장면을 만들지 않는다. 합계는 최대 5개다.
    if any(type(n) is not int or n < 0 for n in (cuboids, cylinders)) or not 1 <= cuboids + cylinders <= 5:
        raise ValueError("도형 개수는 각각 0 이상, 합계 1~5개여야 합니다")


def sample_spawns(cuboids, cylinders, seed, region):
    validate_counts(cuboids, cylinders)
    rng = random.Random(seed)
    spawns = []
    for shape in ["cuboid"] * cuboids + ["cylinder"] * cylinders:
        for _ in range(LAYOUT["max_placement_attempts"]):
            yaw = rng.uniform(-math.pi, math.pi)
            # 회전한 직육면체의 모서리까지 감싸는 가로·세로 반길이를 계산한다.
            half = [(abs(math.cos(yaw)) * 0.06 + abs(math.sin(yaw)) * 0.04) / 2,
                    (abs(math.sin(yaw)) * 0.06 + abs(math.cos(yaw)) * 0.04) / 2] if shape == "cuboid" else [0.026 * (abs(math.cos(yaw)) + abs(math.sin(yaw)))] * 2
            # 원기둥도 PyBullet의 회전된 충돌 AABB와 1mm 여유를 고려해 보수적으로 배치한다.
            limits = [(region[k][0] + half[a] + 0.002, region[k][1] - half[a] - 0.002) for a, k in enumerate(("x", "y"))]
            if any(low > high for low, high in limits):
                break
            xy = [rng.uniform(low, high) for low, high in limits]
            # 한 축 이상에서 물체 사이에 손가락을 위한 3cm 간격을 남긴다.
            if all(any(abs(xy[a] - other["xy"][a]) >= half[a] + other["half_xy"][a] + LAYOUT["gap_m"] + 0.002
                       for a in (0, 1)) for other in spawns):
                spawns.append({"shape": shape, "xy": xy, "yaw": yaw, "half_xy": half})
                break
        else:
            raise ValueError("placement_exhausted")
    if len(spawns) != cuboids + cylinders:
        raise ValueError("placement_exhausted")
    return spawns


def box(half, position, mass, color):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(mass, collision, visual, position)


def scene_camera(layout):
    # 기존 단일 물체 카메라는 그대로 두고 새 구역 전체를 보는 설정을 별도로 만든다.
    camera = camera_for_angle(65)
    center = [(layout["region"][k][0] + layout["region"][k][1]) / 2 for k in ("x", "y")]
    camera["eye"] = [camera["eye"][0] + center[0] - 0.5, camera["eye"][1] + center[1], camera["eye"][2]]
    camera["target"] = [*center, CONFIG["table_top_m"]]
    camera["roi_xy_m"] = [layout["region"][k] for k in ("x", "y")]
    return camera


def add_camera_visual(camera):
    # 촬영 위치·방향 그대로 작은 본체와 렌즈를 표시한다. 충돌 형상은 만들지 않는다.
    eye = np.asarray(camera["eye"], dtype=float)
    direction = np.asarray(camera["target"], dtype=float) - eye
    direction /= np.linalg.norm(direction)
    yaw = math.atan2(direction[1], direction[0])
    pitch = math.acos(float(direction[2]))
    rotation = p.getQuaternionFromEuler([0, pitch, yaw])
    housing = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.02, 0.018], rgbaColor=[0.1, 0.12, 0.15, 1])
    lens = p.createVisualShape(p.GEOM_CYLINDER, radius=0.012, length=0.018, rgbaColor=[0.1, 0.65, 0.9, 1])
    bodies = [p.createMultiBody(baseMass=0, baseVisualShapeIndex=housing, basePosition=eye.tolist(), baseOrientation=rotation),
              p.createMultiBody(baseMass=0, baseVisualShapeIndex=lens, basePosition=(eye + direction * 0.026).tolist(),
                                baseOrientation=rotation)]
    # 실제 시야각·가로세로비로 화면 네 모서리의 광선을 구하고 상판 높이와 만나는 점을 찾는다.
    view = np.asarray(p.computeViewMatrix(camera["eye"], camera["target"], camera["up"])).reshape(4, 4, order="F")
    projection = np.asarray(p.computeProjectionMatrixFOV(camera["fov_deg"], camera["width"] / camera["height"],
                                                        camera["near_m"], camera["far_m"])).reshape(4, 4, order="F")
    inverse = np.linalg.inv(projection @ view)
    footprint = []
    for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        point = inverse @ np.array([x, y, 1, 1])
        ray = point[:3] / point[3] - eye
        scale = (camera["table_top_m"] - eye[2]) / ray[2]
        footprint.append((eye + scale * ray).tolist())
    lines = []
    for index, corner in enumerate(footprint):
        # 면을 채우지 않는다. 네 광선과 상판 위 사각형 테두리만 그린다.
        lines.append(p.addUserDebugLine(eye.tolist(), corner, [0.95, 0.75, 0.1], lineWidth=1))
        lines.append(p.addUserDebugLine(corner, footprint[(index + 1) % 4], [0.95, 0.75, 0.1], lineWidth=2))
    p.addUserDebugText("RGB-D", (eye + [0, 0, 0.05]).tolist(), [0.1, 0.5, 0.9], textSize=1.2)
    return {"bodies": bodies, "lines": lines, "footprint": footprint}


def build_scene(spawns, layout):
    # reset은 초기 장면 재생성에만 쓴다. 운반 중 순간이동에는 쓰지 않는다.
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)
    p.setTimeStep(CONFIG["dt_s"])
    p.setPhysicsEngineParameter(numSolverIterations=100)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    bounds = layout["table_bounds"]
    table = box([(bounds[k][1] - bounds[k][0]) / 2 for k in ("x", "y")] + [0.02],
                [(bounds[k][1] + bounds[k][0]) / 2 for k in ("x", "y")] + [0.28], 0, [0.65, 0.65, 0.65, 1])
    # 생성 구역은 청록색 얇은 표시로 구별한다. 충돌이 없어 물체는 기존 상판에 닿는다.
    region = layout["region"]
    area_shape = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[(region[k][1] - region[k][0]) / 2 for k in ("x", "y")] + [0.0001],
                                    rgbaColor=[0.25, 0.55, 0.58, 1])
    spawn_area = p.createMultiBody(baseMass=0, baseVisualShapeIndex=area_shape,
                                  basePosition=[(region[k][1] + region[k][0]) / 2 for k in ("x", "y")] + [0.3001])
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.30], useFixedBase=True)
    infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    movable = sorted((i for i in infos if i[2] != p.JOINT_FIXED), key=lambda i: i[3])
    home = [0, -0.6, 0, -2.0, 0, 1.6, 0.8, 0.04, 0.04]
    for info, value in zip(movable, home):
        p.resetJointState(robot, info[0], value)
        p.setJointMotorControl2(robot, info[0], p.POSITION_CONTROL, targetPosition=value,
                                force=20 if info[2] == p.JOINT_PRISMATIC else info[10])
    # Panda URDF의 mimic 관계를 물리 기어로 연결해 두 손가락이 함께 열리고 닫히게 한다.
    # 로봇의 손가락끼리만 연결한다. 물체를 로봇에 붙이는 제약은 만들지 않는다.
    fingers = [next(i[0] for i in infos if i[1].decode() == f"panda_finger_joint{n}") for n in (1, 2)]
    gear = p.createConstraint(robot, fingers[0], robot, fingers[1], p.JOINT_GEAR,
                              [1, 0, 0], [0, 0, 0], [0, 0, 0])
    p.changeConstraint(gear, gearRatio=-1, erp=0.1, maxForce=50)
    bins = {}
    # 로봇 뒤(-X)에 바닥 1개·벽 4개로 된 상자를 만든다.
    hx, hy = [n / 2 for n in layout["bin_inner_size_m"]]
    thickness, wall, floor_top = 0.01, CONFIG["bin_wall_height_m"], 0.31
    for shape, (x, y) in layout["bin_centers"].items():
        color = [0.15, 0.35, 0.9, 1] if shape == "cuboid" else [0.15, 0.7, 0.3, 1]
        floor = box([hx + thickness, hy + thickness, 0.005], [x, y, 0.305], 0, color)
        parts = [floor]
        for sign in (-1, 1):
            parts.append(box([thickness / 2, hy + thickness, wall / 2],
                             [x + sign * (hx + thickness / 2), y, floor_top + wall / 2], 0, color))
            parts.append(box([hx, thickness / 2, wall / 2],
                             [x, y + sign * (hy + thickness / 2), floor_top + wall / 2], 0, color))
        # 한 상자에 최대 5개를 놓을 자리다. 실제 운반 단계에서 각 자리의 도달을 검증한다.
        slots = [[x + dx, y + dy] for dy in (-0.05, 0.05) for dx in (-0.09, 0, 0.09)]
        # 정적 IK 검사에서 손바닥이 벽에 닿은 바깥 모서리 하나를 제외해 다섯 자리를 남긴다.
        del slots[5 if shape == "cuboid" else 2]
        bins[shape] = {"center": [x, y], "floor": floor, "parts": parts, "slots": slots, "floor_top": floor_top}
    camera = scene_camera(layout)
    # 물체 없는 기준 영상도 같은 카메라·로봇 대기 자세에서 촬영한다.
    empty = capture(camera)
    objects = []
    for spawn in spawns:
        z = CONFIG["table_top_m"] + 0.03 + layout["drop_clearance_m"]
        if spawn["shape"] == "cuboid":
            body = box([n / 2 for n in CONFIG["block_size_m"]], [*spawn["xy"], z], 0.05, [0.9, 0.15, 0.1, 1])
        else:
            collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.025, height=0.06)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.06, rgbaColor=[0.9, 0.15, 0.1, 1])
            body = p.createMultiBody(0.05, collision, visual, [*spawn["xy"], z])
        # 수평 회전은 생성 시에만 지정한다. 이후 낙하·정착은 물리 계산에 맡긴다.
        p.resetBasePositionAndOrientation(body, [*spawn["xy"], z], p.getQuaternionFromEuler([0, 0, spawn["yaw"]]))
        p.changeDynamics(body, -1, lateralFriction=CONFIG["lateral_friction"])
        objects.append({**spawn, "body": body})
    return {"robot": robot, "table": table, "spawn_area_visual": spawn_area,
            "objects": objects, "bins": bins, "layout": layout,
            "home": home, "camera": camera, "empty": empty}


def observe_scene(scene, output):
    output.mkdir(parents=True, exist_ok=True)
    # 재관측에서 영역 수가 줄어도 예전 crop·mask가 남지 않도록 이 프로그램의 번호 파일만 정리한다.
    for prefix in ("crop", "mask"):
        for path in output.glob(f"{prefix}_*.png"):
            if path.stem[len(prefix) + 1:].isdigit():
                path.unlink()
    camera = scene["camera"]
    observation = capture(camera)
    empty = scene["empty"]
    points = world_points(observation)
    # 검증용 objects 목록을 읽지 않고 RGB-D와 관측 구역만으로 영역을 찾는다.
    difference = depth_metres(empty["depth"], camera) - depth_metres(observation["depth"], camera)
    mask = difference > camera["depth_difference_m"]
    for axis, (low, high) in enumerate(camera["roi_xy_m"]):
        mask &= (points[..., axis] > low) & (points[..., axis] < high)
    mask &= (points[..., 2] > camera["table_top_m"] + 0.008) & (observation["depth"] < 1)
    # 원본 RGB·float 깊이는 기존 저장 함수를 재사용한다. 클래스·파지 기하는 아직 계산하지 않는다.
    save_observation(output, observation, empty, {"valid": False}, camera)
    regions = []
    groups = sorted(components(mask), key=lambda g: min(u for _, u in g))
    for index, group in enumerate(groups):
        pixels = np.asarray(group)
        region_mask = np.zeros_like(mask)
        region_mask[pixels[:, 0], pixels[:, 1]] = True
        rgb_crop(observation["rgb"], region_mask, camera["crop_size"]).save(output / f"crop_{index + 1}.png")
        Image.fromarray(region_mask.astype(np.uint8) * 255).save(output / f"mask_{index + 1}.png")
        regions.append({"index": index + 1, "pixels": len(group),
                        "bbox_uv": [int(pixels[:, 1].min()), int(pixels[:, 0].min()),
                                    int(pixels[:, 1].max()), int(pixels[:, 0].max())]})
    result = {"observed_regions": len(regions), "regions": regions, "camera": camera,
              "classification_source": "not_implemented", "grasp_geometry_computed": False}
    (output / "observation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class SceneProbe(Probe):
    def command_tcp(self, point, rotation):
        # 현재 자세 가까이에서 IK를 풀고 모터에 전달한다. 실제 움직임은 step에서 계산된다.
        rest = [math.atan2(point[1], point[0]), -0.6, 0, -2.0, 0, 1.6, 0.8, 0.04, 0.04]
        solution = p.calculateInverseKinematics(self.robot, self.tcp, point, rotation,
                                               lowerLimits=[i[8] for i in self.movable],
                                               upperLimits=[i[9] for i in self.movable],
                                               jointRanges=[i[9] - i[8] for i in self.movable],
                                               restPoses=rest, maxNumIterations=100, residualThreshold=1e-7)
        values = {info[0]: value for info, value in zip(self.movable, solution)}
        if any(not self.infos[i][8] <= values[i] <= self.infos[i][9] for i in self.arms):
            raise RuntimeError("ik_joint_limit")
        self.arm_targets = [values[i] for i in self.arms]
        self.hold_arm()

    def move(self, stage, position, seconds):
        # 손끝 경로를 작은 구간으로 나누어 현재 자세 가까이에서 IK를 다시 푼다.
        self.start_stage(stage)
        initial, rotation = p.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)[4:6]
        start_q = np.asarray(rotation)
        target_q = np.asarray(self.orientation)
        # 같은 회전을 나타내는 두 부호 중 짧은 회전 방향을 선택한다.
        if np.dot(start_q, target_q) < 0:
            target_q = -target_q
        ticks = round(seconds / CONFIG["dt_s"])
        for tick in range(1, ticks + 1):
            if tick % 4 == 0 or tick == ticks:
                t = tick / ticks
                blend = t * t * (3 - 2 * t)
                point = [a + blend * (b - a) for a, b in zip(initial, position)]
                q = start_q + blend * (target_q - start_q)
                q /= np.linalg.norm(q)
                self.command_tcp(point, q.tolist())
            self.step()
        # 마지막 실제 자세에서 다시 풀어 모터의 추종 지연을 줄이고 기존 오차 기준으로 검증한다.
        super().move(stage + "_FINAL", position, 0.3)

    def carry_arc(self, direction, seconds=4.0):
        self.start_stage("CARRY_ARC")
        position, rotation = p.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)[4:6]
        start_angle = math.atan2(position[1], position[0])
        start_radius = math.hypot(position[0], position[1])
        start_yaw = p.getEulerFromQuaternion(rotation)[2]
        end_angle = direction * 2.4
        ticks = round(seconds / CONFIG["dt_s"])
        for tick in range(1, ticks + 1):
            if tick % 4 == 0 or tick == ticks:
                t = tick / ticks
                # 호 전체의 출발·도착에서만 감속한다. 중간 각도에서는 멈추지 않는다.
                blend = t * t * (3 - 2 * t)
                angle = start_angle + blend * (end_angle - start_angle)
                transition = min(t / 0.25, 1.0)
                transition = transition * transition * (3 - 2 * transition)
                # 초기 물체 위치·손끝 방향에서 운반 반경과 방향으로 부드럽게 연결한다.
                radius = start_radius + transition * (0.47 - start_radius)
                yaw = angle + (start_yaw - start_angle) * (1 - transition)
                point = [radius * math.cos(angle), radius * math.sin(angle), 0.65]
                self.command_tcp(point, p.getQuaternionFromEuler([math.pi, 0, yaw]))
            self.step()
        self.orientation = p.getQuaternionFromEuler([math.pi, 0, end_angle])
        # 최종 위치 검증과 모든 step의 충돌·낙하 감시는 그대로 유지한다.
        super().move("CARRY_ARC_FINAL", [0.47 * math.cos(end_angle), 0.47 * math.sin(end_angle), 0.65], 0.3)


def scene_probe(scene, log, gui):
    # 기존 접촉 제어를 새 장면에 연결한다. 초기화를 다시 호출하지 않는다.
    probe = SceneProbe(gui, log, pose_source="camera")
    probe.robot, probe.table = scene["robot"], scene["table"]
    probe.infos = [p.getJointInfo(probe.robot, i) for i in range(p.getNumJoints(probe.robot))]
    names = {i[1].decode(): i for i in probe.infos}
    probe.arms = [names[f"panda_joint{i}"][0] for i in range(1, 8)]
    probe.fingers = [names[f"panda_finger_joint{i}"][0] for i in (1, 2)]
    probe.tcp = next(i[0] for i in probe.infos if i[12].decode() == "panda_grasptarget")
    probe.hand = next(i[0] for i in probe.infos if i[12].decode() == "panda_hand")
    probe.movable = sorted((i for i in probe.infos if i[2] != p.JOINT_FIXED), key=lambda i: i[3])
    probe.arm_forces = [probe.infos[i][10] for i in probe.arms]
    probe.arm_targets = scene["home"][:7]
    probe.bins = {name: {**b, "inner_half_size_m": [s / 2 for s in scene["layout"]["bin_inner_size_m"]]}
                  for name, b in scene["bins"].items()}
    probe.supports = [scene["table"]] + [part for b in scene["bins"].values() for part in b["parts"]]
    for finger in probe.fingers:
        p.changeDynamics(probe.robot, finger, lateralFriction=CONFIG["lateral_friction"])
    return probe


def return_to_wait(probe, scene):
    # 모터로 기준 영상과 같은 대기 자세로 돌아온다. 관절을 순간이동시키지 않는다.
    probe.carry_reference = None
    probe.gripper(0.04)
    initial = [p.getJointState(probe.robot, i)[0] for i in probe.arms]
    probe.start_stage("RETURN_WAIT")
    for tick in range(1, 721):
        t = tick / 720
        blend = t * t * (3 - 2 * t)
        probe.arm_targets = [a + blend * (b - a) for a, b in zip(initial, scene["home"][:7])]
        probe.hold_arm()
        probe.step()
    probe.wait("WAIT_SETTLE", 0.5)
    if max(abs(p.getJointState(probe.robot, i)[0] - target) for i, target in zip(probe.arms, scene["home"][:7])) > 0.01:
        raise RuntimeError("wait_pose_not_reached")


def sort_scene(scene, shape, output, gui=False):
    if shape not in ("cuboid", "cylinder") or any(o["shape"] != shape for o in scene["objects"]):
        raise ValueError("CNN 연결 전에는 수동 지정한 한 종류만 운반합니다")
    output.mkdir(parents=True, exist_ok=True)
    report = {"success": False, "classification_source": "manual_single_shape", "scans": [], "picks": [], "failure_reason": ""}
    with (output / "motion.jsonl").open("w") as log:
        probe = scene_probe(scene, log, gui)
        # ID는 평가 대상의 대응에만 쓴다. 이동 좌표는 매번 카메라에서 구한다.
        remaining = {o["body"] for o in scene["objects"]}
        if not remaining:
            raise ValueError("운반 검증에는 물체가 하나 이상 필요합니다")
        probe.block = next(iter(remaining))
        try:
            for iteration in range(6):
                observation = capture(scene["camera"])
                candidates = estimate_many(observation, scene["empty"], scene["camera"], shape)
                scan_dir = output / f"scan_{iteration:02d}"
                scan_dir.mkdir(parents=True, exist_ok=True)
                save_observation(scan_dir, observation, scene["empty"], {"valid": False}, scene["camera"])
                scan = {"valid_targets": sum(g["valid"] for g in candidates),
                        "regions": [{k: v for k, v in g.items() if k != "mask"} for g in candidates]}
                report["scans"].append(scan)
                (scan_dir / "regions.json").write_text(json.dumps(scan, indent=2) + "\n")
                for index, candidate in enumerate(candidates):
                    if "mask" in candidate:
                        rgb_crop(observation["rgb"], candidate["mask"]).save(scan_dir / f"crop_{index + 1}.png")
                        Image.fromarray(candidate["mask"].astype(np.uint8) * 255).save(scan_dir / f"mask_{index + 1}.png")
                valid = [g for g in candidates if g["valid"]]
                if not valid:
                    report["success"] = not remaining
                    report["failure_reason"] = "" if not remaining else "no_valid_target"
                    break
                # 영상에서 윗면의 점이 많이 보이는 물체를 먼저 선택한다.
                geometry = max(valid, key=lambda g: g["top_pixels"])
                plan = grasp_plan(geometry)
                pick = {"grasp_plan": plan, "lift_success": False, "arrival_success": False}
                report["picks"].append(pick)
                matches = sorted((math.dist(geometry["center_xy_m"], p.getBasePositionAndOrientation(body)[0][:2]), body)
                                 for body in remaining)
                if not matches or matches[0][0] > 0.01:
                    raise RuntimeError("evaluation_target_unmatched")
                probe.block = matches[0][1]
                pick["evaluation_body_id"] = probe.block
                pick["evaluation_xy_error_m"] = matches[0][0]
                probe.supports = [scene["table"]] + [part for b in scene["bins"].values() for part in b["parts"]] + [o["body"] for o in scene["objects"] if o["body"] != probe.block]
                probe.drop_via_relative_motion = False
                x, y, z = plan["position_m"]
                probe.orientation = p.getQuaternionFromEuler([math.pi, 0, plan["yaw_rad"]])
                probe.gripper(plan["opening_per_finger_m"])
                probe.move("APPROACH", [x, y, 0.50], 1.5)
                probe.move("DESCEND", [x, y, z], 1.2)
                probe.gripper(0)
                probe.wait("CLOSE", 0.8)
                probe.move("LIFT", [x, y, z + 0.10], 1.5)
                probe.verify_hold(pick)
                if not pick["lift_success"]:
                    raise RuntimeError("lift_not_verified")
                probe.move("CLEARANCE", [x, y, 0.65], 1.0)
                # 몸통을 가로지르지 않도록 높은 위치에서 바깥쪽으로 돌아 뒤 상자로 간다.
                destination = scene["bins"][shape]["slots"][len(report["picks"]) - 1]
                direction = 1 if destination[1] > 0 else -1
                probe.carry_arc(direction)
                # 상자에서는 긴 변과 손끝 방향을 나란하게 맞춰 벽과 개방 공간을 확보한다.
                probe.orientation = p.getQuaternionFromEuler([math.pi, 0, direction * math.pi])
                probe.move("BIN_ABOVE", [*destination, 0.50], 1.0)
                # 바닥 가까이 내려놓는다. 실제 정착은 접촉과 속도로 따로 확인한다.
                release_z = max(z + CONFIG["bin_floor_thickness_m"],
                                scene["bins"][shape]["floor_top"] + CONFIG["bin_release_tcp_floor_clearance_m"])
                pick["release_tcp_z_m"] = release_z
                probe.move("BIN_PLACE", [*destination, release_z], 1.2)
                if probe.drop_via_relative_motion:
                    raise RuntimeError("drop_during_transport")
                probe.carry_reference = None
                # 놓을 때만 낮은 고정 개방 힘을 쓴다. 집는 힘·마찰을 추정하거나 자동 조절하지 않는다.
                p.setJointMotorControlArray(probe.robot, probe.fingers, p.POSITION_CONTROL,
                                            targetPositions=[0.04, 0.04], forces=[0.1, 0.1])
                probe.wait("RELEASE", 1.0)
                probe.move("RETREAT", [*destination, 0.65], 1.0)
                # 낙하 후 흔들림이 멎는 시간은 일정하지 않다. 최대 5초 안에 0.5초 연속 정착을 확인한다.
                probe.start_stage("VERIFY_ARRIVAL")
                consecutive = {name: 0 for name in probe.bins}
                for _ in range(round(5.0 / CONFIG["dt_s"])):
                    sample = probe.step()
                    for name in consecutive:
                        consecutive[name] = consecutive[name] + 1 if sample["bin_arrivals"][name] else 0
                    if any(n >= round(CONFIG["arrival_hold_s"] / CONFIG["dt_s"]) for n in consecutive.values()):
                        break
                # 지정한 상자 도착, 다른 상자 도착, 정착 실패를 구분한다.
                pick["bin_arrivals"] = {name: n >= round(CONFIG["arrival_hold_s"] / CONFIG["dt_s"])
                                        for name, n in consecutive.items()}
                pick["arrival_success"] = pick["bin_arrivals"][shape]
                if not pick["arrival_success"]:
                    raise RuntimeError("wrong_bin_arrival" if any(pick["bin_arrivals"].values()) else "arrival_not_verified")
                remaining.remove(probe.block)
                return_to_wait(probe, scene)
        except RuntimeError as error:
            report["failure_reason"] = str(error)
        # 앞서 놓은 물체가 나중 동작에 밀려나지 않았는지 최종 상태도 별도로 검사한다.
        if report["success"]:
            final = {str(o["body"]): True for o in scene["objects"]}
            for _ in range(round(CONFIG["arrival_hold_s"] / CONFIG["dt_s"])):
                probe.step()
                for o in scene["objects"]:
                    probe.block = o["body"]
                    final[str(o["body"])] &= probe.measure()["bin_arrivals"][shape]
            report["final_arrivals"] = final
            if not all(final.values()):
                report.update(success=False, failure_reason="final_arrival_not_verified")
    (output / "sort_result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def check_reachability(scene):
    # 초기 자세 진단 전용이다. 관절을 직접 배치한 정적 IK 검사를 실제 운반 성공과 구분한다.
    robot = scene["robot"]
    infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    movable = sorted((i for i in infos if i[2] != p.JOINT_FIXED), key=lambda i: i[3])
    tcp = next(i[0] for i in infos if i[12].decode() == "panda_grasptarget")
    initial = [p.getJointState(robot, i[0])[0] for i in movable]
    checks = []
    try:
        for name, bin_info in scene["bins"].items():
            for slot, xy in enumerate(bin_info["slots"]):
                for z in (0.50, 0.355):
                    # 뒤쪽 목표에 가까운 팔 방향으로 IK를 시작한다. 이것은 이동 경로가 아니다.
                    seed_pose = list(scene["home"])
                    seed_pose[0] = math.atan2(xy[1], xy[0])
                    for i, angle in zip(movable, seed_pose):
                        p.resetJointState(robot, i[0], angle)
                    target_q = p.getQuaternionFromEuler([math.pi, 0, seed_pose[0]])
                    solution = p.calculateInverseKinematics(robot, tcp, [*xy, z],
                                                           target_q,
                                                           maxNumIterations=500, residualThreshold=1e-7)
                    limits = all(i[8] <= angle <= i[9] for i, angle in zip(movable, solution))
                    for i, angle in zip(movable, solution):
                        p.resetJointState(robot, i[0], angle)
                    actual = p.getLinkState(robot, tcp, computeForwardKinematics=True)[4:6]
                    error = math.dist(actual[0], [*xy, z])
                    rotation_error = 2 * math.acos(min(1, abs(sum(a * b for a, b in zip(actual[1], target_q)))))
                    p.performCollisionDetection()
                    collisions = any(p.getClosestPoints(robot, part, 0) for b in scene["bins"].values() for part in b["parts"])
                    checks.append({"bin": name, "slot": slot + 1, "target_m": [*xy, z], "joint_limits_ok": limits,
                                   "position_error_m": error, "orientation_error_rad": rotation_error, "yaw_rad": seed_pose[0],
                                   "bin_collision": collisions,
                                   "valid": limits and error < 0.006 and rotation_error < 0.10 and not collisions})
    finally:
        for i, angle in zip(movable, initial):
            p.resetJointState(robot, i[0], angle)
        p.performCollisionDetection()
    return {"method": "initial_static_ik_only", "checks": checks,
            "all_valid": all(c["valid"] for c in checks), "transport_verified": False}


def settle_scene(scene, timeout_s=5.0):
    consecutive = 0
    for tick in range(round(timeout_s / CONFIG["dt_s"])):
        p.stepSimulation()
        stable = all(math.dist(p.getBaseVelocity(o["body"])[0], (0, 0, 0)) < 0.01
                     and math.dist(p.getBaseVelocity(o["body"])[1], (0, 0, 0)) < 0.01
                     and any(c[9] > 0.001 for c in p.getContactPoints(o["body"], scene["table"])) for o in scene["objects"])
        consecutive = consecutive + 1 if stable else 0
        if consecutive < round(0.5 / CONFIG["dt_s"]):
            continue
        # 정착한 실제 자세·영역·간격을 검사한다. 초기 생성값만으로 안정성을 판단하지 않는다.
        for o in scene["objects"]:
            q = p.getBasePositionAndOrientation(o["body"])[1]
            if p.getMatrixFromQuaternion(q)[8] < math.cos(math.radians(5)):
                return {"settled": False, "failure_reason": "tilted_object", "sim_duration_s": (tick + 1) * CONFIG["dt_s"]}
        aabbs = [p.getAABB(o["body"]) for o in scene["objects"]]
        for low, high in aabbs:
            if any(low[a] < scene["layout"]["region"][k][0] or high[a] > scene["layout"]["region"][k][1]
                   for a, k in enumerate(("x", "y"))):
                return {"settled": False, "failure_reason": "object_outside_region"}
        for i, (low, high) in enumerate(aabbs):
            for other_low, other_high in aabbs[i + 1:]:
                if not any(high[a] + scene["layout"]["gap_m"] <= other_low[a]
                           or other_high[a] + scene["layout"]["gap_m"] <= low[a] for a in (0, 1)):
                    return {"settled": False, "failure_reason": "insufficient_gap"}
        return {"settled": True, "failure_reason": "", "sim_duration_s": (tick + 1) * CONFIG["dt_s"]}
    return {"settled": False, "failure_reason": "settle_timeout", "sim_duration_s": timeout_s}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuboids", type=int, default=3)
    parser.add_argument("--cylinders", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("direct", "gui"), default="direct")
    parser.add_argument("--duration", type=float, default=20, help="GUI 표시 시간(초)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task", choices=("scene", "sort"), default="scene", help="장면 확인 또는 한 종류 순차 운반")
    args = parser.parse_args()
    try:
        validate_counts(args.cuboids, args.cylinders)
    except ValueError as error:
        parser.error(str(error))
    if args.task == "sort" and args.cuboids and args.cylinders:
        parser.error("CNN 연결 전에는 --cuboids N --cylinders 0 또는 반대로 지정하세요")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    if args.mode == "gui" and not os.getenv("DISPLAY"):
        parser.error("GUI needs DISPLAY; use --mode direct")
    output = args.output_dir or Path("outputs/multi-object-scene") / datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({
        "layout": LAYOUT, "cuboids": args.cuboids, "cylinders": args.cylinders, "initial_seed": args.seed,
        "dt_s": CONFIG["dt_s"], "block_size_m": CONFIG["block_size_m"], "cylinder_size_m": [0.05, 0.06],
        "mass_kg": 0.05, "lateral_friction": CONFIG["lateral_friction"], "mode": args.mode, "task": args.task,
        "release_open_force_n": 0.1, "perception_size_tolerance_ratio": 0.15,
        "camera": scene_camera(LAYOUT), "pybullet_version": importlib.metadata.version("pybullet"),
        "numpy_version": importlib.metadata.version("numpy"), "pillow_version": importlib.metadata.version("Pillow"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("rgbd_camera.py", "contact_grasp_probe.py")},
    }, indent=2) + "\n")
    client = p.connect(p.GUI if args.mode == "gui" else p.DIRECT)
    attempts = []
    try:
        for attempt in range(LAYOUT["max_scene_attempts"]):
            seed = args.seed + attempt
            try:
                scene = build_scene(sample_spawns(args.cuboids, args.cylinders, seed, LAYOUT["region"]), LAYOUT)
                result = settle_scene(scene)
            except ValueError as error:
                result = {"settled": False, "failure_reason": str(error)}
            attempts.append({"seed": seed, **result})
            if result["settled"]:
                break
        summary = {"initial_seed": args.seed, "attempts": attempts, "settled": result["settled"],
                   "config_ref": str(output / "config.json"),
                   "layout": LAYOUT, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if result["settled"]:
            summary["objects"] = [{**o, "actual_position": p.getBasePositionAndOrientation(o["body"])[0]} for o in scene["objects"]]
            summary["observation"] = observe_scene(scene, output / "observation")
            # 정적 진단의 관절 reset은 장면 확인에서만 사용한다. 운반에는 호출하지 않는다.
            if args.task == "scene":
                summary["reachability"] = check_reachability(scene)
            else:
                if args.mode == "gui":
                    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                    p.resetDebugVisualizerCamera(1.9, 70, -35, [0.15, 0, 0.55])
                    add_camera_visual(scene["camera"])
                summary["sort"] = sort_scene(scene, "cuboid" if args.cuboids else "cylinder", output / "sort", args.mode == "gui")
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print("SUMMARY_PATH=" + str(output / "summary.json"), flush=True)
        print("낙하·정착 확인: " + ("PASS" if result["settled"] else "FAIL"), flush=True)
        if result["settled"]:
            print(f"초기 관측 영역: {summary['observation']['observed_regions']}개", flush=True)
            if args.task == "scene":
                print("뒤쪽 자리 정적 IK: " + ("PASS" if summary["reachability"]["all_valid"] else "FAIL"), flush=True)
            else:
                print("순차 운반: " + ("PASS" if summary["sort"]["success"] else "FAIL: " + summary["sort"]["failure_reason"]), flush=True)
        if args.mode == "gui" and result["settled"] and args.task == "scene":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.9, 70, -35, [0.15, 0, 0.55])
            add_camera_visual(scene["camera"])
            for _ in range(round(args.duration / CONFIG["dt_s"])):
                if not p.isConnected():
                    return 130
                p.stepSimulation()
                time.sleep(CONFIG["dt_s"])
        verified = result["settled"] and (summary["reachability"]["all_valid"] if args.task == "scene" else summary["sort"]["success"])
        return 0 if verified else 1
    except KeyboardInterrupt:
        return 130
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    raise SystemExit(main())

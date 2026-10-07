"""로봇 앞 최대 5개 낙하 배치와 도형의 카메라 관찰·CNN 분류·순차 운반을 확인한다."""

from terminal_ko import KoreanArgumentParser, failure_name, verdict
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
from rgbd_camera import (
    camera_for_angle,
    capture,
    components,
    depth_metres,
    estimate_many,
    rgb_crop,
    save_observation,
    world_points,
    validate_size_range,
)

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
    if (
        any(type(n) is not int or n < 0 for n in (cuboids, cylinders))
        or not 1 <= cuboids + cylinders <= 5
    ):
        raise ValueError("도형 개수는 각각 0 이상, 합계 1~5개여야 합니다")


def sample_spawns(cuboids, cylinders, seed, region, size_range_m=None):
    validate_counts(cuboids, cylinders)
    if size_range_m is not None:
        minimum, maximum, min_height, max_height = validate_size_range(size_range_m)
    rng = random.Random(seed)
    spawns = []
    for shape in ["cuboid"] * cuboids + ["cylinder"] * cylinders:
        # 학습 데이터와 같은 범위에서 개별 치수를 정한다. 기본 모드의 난수 순서는 유지한다.
        size = (
            (
                [
                    rng.uniform(minimum, maximum)
                    for _ in range(2 if shape == "cuboid" else 1)
                ]
                + [rng.uniform(min_height, max_height)]
            )
            if size_range_m is not None
            else None
        )
        width, length = (
            size[:2] if size is not None and shape == "cuboid" else (0.06, 0.04)
        )
        diameter = size[0] if size is not None and shape == "cylinder" else 0.05
        for _ in range(LAYOUT["max_placement_attempts"]):
            yaw = rng.uniform(-math.pi, math.pi)
            # 회전한 직육면체의 모서리까지 감싸는 가로·세로 반길이를 계산한다.
            half = (
                [
                    (abs(math.cos(yaw)) * width + abs(math.sin(yaw)) * length) / 2,
                    (abs(math.sin(yaw)) * width + abs(math.cos(yaw)) * length) / 2,
                ]
                if shape == "cuboid"
                else [
                    (diameter / 2 + 0.001 if size is not None else 0.026)
                    * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
                ]
                * 2
            )
            # 원기둥도 PyBullet의 회전된 충돌 AABB와 1mm 여유를 고려해 보수적으로 배치한다.
            limits = [
                (region[k][0] + half[a] + 0.002, region[k][1] - half[a] - 0.002)
                for a, k in enumerate(("x", "y"))
            ]
            if any(low > high for low, high in limits):
                break
            xy = [rng.uniform(low, high) for low, high in limits]
            # 한 축 이상에서 물체 사이에 손가락을 위한 3cm 간격을 남긴다.
            if all(
                any(
                    abs(xy[a] - other["xy"][a])
                    >= half[a] + other["half_xy"][a] + LAYOUT["gap_m"] + 0.002
                    for a in (0, 1)
                )
                for other in spawns
            ):
                spawn = {"shape": shape, "xy": xy, "yaw": yaw, "half_xy": half}
                if size is not None:
                    spawn['size_m'] = size
                spawns.append(spawn)
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
    camera["eye"] = [
        camera["eye"][0] + center[0] - 0.5,
        camera["eye"][1] + center[1],
        camera["eye"][2],
    ]
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
    housing = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.025, 0.02, 0.018], rgbaColor=[0.1, 0.12, 0.15, 1]
    )
    lens = p.createVisualShape(
        p.GEOM_CYLINDER, radius=0.012, length=0.018, rgbaColor=[0.1, 0.65, 0.9, 1]
    )
    bodies = [
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=housing,
            basePosition=eye.tolist(),
            baseOrientation=rotation,
        ),
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=lens,
            basePosition=(eye + direction * 0.026).tolist(),
            baseOrientation=rotation,
        ),
    ]
    # 실제 시야각·가로세로비로 화면 네 모서리의 광선을 구하고 상판 높이와 만나는 점을 찾는다.
    view = np.asarray(
        p.computeViewMatrix(camera["eye"], camera["target"], camera["up"])
    ).reshape(4, 4, order="F")
    projection = np.asarray(
        p.computeProjectionMatrixFOV(
            camera["fov_deg"],
            camera["width"] / camera["height"],
            camera["near_m"],
            camera["far_m"],
        )
    ).reshape(4, 4, order="F")
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
        lines.append(
            p.addUserDebugLine(eye.tolist(), corner, [0.95, 0.75, 0.1], lineWidth=1)
        )
        lines.append(
            p.addUserDebugLine(
                corner, footprint[(index + 1) % 4], [0.95, 0.75, 0.1], lineWidth=2
            )
        )
    p.addUserDebugText(
        "RGB-D", (eye + [0, 0, 0.05]).tolist(), [0.1, 0.5, 0.9], textSize=1.2
    )
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
    table = box(
        [(bounds[k][1] - bounds[k][0]) / 2 for k in ("x", "y")] + [0.02],
        [(bounds[k][1] + bounds[k][0]) / 2 for k in ("x", "y")] + [0.28],
        0,
        [0.65, 0.65, 0.65, 1],
    )
    # 생성 구역은 청록색 얇은 표시로 구별한다. 충돌이 없어 물체는 기존 상판에 닿는다.
    region = layout["region"]
    area_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[(region[k][1] - region[k][0]) / 2 for k in ("x", "y")] + [0.0001],
        rgbaColor=[0.25, 0.55, 0.58, 1],
    )
    spawn_area = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=area_shape,
        basePosition=[(region[k][1] + region[k][0]) / 2 for k in ("x", "y")] + [0.3001],
    )
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.30], useFixedBase=True)
    infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    movable = sorted((i for i in infos if i[2] != p.JOINT_FIXED), key=lambda i: i[3])
    home = [0, -0.6, 0, -2.0, 0, 1.6, 0.8, 0.04, 0.04]
    for info, value in zip(movable, home):
        p.resetJointState(robot, info[0], value)
        p.setJointMotorControl2(
            robot,
            info[0],
            p.POSITION_CONTROL,
            targetPosition=value,
            force=20 if info[2] == p.JOINT_PRISMATIC else info[10],
        )
    # Panda URDF의 mimic 관계를 물리 기어로 연결해 두 손가락이 함께 열리고 닫히게 한다.
    # 로봇의 손가락끼리만 연결한다. 물체를 로봇에 붙이는 제약은 만들지 않는다.
    fingers = [
        next(i[0] for i in infos if i[1].decode() == f"panda_finger_joint{n}")
        for n in (1, 2)
    ]
    gear = p.createConstraint(
        robot,
        fingers[0],
        robot,
        fingers[1],
        p.JOINT_GEAR,
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    )
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
            parts.append(
                box(
                    [thickness / 2, hy + thickness, wall / 2],
                    [x + sign * (hx + thickness / 2), y, floor_top + wall / 2],
                    0,
                    color,
                )
            )
            parts.append(
                box(
                    [hx, thickness / 2, wall / 2],
                    [x, y + sign * (hy + thickness / 2), floor_top + wall / 2],
                    0,
                    color,
                )
            )
        # 한 상자에 최대 5개를 놓을 자리다. 실제 운반 단계에서 각 자리의 도달을 검증한다.
        slots = [[x + dx, y + dy] for dy in (-0.05, 0.05) for dx in (-0.09, 0, 0.09)]
        # 정적 IK 검사에서 손바닥이 벽에 닿은 바깥 모서리 하나를 제외해 다섯 자리를 남긴다.
        del slots[5 if shape == "cuboid" else 2]
        bins[shape] = {
            "center": [x, y],
            "floor": floor,
            "parts": parts,
            "slots": slots,
            "floor_top": floor_top,
        }
    camera = scene_camera(layout)
    # 물체 없는 기준 영상도 같은 카메라·로봇 대기 자세에서 촬영한다.
    empty = capture(camera)
    objects = []
    for spawn in spawns:
        # 데이터 수집에서는 크기를 지정한다. 기존 실행은 원래 기본 크기를 유지한다.
        size = spawn.get(
            "size_m",
            CONFIG["block_size_m"] if spawn["shape"] == "cuboid" else [0.05, 0.06],
        )
        if len(size) != (3 if spawn["shape"] == "cuboid" else 2) or any(
            not math.isfinite(n) or n <= 0 for n in size
        ):
            raise ValueError("물체 치수는 양수이며 유한한 값이어야 합니다")
        z = CONFIG["table_top_m"] + size[-1] / 2 + layout["drop_clearance_m"]
        if spawn["shape"] == "cuboid":
            body = box(
                [n / 2 for n in size], [*spawn["xy"], z], 0.05, [0.9, 0.15, 0.1, 1]
            )
        else:
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=size[0] / 2, height=size[1]
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=size[0] / 2,
                length=size[1],
                rgbaColor=[0.9, 0.15, 0.1, 1],
            )
            body = p.createMultiBody(0.05, collision, visual, [*spawn["xy"], z])
        # 수평 회전은 생성 시에만 지정한다. 이후 낙하·정착은 물리 계산에 맡긴다.
        p.resetBasePositionAndOrientation(
            body, [*spawn["xy"], z], p.getQuaternionFromEuler([0, 0, spawn["yaw"]])
        )
        p.changeDynamics(body, -1, lateralFriction=CONFIG["lateral_friction"])
        objects.append({**spawn, "body": body})
    return {
        "robot": robot,
        "table": table,
        "spawn_area_visual": spawn_area,
        "objects": objects,
        "bins": bins,
        "layout": layout,
        "home": home,
        "camera": camera,
        "empty": empty,
    }


def observe_scene(scene, output):
    output.mkdir(parents=True, exist_ok=True)
    # 재관측에서 영역 수가 줄어도 예전 crop·mask가 남지 않도록 이 프로그램의 번호 파일만 정리한다.
    for prefix in ("crop", "mask"):
        for path in output.glob(f"{prefix}_*.png"):
            if path.stem[len(prefix) + 1 :].isdigit():
                path.unlink()
    camera = scene["camera"]
    observation = capture(camera)
    empty = scene["empty"]
    points = world_points(observation)
    # 검증용 objects 목록을 읽지 않고 RGB-D와 관측 구역만으로 영역을 찾는다.
    difference = depth_metres(empty["depth"], camera) - depth_metres(
        observation["depth"], camera
    )
    mask = difference > camera["depth_difference_m"]
    for axis, (low, high) in enumerate(camera["roi_xy_m"]):
        mask &= (points[..., axis] > low) & (points[..., axis] < high)
    mask &= (points[..., 2] > camera["table_top_m"] + 0.008) & (
        observation["depth"] < 1
    )
    # 원본 RGB·float 깊이는 기존 저장 함수를 재사용한다. 클래스·파지 기하는 아직 계산하지 않는다.
    save_observation(output, observation, empty, {"valid": False}, camera)
    regions = []
    groups = sorted(components(mask), key=lambda g: min(u for _, u in g))
    for index, group in enumerate(groups):
        pixels = np.asarray(group)
        region_mask = np.zeros_like(mask)
        region_mask[pixels[:, 0], pixels[:, 1]] = True
        rgb_crop(observation["rgb"], region_mask, camera["crop_size"]).save(
            output / f"crop_{index + 1}.png"
        )
        Image.fromarray(region_mask.astype(np.uint8) * 255).save(
            output / f"mask_{index + 1}.png"
        )
        regions.append(
            {
                "index": index + 1,
                "pixels": len(group),
                "bbox_uv": [
                    int(pixels[:, 1].min()),
                    int(pixels[:, 0].min()),
                    int(pixels[:, 1].max()),
                    int(pixels[:, 0].max()),
                ],
            }
        )
    result = {
        "observed_regions": len(regions),
        "regions": regions,
        "camera": camera,
        "classification_source": "not_implemented",
        "grasp_geometry_computed": False,
    }
    (output / "observation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class SceneProbe(Probe):
    def step(self):
        measurement = super().step()
        # 접촉을 잃고 상대 위치가 크게 변하면 현재 이동부터 중단한다.
        # 정상 내려놓기·복구는 carry_reference를 먼저 해제하여 의도한 개방과 구분한다.
        if self.carry_reference is not None and self.drop_via_relative_motion:
            raise RuntimeError("drop_during_transport")
        return measurement

    def command_tcp(self, point, rotation):
        # 현재 자세 가까이에서 IK를 풀고 모터에 전달한다. 실제 움직임은 step에서 계산된다.
        rest = [math.atan2(point[1], point[0]), -0.6, 0, -2.0, 0, 1.6, 0.8, 0.04, 0.04]
        solution = p.calculateInverseKinematics(
            self.robot,
            self.tcp,
            point,
            rotation,
            lowerLimits=[i[8] for i in self.movable],
            upperLimits=[i[9] for i in self.movable],
            jointRanges=[i[9] - i[8] for i in self.movable],
            restPoses=rest,
            maxNumIterations=100,
            residualThreshold=1e-7,
        )
        values = {info[0]: value for info, value in zip(self.movable, solution)}
        if any(
            not self.infos[i][8] <= values[i] <= self.infos[i][9] for i in self.arms
        ):
            raise RuntimeError("ik_joint_limit")
        self.arm_targets = [values[i] for i in self.arms]
        self.hold_arm()

    def move(self, stage, position, seconds):
        # 손끝 경로를 작은 구간으로 나누어 현재 자세 가까이에서 IK를 다시 푼다.
        self.start_stage(stage)
        initial, rotation = p.getLinkState(
            self.robot, self.tcp, computeForwardKinematics=True
        )[4:6]
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

    def carry_arc(self, direction, seconds=4.0, end_angle=None):
        self.start_stage("CARRY_ARC" if end_angle is None else "RECOVERY_ARC")
        position, rotation = p.getLinkState(
            self.robot, self.tcp, computeForwardKinematics=True
        )[4:6]
        start_angle = math.atan2(position[1], position[0])
        start_radius = math.hypot(position[0], position[1])
        start_yaw = p.getEulerFromQuaternion(rotation)[2]
        end_angle = direction * 2.4 if end_angle is None else end_angle
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
        super().move(
            "CARRY_ARC_FINAL",
            [0.47 * math.cos(end_angle), 0.47 * math.sin(end_angle), 0.65],
            0.3,
        )


def scene_probe(scene, log, gui):
    # 기존 접촉 제어를 새 장면에 연결한다. 초기화를 다시 호출하지 않는다.
    probe = SceneProbe(gui, log, pose_source="camera")
    probe.robot, probe.table = scene["robot"], scene["table"]
    probe.infos = [
        p.getJointInfo(probe.robot, i) for i in range(p.getNumJoints(probe.robot))
    ]
    names = {i[1].decode(): i for i in probe.infos}
    probe.arms = [names[f"panda_joint{i}"][0] for i in range(1, 8)]
    probe.fingers = [names[f"panda_finger_joint{i}"][0] for i in (1, 2)]
    probe.tcp = next(i[0] for i in probe.infos if i[12].decode() == "panda_grasptarget")
    probe.hand = next(i[0] for i in probe.infos if i[12].decode() == "panda_hand")
    probe.movable = sorted(
        (i for i in probe.infos if i[2] != p.JOINT_FIXED), key=lambda i: i[3]
    )
    probe.arm_forces = [probe.infos[i][10] for i in probe.arms]
    probe.arm_targets = scene["home"][:7]
    probe.bins = {
        name: {
            **b,
            "inner_half_size_m": [s / 2 for s in scene["layout"]["bin_inner_size_m"]],
        }
        for name, b in scene["bins"].items()
    }
    probe.supports = [scene["table"]] + [
        part for b in scene["bins"].values() for part in b["parts"]
    ]
    for finger in probe.fingers:
        p.changeDynamics(
            probe.robot, finger, lateralFriction=CONFIG["lateral_friction"]
        )
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
        probe.arm_targets = [
            a + blend * (b - a) for a, b in zip(initial, scene["home"][:7])
        ]
        probe.hold_arm()
        probe.step()
    probe.wait("WAIT_SETTLE", 0.5)
    if (
        max(
            abs(p.getJointState(probe.robot, i)[0] - target)
            for i, target in zip(probe.arms, scene["home"][:7])
        )
        > 0.01
    ):
        raise RuntimeError("wait_pose_not_reached")


def recover_pick(probe, scene, plan, cause):
    # 실제 충돌·관절 오류에서는 호출하지 않는다. 복구 중에도 기존 충돌 검사를 계속한다.
    probe.carry_reference = None
    if cause == "lift_not_verified":
        # 불안정하게 잡혀 있을 수도 있으므로, 카메라가 정한 원래 높이에서 손을 연다.
        # 물체의 시뮬레이터 정답 좌표로 위치를 보정하지 않는다.
        probe.move("RECOVERY_LOWER", plan["position_m"], 1.0)
    probe.gripper(0.04)
    probe.wait("RECOVERY_RELEASE", 0.8)
    position, rotation = p.getLinkState(
        probe.robot, probe.tcp, computeForwardKinematics=True
    )[4:6]
    probe.orientation = rotation
    probe.move("RECOVERY_CLEARANCE", [position[0], position[1], 0.65], 1.0)
    # 뒤쪽에서 대기 자세로 직행하지 않고, 운반 호를 역방향으로 따라 앞쪽까지 돌아온다.
    angle = math.atan2(position[1], position[0])
    if abs(angle) > math.pi / 2:
        probe.carry_arc(1 if angle > 0 else -1, end_angle=0.0)
    return_to_wait(probe, scene)
    probe.drop_via_relative_motion = False


def execute_pick(probe, scene, geometry, pick, placed_counts, remaining):
    # 한 물체를 실제 접촉으로 옮긴다. 기존 CLI와 ROS2에서 같은 제어·판정을 사용한다.
    # 목표는 카메라 기하와 CNN 종류에서 만들고, 생성 정답은 평가에만 사용한다.
    plan = pick["grasp_plan"]
    target_shape = geometry["object_shape"]
    matches = sorted(
        (
            math.dist(
                geometry["center_xy_m"],
                p.getBasePositionAndOrientation(body)[0][:2],
            ),
            body,
        )
        for body in remaining
    )
    if not matches or matches[0][0] > 0.01:
        raise RuntimeError("evaluation_target_unmatched")
    probe.block = matches[0][1]
    pick["evaluation_body_id"] = probe.block
    pick["evaluation_xy_error_m"] = matches[0][0]
    # 정답 종류는 점수 기록·최종 도착 평가에만 쓴다. 계획과 목적지를 바꾸지 않는다.
    true_shape = next(o["shape"] for o in scene["objects"] if o["body"] == probe.block)
    pick.update(
        true_class=true_shape,
        classification_correct=target_shape == true_shape,
    )
    probe.supports = (
        [scene["table"]]
        + [part for b in scene["bins"].values() for part in b["parts"]]
        + [o["body"] for o in scene["objects"] if o["body"] != probe.block]
    )
    probe.drop_via_relative_motion = False
    x, y, z = plan["position_m"]
    probe.orientation = p.getQuaternionFromEuler([math.pi, 0, plan["yaw_rad"]])
    # 물체 위에서 집게를 벌린 다음 내려가서 닫는다.
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
    if probe.drop_via_relative_motion:
        raise RuntimeError("drop_during_transport")
    # 몸통을 가로지르지 않도록 높은 위치에서 바깥쪽으로 돌아 뒤 상자로 간다.
    destination = scene["bins"][target_shape]["slots"][placed_counts[target_shape]]
    direction = 1 if destination[1] > 0 else -1
    probe.carry_arc(direction)
    if probe.drop_via_relative_motion:
        raise RuntimeError("drop_during_transport")
    # 상자에서는 긴 변과 손끝 방향을 나란하게 맞춰 벽과 개방 공간을 확보한다.
    probe.orientation = p.getQuaternionFromEuler([math.pi, 0, direction * math.pi])
    probe.move("BIN_ABOVE", [*destination, 0.50], 1.0)
    if probe.drop_via_relative_motion:
        raise RuntimeError("drop_during_transport")
    # 바닥 가까이 내려놓는다. 실제 정착은 접촉과 속도로 따로 확인한다.
    release_z = max(
        z + CONFIG["bin_floor_thickness_m"],
        scene["bins"][target_shape]["floor_top"]
        + CONFIG["bin_release_tcp_floor_clearance_m"],
    )
    pick["release_tcp_z_m"] = release_z
    probe.move("BIN_PLACE", [*destination, release_z], 1.2)
    if probe.drop_via_relative_motion:
        raise RuntimeError("drop_during_transport")
    probe.carry_reference = None
    # 놓을 때만 낮은 고정 개방 힘을 쓴다. 집는 힘·마찰을 추정하거나 자동 조절하지 않는다.
    p.setJointMotorControlArray(
        probe.robot,
        probe.fingers,
        p.POSITION_CONTROL,
        targetPositions=[0.04, 0.04],
        forces=[0.1, 0.1],
    )
    probe.wait("RELEASE", 1.0)
    probe.move("RETREAT", [*destination, 0.65], 1.0)
    # 낙하 후 흔들림이 멎는 시간은 일정하지 않다. 최대 5초 안에 0.5초 연속 정착을 확인한다.
    probe.start_stage("VERIFY_ARRIVAL")
    consecutive = {name: 0 for name in probe.bins}
    for _ in range(round(5.0 / CONFIG["dt_s"])):
        sample = probe.step()
        for name in consecutive:
            consecutive[name] = (
                consecutive[name] + 1 if sample["bin_arrivals"][name] else 0
            )
        if any(
            n >= round(CONFIG["arrival_hold_s"] / CONFIG["dt_s"])
            for n in consecutive.values()
        ):
            break
    # 지정한 상자 도착, 다른 상자 도착, 정착 실패를 구분한다.
    pick["bin_arrivals"] = {
        name: n >= round(CONFIG["arrival_hold_s"] / CONFIG["dt_s"])
        for name, n in consecutive.items()
    }
    pick["arrival_success"] = pick["bin_arrivals"][target_shape]
    pick["correct_bin_arrival"] = pick["bin_arrivals"][true_shape]
    if not pick["arrival_success"]:
        raise RuntimeError(
            "wrong_bin_arrival"
            if any(pick["bin_arrivals"].values())
            else "arrival_not_verified"
        )
    if not pick["correct_bin_arrival"]:
        raise RuntimeError("classification_wrong_bin")
    placed_counts[target_shape] += 1
    remaining.remove(probe.block)
    # 팔을 촬영 때의 대기 자세로 돌린 뒤 다음 물체의 시도 횟수를 새로 센다.
    return_to_wait(probe, scene)


def verify_final_arrivals(probe, scene):
    # 앞서 놓은 물체도 모두 함께 안정된 상태인지 마지막에 다시 확인한다.
    probe.start_stage('VERIFY_FINAL_ARRIVAL')
    consecutive = {str(o['body']): 0 for o in scene['objects']}
    required = round(CONFIG['arrival_hold_s'] / CONFIG['dt_s'])
    # 마지막 후퇴 직후의 순간 흔들림은 기다린다. 모두가0.5초 연속 정착해야 통과한다.
    # 개별 도착 검사와 같이 최대5초까지만 기다리고 판정 조건 자체는 완화하지 않는다.
    for _ in range(round(5.0 / CONFIG['dt_s'])):
        probe.step()
        for o in scene['objects']:
            probe.block = o['body']
            key = str(o['body'])
            stable = probe.measure()['bin_arrivals'][o['shape']]
            consecutive[key] = consecutive[key] + 1 if stable else 0
        if all(n >= required for n in consecutive.values()):
            break
    final = {key: n >= required for key, n in consecutive.items()}
    return final


def sort_scene(
    scene,
    shape,
    output,
    gui=False,
    supported_top_m=None,
    classifier=None,
    min_class_score=0.8,
    size_range_m=None,
    max_retries=2,
):
    # 최초 시도 외의 추가 시도는 최대2회다. 0이면 기존처럼 첫 실패에서 중단한다.
    if type(max_retries) is not int or not 0 <= max_retries <= 2:
        raise ValueError("재시도 횟수는0~2 사이의 정수여야 합니다")
    if classifier is None and (
        shape not in ("cuboid", "cylinder")
        or any(o["shape"] != shape for o in scene["objects"])
    ):
        raise ValueError("수동 모드에서는 지정한 한 종류만 운반합니다")
    if classifier is not None:
        classifier.validate_camera(scene["camera"])
    # 크기가 다른 원기둥의 바닥 접촉이 수치적으로 흔들리지 않도록 접촉 계산을 더 반복한다.
    # 속도·접촉에 대한 성공 기준은 그대로이며, 기본 크기 모드의 물리 설정은 유지한다.
    if size_range_m is not None:
        p.setPhysicsEngineParameter(numSolverIterations=200)
    output.mkdir(parents=True, exist_ok=True)
    placed_counts = {"cuboid": 0, "cylinder": 0}
    report = {
        "success": False,
        "classification_source": (
            "cnn" if classifier is not None else "manual_single_shape"
        ),
        "scans": [],
        "picks": [],
        "failure_reason": "",
    }
    report["solver_iterations"] = p.getPhysicsEngineParameters()["numSolverIterations"]
    report.update(max_retries=max_retries, recoveries=[], last_failure_reason="")
    with (output / "motion.jsonl").open("w") as log:
        probe = scene_probe(scene, log, gui)
        # ID는 평가 대상의 대응에만 쓴다. 이동 좌표는 매번 카메라에서 구한다.
        remaining = {o["body"] for o in scene["objects"]}
        if not remaining:
            raise ValueError("운반 검증에는 물체가 하나 이상 필요합니다")
        probe.block = next(iter(remaining))
        retry_count = 0
        retry_xy = None
        try:
            # 성공한 운반 뒤에는 새 작업으로 센다. 반복 상한도 두어 무한 재촬영을 막는다.
            for iteration in range(
                len(remaining) * (max_retries + 1) + max_retries + 1
            ):
                # 한 물체를 옮긴 뒤 바뀐 장면을 새로 촬영한다. 이전 좌표를 계속 쓰지 않는다.
                observation = capture(scene["camera"])
                # 평소에는 첫 유효 후보까지만 분류한다. 재시도 때는 이전 중심 근처를 찾기 위해 전체 검사한다.
                candidates = estimate_many(
                    observation,
                    scene["empty"],
                    scene["camera"],
                    shape,
                    supported_top_m,
                    classifier,
                    min_class_score,
                    first_valid_only=classifier is not None and retry_xy is None,
                    size_range_m=size_range_m,
                )
                scan_dir = output / f"scan_{iteration:02d}"
                scan_dir.mkdir(parents=True, exist_ok=True)
                save_observation(
                    scan_dir,
                    observation,
                    scene["empty"],
                    {"valid": False},
                    scene["camera"],
                )
                scan = {
                    "valid_targets": sum(g["valid"] for g in candidates),
                    "cnn_evaluations": sum(
                        g.get("cnn_evaluated", False) for g in candidates
                    ),
                    "retry_index": retry_count,
                    "retry_anchor_xy_m": retry_xy,
                    "regions": [
                        {k: v for k, v in g.items() if k != "mask"} for g in candidates
                    ],
                }
                report["scans"].append(scan)
                (scan_dir / "regions.json").write_text(
                    json.dumps(scan, indent=2) + "\n"
                )
                for index, candidate in enumerate(candidates):
                    if "mask" in candidate:
                        rgb_crop(observation["rgb"], candidate["mask"]).save(
                            scan_dir / f"crop_{index + 1}.png"
                        )
                        Image.fromarray(candidate["mask"].astype(np.uint8) * 255).save(
                            scan_dir / f"mask_{index + 1}.png"
                        )
                valid = [g for g in candidates if g["valid"]]
                if not valid:
                    if not remaining:
                        report["success"] = True
                        break
                    report["last_failure_reason"] = "no_valid_target"
                    # 손상된 깊이와 잃어버린 재시도 대상은 다른 물체로 바꾸어 집지 않는다.
                    if any(
                        g.get("failure_reason") == "invalid_depth" for g in candidates
                    ):
                        report["failure_reason"] = "invalid_depth"
                        break
                    if retry_xy is not None:
                        report["failure_reason"] = "retry_target_not_found"
                        break
                    if retry_count >= max_retries:
                        report["failure_reason"] = (
                            "retry_limit_reached" if max_retries else "no_valid_target"
                        )
                        break
                    # 모든 후보가 보류되면 팔은 대기 자세에 두고 잠시 기다린 뒤 다시 촬영한다.
                    retry_count += 1
                    print(
                        f"집을 수 있는 후보가 없어 재촬영합니다: 추가 시도 {retry_count}/{max_retries}",
                        flush=True,
                    )
                    probe.wait("REOBSERVE_WAIT", 0.5)
                    report["recoveries"].append(
                        {
                            "cause": "no_valid_target",
                            "action": "wait_and_rescan",
                            "retry_index": retry_count,
                            "scan_index": iteration,
                            "status": "ready_to_rescan",
                        }
                    )
                    continue
                # CNN 모드는 첫 유효 후보를 사용한다. 수동 모드는 전체 검사 후 윗면이 큰 후보를 고른다.
                geometry = max(valid, key=lambda g: g["top_pixels"])
                if retry_xy is not None:
                    # 재시도는 지난 영상 중심에서3cm 이내인 가장 가까운 후보에 한정한다.
                    # 생성 ID나 정답 위치를 대상 추적에 쓰지 않는다. 크게 이동한 대상은 중단한다.
                    nearby = sorted(
                        valid, key=lambda g: math.dist(g["center_xy_m"], retry_xy)
                    )
                    if math.dist(nearby[0]["center_xy_m"], retry_xy) > 0.03:
                        raise RuntimeError("retry_target_not_found")
                    if (
                        len(nearby) > 1
                        and math.dist(nearby[1]["center_xy_m"], retry_xy) <= 0.03
                    ):
                        raise RuntimeError("retry_target_ambiguous")
                    geometry = nearby[0]
                # 상자 선택도 생성 정답이 아닌 카메라 영역의 CNN 예측으로 결정한다.
                target_shape = geometry["object_shape"]
                # 영상에서 구한 중심·폭·각도를 손끝 위치와 손가락 간격으로 바꾼다.
                plan = grasp_plan(geometry)
                pick = {
                    "grasp_plan": plan,
                    "lift_success": False,
                    "arrival_success": False,
                    "predicted_class": target_shape,
                    "class_score": geometry.get("class_score"),
                    "class_probabilities": geometry.get("class_probabilities"),
                    "target_bin": target_shape,
                    "scan_index": iteration,
                    "retry_index": retry_count,
                    "failure_reason": "",
                }
                report["picks"].append(pick)
                try:
                    execute_pick(probe, scene, geometry, pick, placed_counts, remaining)
                    retry_count = 0
                    retry_xy = None
                except RuntimeError as error:
                    cause = str(error)
                    pick["failure_reason"] = cause
                    report["last_failure_reason"] = cause
                    # 충돌 등의 위험한 실패는 재시도하지 않도록 두 실패만 허용한다.
                    if (
                        cause not in ("lift_not_verified", "drop_during_transport")
                        or max_retries == 0
                    ):
                        raise
                    limit_reached = retry_count >= max_retries
                    if not limit_reached:
                        retry_count += 1
                    print("집기·운반 실패: " + failure_name(cause), flush=True)
                    print(
                        (
                            "재시도 한도에 도달하여 안전 복귀 후 종료합니다"
                            if limit_reached
                            else f"안전 복귀 후 다시 촬영합니다: 추가 시도 {retry_count}/{max_retries}"
                        ),
                        flush=True,
                    )
                    # 한도를 소진한 마지막 실패도 손을 열고 안전 복귀한 뒤 종료한다.
                    recovery = {
                        "cause": cause,
                        "action": (
                            "safe_stop_after_limit"
                            if limit_reached
                            else "release_retreat_and_rescan"
                        ),
                        "retry_index": retry_count,
                        "scan_index": iteration,
                        "anchor_xy_m": geometry["center_xy_m"],
                        "status": "recovering",
                    }
                    report["recoveries"].append(recovery)
                    try:
                        recover_pick(probe, scene, plan, cause)
                    except RuntimeError as recovery_error:
                        recovery.update(
                            status="failed", failure_reason=str(recovery_error)
                        )
                        raise RuntimeError("recovery_failed") from recovery_error
                    recovery["status"] = (
                        "stopped_after_limit" if limit_reached else "ready_to_rescan"
                    )
                    if limit_reached:
                        raise RuntimeError("retry_limit_reached") from error
                    retry_xy = geometry["center_xy_m"]
            else:
                report["failure_reason"] = "task_iteration_limit"
        except RuntimeError as error:
            report["failure_reason"] = str(error)
        # 앞서 놓은 물체가 나중 동작에 밀려나지 않았는지 최종 상태도 별도로 검사한다.
        try:
            if report["success"]:
                final = verify_final_arrivals(probe, scene)
                report['final_arrivals'] = final
                if not all(final.values()):
                    report.update(
                        success=False, failure_reason="final_arrival_not_verified"
                    )
        except RuntimeError as error:
            # 최종 검사 중의 안전 오류도 결과 파일에 남기고 자동 복구 없이 중단한다.
            report.update(
                success=False, failure_reason=str(error), last_failure_reason=str(error)
            )
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
                    solution = p.calculateInverseKinematics(
                        robot,
                        tcp,
                        [*xy, z],
                        target_q,
                        maxNumIterations=500,
                        residualThreshold=1e-7,
                    )
                    limits = all(
                        i[8] <= angle <= i[9] for i, angle in zip(movable, solution)
                    )
                    for i, angle in zip(movable, solution):
                        p.resetJointState(robot, i[0], angle)
                    actual = p.getLinkState(robot, tcp, computeForwardKinematics=True)[
                        4:6
                    ]
                    error = math.dist(actual[0], [*xy, z])
                    rotation_error = 2 * math.acos(
                        min(1, abs(sum(a * b for a, b in zip(actual[1], target_q))))
                    )
                    p.performCollisionDetection()
                    collisions = any(
                        p.getClosestPoints(robot, part, 0)
                        for b in scene["bins"].values()
                        for part in b["parts"]
                    )
                    checks.append(
                        {
                            "bin": name,
                            "slot": slot + 1,
                            "target_m": [*xy, z],
                            "joint_limits_ok": limits,
                            "position_error_m": error,
                            "orientation_error_rad": rotation_error,
                            "yaw_rad": seed_pose[0],
                            "bin_collision": collisions,
                            "valid": limits
                            and error < 0.006
                            and rotation_error < 0.10
                            and not collisions,
                        }
                    )
    finally:
        for i, angle in zip(movable, initial):
            p.resetJointState(robot, i[0], angle)
        p.performCollisionDetection()
    return {
        "method": "initial_static_ik_only",
        "checks": checks,
        "all_valid": all(c["valid"] for c in checks),
        "transport_verified": False,
    }


def settle_scene(scene, timeout_s=5.0):
    consecutive = 0
    for tick in range(round(timeout_s / CONFIG["dt_s"])):
        p.stepSimulation()
        stable = all(
            math.dist(p.getBaseVelocity(o["body"])[0], (0, 0, 0)) < 0.01
            and math.dist(p.getBaseVelocity(o["body"])[1], (0, 0, 0)) < 0.01
            and any(c[9] > 0.001 for c in p.getContactPoints(o["body"], scene["table"]))
            for o in scene["objects"]
        )
        consecutive = consecutive + 1 if stable else 0
        if consecutive < round(0.5 / CONFIG["dt_s"]):
            continue
        # 정착한 실제 자세·영역·간격을 검사한다. 초기 생성값만으로 안정성을 판단하지 않는다.
        for o in scene["objects"]:
            q = p.getBasePositionAndOrientation(o["body"])[1]
            if p.getMatrixFromQuaternion(q)[8] < math.cos(math.radians(5)):
                return {
                    "settled": False,
                    "failure_reason": "tilted_object",
                    "sim_duration_s": (tick + 1) * CONFIG["dt_s"],
                }
        aabbs = [p.getAABB(o["body"]) for o in scene["objects"]]
        for low, high in aabbs:
            if any(
                low[a] < scene["layout"]["region"][k][0]
                or high[a] > scene["layout"]["region"][k][1]
                for a, k in enumerate(("x", "y"))
            ):
                return {"settled": False, "failure_reason": "object_outside_region"}
        for i, (low, high) in enumerate(aabbs):
            for other_low, other_high in aabbs[i + 1 :]:
                if not any(
                    high[a] + scene["layout"]["gap_m"] <= other_low[a]
                    or other_high[a] + scene["layout"]["gap_m"] <= low[a]
                    for a in (0, 1)
                ):
                    return {"settled": False, "failure_reason": "insufficient_gap"}
        return {
            "settled": True,
            "failure_reason": "",
            "sim_duration_s": (tick + 1) * CONFIG["dt_s"],
        }
    return {
        "settled": False,
        "failure_reason": "settle_timeout",
        "sim_duration_s": timeout_s,
    }


def main():
    parser = KoreanArgumentParser(description=__doc__)
    parser.add_argument("--cuboids", type=int, default=3)
    parser.add_argument("--cylinders", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("direct", "gui"), default="direct")
    parser.add_argument("--duration", type=float, default=20, help="GUI 표시 시간(초)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--task",
        choices=("scene", "sort"),
        default="scene",
        help="장면 확인 또는 순차 운반",
    )
    parser.add_argument(
        "--classification-source",
        choices=("manual", "cnn"),
        default="manual",
        help="종류 입력 또는 CNN 자동 분류",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("checkpoints/shape_cnn_v1")
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="최초 시도 후 추가 재시도 횟수(0~2)",
    )
    parser.add_argument(
        "--min-class-score",
        type=float,
        default=0.8,
        help="이보다 낮은 분류 점수의 영역은 보류",
    )
    parser.add_argument(
        '--size-mode',
        choices=('fixed', 'random'),
        default='fixed',
        help='기본크기 또는 개별 랜덤 크기',
    )
    parser.add_argument('--min-width-cm', type=float, default=3)
    parser.add_argument('--max-width-cm', type=float, default=6.4)
    parser.add_argument('--min-height-cm', type=float, default=3)
    parser.add_argument('--max-height-cm', type=float, default=7)
    args = parser.parse_args()
    size_range_m = None
    if args.size_mode == 'random':
        try:
            # 터미널의 cm를 카메라·물리 계산에 쓰는 m로 바꾼다.
            size_range_m = validate_size_range(
                tuple(
                    n / 100
                    for n in (
                        args.min_width_cm,
                        args.max_width_cm,
                        args.min_height_cm,
                        args.max_height_cm,
                    )
                )
            )
        except ValueError as error:
            parser.error(str(error))
        if args.task == 'sort' and args.classification_source != 'cnn':
            parser.error('랜덤 크기 운반에는 --classification-source cnn이 필요합니다')

    try:
        validate_counts(args.cuboids, args.cylinders)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.task == "sort"
        and args.classification_source == "manual"
        and args.cuboids
        and args.cylinders
    ):
        parser.error("수동 모드에서는 --cuboids N --cylinders 0 또는 반대로 지정하세요")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("표시 시간은 양수이며 유한한 값이어야 합니다")
    if args.mode == "gui" and not os.getenv("DISPLAY"):
        parser.error(
            "창을 표시할 수 없습니다. 데스크톱 터미널에서 실행하거나 --mode direct를 사용하세요"
        )
    if not math.isfinite(args.min_class_score) or not 0.5 <= args.min_class_score <= 1:
        parser.error("최소 분류 점수는0.5~1 사이여야 합니다")
    classifier = None
    if args.classification_source == "cnn":
        if args.task != "sort":
            parser.error("CNN 분류는 --task sort와 함께 사용하세요")
        # 기존 수동 실험은 PyTorch 없이도 실행된다. CNN을 선택했을 때만 가져온다.
        try:
            from cnn_inference import CnnClassifier

            classifier = CnnClassifier(args.model_dir)
            classifier.validate_camera(scene_camera(LAYOUT))
        except (ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
            # 파일·패키지 누락은 운영체제의 영문 예외 대신 필요한 조치를 표시한다.
            detail = str(error)
            if isinstance(error, OSError):
                detail = f"모델 또는 설정 파일을 읽을 수 없습니다: {error.filename or args.model_dir}"
            elif isinstance(error, ImportError):
                detail = f"추론 패키지를 불러올 수 없습니다: {error.name}"
            parser.error(
                f"CNN 준비 실패: {detail}; requirements-inference.txt와 --model-dir를 확인하세요"
            )
    output = args.output_dir or Path(
        "outputs/multi-object-scene"
    ) / datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "layout": LAYOUT,
                "cuboids": args.cuboids,
                "cylinders": args.cylinders,
                "initial_seed": args.seed,
                "dt_s": CONFIG["dt_s"],
                "block_size_m": CONFIG["block_size_m"],
                "cylinder_size_m": [0.05, 0.06],
                "mass_kg": 0.05,
                "lateral_friction": CONFIG["lateral_friction"],
                "mode": args.mode,
                "task": args.task,
                "release_open_force_n": 0.1,
                "perception_size_tolerance_ratio": 0.15,
                "classification_source": args.classification_source,
                "min_class_score": args.min_class_score,
                "max_retries": args.max_retries,
                "size_mode": args.size_mode,
                "size_range_m": size_range_m,
                "surface_policy": (
                    "range_footprint_occlusion"
                    if size_range_m is not None
                    else "known_top_15_percent"
                ),
                "candidate_policy": (
                    "depth_priority_first_valid"
                    if classifier is not None
                    else "all_then_largest_top"
                ),
                "model_dir": str(args.model_dir) if classifier is not None else None,
                "model_sha256": (
                    classifier.model_sha256 if classifier is not None else None
                ),
                "model_params_sha256": (
                    classifier.params_sha256 if classifier is not None else None
                ),
                "torch_version": (
                    importlib.metadata.version("torch")
                    if classifier is not None
                    else None
                ),
                "camera": scene_camera(LAYOUT),
                "pybullet_version": importlib.metadata.version("pybullet"),
                "numpy_version": importlib.metadata.version("numpy"),
                "pillow_version": importlib.metadata.version("Pillow"),
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "dependency_sha256": {
                    name: hashlib.sha256(
                        Path(__file__).with_name(name).read_bytes()
                    ).hexdigest()
                    for name in (
                        ("rgbd_camera.py", "contact_grasp_probe.py", "cnn_inference.py")
                        if classifier is not None
                        else ("rgbd_camera.py", "contact_grasp_probe.py")
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    # GUI는 창을 띄우고, DIRECT는 창 없이 같은 물리 계산을 실행한다.
    client = p.connect(p.GUI if args.mode == "gui" else p.DIRECT)
    attempts = []
    try:
        for attempt in range(LAYOUT["max_scene_attempts"]):
            seed = args.seed + attempt
            try:
                scene = build_scene(
                    sample_spawns(
                        args.cuboids,
                        args.cylinders,
                        seed,
                        LAYOUT["region"],
                        size_range_m,
                    ),
                    LAYOUT,
                )
                result = settle_scene(scene)
            except ValueError as error:
                result = {"settled": False, "failure_reason": str(error)}
            attempts.append({"seed": seed, **result})
            if result["settled"]:
                break
        summary = {
            "initial_seed": args.seed,
            "attempts": attempts,
            "settled": result["settled"],
            "config_ref": str(output / "config.json"),
            "layout": LAYOUT,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if result["settled"]:
            summary["objects"] = [
                {**o, "actual_position": p.getBasePositionAndOrientation(o["body"])[0]}
                for o in scene["objects"]
            ]
            summary["observation"] = observe_scene(scene, output / "observation")
            # 정적 진단의 관절 reset은 장면 확인에서만 사용한다. 운반에는 호출하지 않는다.
            if args.task == "scene":
                summary["reachability"] = check_reachability(scene)
            else:
                if args.mode == "gui":
                    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                    p.resetDebugVisualizerCamera(1.9, 70, -35, [0.15, 0, 0.55])
                    add_camera_visual(scene["camera"])
                summary["sort"] = sort_scene(
                    scene,
                    (
                        None
                        if classifier is not None
                        else ("cuboid" if args.cuboids else "cylinder")
                    ),
                    output / "sort",
                    args.mode == "gui",
                    classifier=classifier,
                    min_class_score=args.min_class_score,
                    size_range_m=size_range_m,
                    max_retries=args.max_retries,
                )
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print("결과 파일: " + str(output / "summary.json"), flush=True)
        print("낙하·정착 확인: " + verdict(result["settled"]), flush=True)
        if not result["settled"]:
            print(
                "배치 실패 원인: " + failure_name(result["failure_reason"]), flush=True
            )
        if result["settled"]:
            print(
                f"초기 관측 영역: {summary['observation']['observed_regions']}개",
                flush=True,
            )
            if args.task == "scene":
                print(
                    "뒤쪽 자리 역기구학 확인: "
                    + verdict(summary["reachability"]["all_valid"]),
                    flush=True,
                )
            else:
                print(
                    "순차 운반: "
                    + (
                        "성공"
                        if summary["sort"]["success"]
                        else "실패: " + failure_name(summary["sort"]["failure_reason"])
                    ),
                    flush=True,
                )
        if args.mode == "gui" and result["settled"] and args.task == "scene":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.9, 70, -35, [0.15, 0, 0.55])
            add_camera_visual(scene["camera"])
            for _ in range(round(args.duration / CONFIG["dt_s"])):
                if not p.isConnected():
                    return 130
                p.stepSimulation()
                time.sleep(CONFIG["dt_s"])
        verified = result["settled"] and (
            summary["reachability"]["all_valid"]
            if args.task == "scene"
            else summary["sort"]["success"]
        )
        return 0 if verified else 1
    except KeyboardInterrupt:
        return 130
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    raise SystemExit(main())

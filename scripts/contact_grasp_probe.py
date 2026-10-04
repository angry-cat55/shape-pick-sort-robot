"""직육면체·수직 원기둥의 접촉 집기와 운반을 검증한다."""

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import pybullet as p
import pybullet_data

from rgbd_camera import CAMERA, camera_for_angle, capture, estimate, save_observation


KST = timezone(timedelta(hours=9))
CONFIG = {
    # ponytail: 두 도형의 위에서 잡기만 다룬다. 새 도형은 기하·파지 계산을 검증해 추가한다.
    "version": "grasp-v5-larger-defaults",
    "pose_source": "oracle",
    "dt_s": 1 / 240,
    "table_top_m": 0.30,
    "block_size_m": [0.06, 0.04, 0.06],
    "block_mass_kg": 0.05,
    "block_xy_m": [0.5, 0.0],
    "grasp_tcp_z_m": 0.345,  # 고정 정답 실험: 높이 6cm 상면에서 15mm 아래. 카메라 제어에서는 쓰지 않는다.
    "grasp_depth_below_top_m": 0.015,
    "gripper_opening_margin_m": 0.015,
    "lift_distance_m": 0.10,
    "lift_bottom_threshold_m": 0.05,
    "hold_duration_s": 1.0,
    "relative_slip_limit_m": 0.01,
    "finger_force_n": 20.0,  # 손가락 모터의 최대 힘이다. 실제 접촉 힘과는 다르다.
    # 마찰은 고정한다. 다양화와 잡는 힘 조절은 후속 후보다.
    "lateral_friction": 1.0,
    "tcp_position_tolerance_m": 0.006,
    "tcp_orientation_tolerance_rad": 0.10,
    "miss_offset_y_m": 0.12,
    "bin_centers_xy_m": {"blue": [0.35, 0.25], "green": [0.35, -0.25]},
    "bin_inner_half_size_m": 0.08,
    "bin_wall_thickness_m": 0.01,
    "bin_wall_height_m": 0.08,
    "bin_floor_thickness_m": 0.01,
    # 현재 Panda 손바닥과 높이 8cm 상자 벽이 겹치지 않는 최소 TCP 높이(상자 바닥 기준).
    "bin_release_tcp_floor_clearance_m": 0.045,  # 벽이 2cm 높아져 놓기 최소 높이도 2cm 올린다.
    "bin_inside_margin_m": 0.003,
    "transport_tcp_z_m": 0.50,
    "arrival_hold_s": 0.5,
}


def grasp_plan(geometry):
    # 도형별 함수가 구한 같은 형식의 정보로 손끝 목표를 만든다. 정답 pose·크기는 받지 않는다.
    return {
        "position_m": [*geometry["center_xy_m"], geometry["top_z_m"] - CONFIG["grasp_depth_below_top_m"]],
        "yaw_rad": geometry["yaw_rad"],
        "opening_per_finger_m": (geometry["width_m"] + CONFIG["gripper_opening_margin_m"]) / 2,
    }


class Probe:
    def __init__(self, gui, log, pose_source="oracle", output=None, spawn=(0.5, 0, 0), camera=None,
                 object_shape="cuboid", cylinder_size=(0.05, 0.06)):
        self.gui = gui
        self.log = log
        self.step_index = 0
        self.stage = "SETUP"
        self.max_tcp_error = 0.0
        self.carry_reference = None
        self.drop_via_relative_motion = False
        self.pose_source = pose_source
        self.output = output
        self.spawn = spawn
        self.camera = camera or CAMERA
        self.object_shape = object_shape
        self.cylinder_size = cylinder_size
        self.grasp_xy = list(CONFIG["block_xy_m"])
        self.grasp_z = CONFIG["grasp_tcp_z_m"]
        self.opening = 0.04

    def box(self, half, position, mass, color):
        # 충돌 계산용 모양과 화면에 보이는 모양을 함께 만든다. 질량 0은 고정 물체다.
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual, basePosition=position,
        )

    def setup(self, scenario):
        # resetSimulation과 resetJointState는 각 독립 실험의 초기 배치에만 사용한다.
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        p.setTimeStep(CONFIG["dt_s"])
        p.setPhysicsEngineParameter(numSolverIterations=100)
        p.setGravity(0, 0, -9.81)
        self.ground = p.loadURDF("plane.urdf")
        self.table = self.box([0.65, 0.45, 0.02], [0.3, 0, 0.28], 0, [0.65, 0.65, 0.65, 1])
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.30], useFixedBase=True)
        self.infos = [p.getJointInfo(self.robot, i) for i in range(p.getNumJoints(self.robot))]
        # 관절 이름으로 팔·손가락을 찾고, TCP(잡는 기준점)를 지정한다.
        by_name = {info[1].decode(): info for info in self.infos}
        self.arms = [by_name[f"panda_joint{i}"][0] for i in range(1, 8)]
        self.fingers = [by_name[f"panda_finger_joint{i}"][0] for i in (1, 2)]
        self.tcp = next(info[0] for info in self.infos if info[12].decode() == "panda_grasptarget")
        self.hand = next(info[0] for info in self.infos if info[12].decode() == "panda_hand")
        self.movable = sorted((info for info in self.infos if info[2] != p.JOINT_FIXED), key=lambda info: info[3])
        home = [0, -0.6, 0, -2.0, 0, 1.6, 0.8, 0.04, 0.04]
        for info, value in zip(self.movable, home):
            p.resetJointState(self.robot, info[0], value)
        self.arm_forces = [self.infos[index][10] for index in self.arms]
        self.arm_targets = home[:7]
        self.hold_arm()
        self.gripper(0.04)
        # 초기 배치 높이는 도형 높이의 절반과 바닥 위 2mm로 정한다.
        object_height = self.cylinder_size[1] if self.object_shape == "cylinder" else CONFIG["block_size_m"][2]
        block_z = CONFIG["table_top_m"] + object_height / 2 + 0.002
        self.supports = [self.ground, self.table]
        self.bins = {}
        if scenario in ("transport", "transport_drop", "wrong_bin", "camera_empty"):
            self.create_bins()
        if scenario == "support":
            # 받침대에 올라간 물체를 집기 성공으로 착각하는지 검사한다.
            support = self.box([0.06, 0.06, 0.05], [0.5, 0, 0.35], 0, [0.2, 0.4, 0.8, 1])
            self.supports.append(support)
            block_z = 0.40 + object_height / 2 + 0.002
        # 카메라 모드의 기준 영상은 물체 생성 전에 찍는다. 로봇·상자는 같은 상태다.
        if self.pose_source == "camera":
            self.empty_observation = capture(self.camera)
        position = [self.spawn[0], self.spawn[1], block_z]
        color = [0.9, 0.15, 0.1, 1]
        if self.object_shape == "cylinder":
            # 충돌 모양과 보이는 모양에 같은 지름·높이를 준다. 수직 원기둥으로 생성한다.
            diameter, height = self.cylinder_size
            collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=diameter / 2, height=height)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=diameter / 2, length=height, rgbaColor=color)
            self.block = p.createMultiBody(CONFIG["block_mass_kg"], collision, visual, position)
        else:
            self.block = self.box([length / 2 for length in CONFIG["block_size_m"]], position,
                                  CONFIG["block_mass_kg"], color)
        # 회전 설정은 독립 실험의 초기 배치에만 적용한다.
        p.resetBasePositionAndOrientation(self.block, [self.spawn[0], self.spawn[1], block_z],
                                          p.getQuaternionFromEuler([0, 0, self.spawn[2]]))
        p.changeDynamics(self.block, -1, lateralFriction=CONFIG["lateral_friction"])
        for index in self.fingers:
            p.changeDynamics(self.robot, index, lateralFriction=CONFIG["lateral_friction"])
        self.orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
        if self.gui:
            # 화면 관찰용 시점이며, 학습용 카메라 설정은 아니다.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.3, 45, -25, [0.35, 0, 0.5])
        self.log.write(json.dumps({
            "event": "model", "tcp_link": self.tcp,
            "joints": [{"index": i[0], "joint": i[1].decode(), "link": i[12].decode(),
                        "lower": i[8], "upper": i[9]} for i in self.infos],
        }) + "\n")
        self.wait("SETTLE", 1.0)

    def observe(self, scenario, result):
        self.start_stage("OBSERVE")
        if scenario == "camera_empty":
            # 빈 관측 대조군: 물체를 제거한 뒤 움직임 없이 영상 판정만 검사한다.
            p.removeBody(self.block)
        observation = capture(self.camera)
        geometry = estimate(observation, self.empty_observation, self.camera, self.object_shape)
        save_observation(self.output, observation, self.empty_observation, geometry, self.camera)
        result["perception"] = {key: value for key, value in geometry.items() if key != "mask"}
        if not geometry["valid"]:
            raise RuntimeError(geometry["failure_reason"])
        # 제어 입력은 오직 영상 추정값이다. 검증용 정답을 이 값에 섞지 않는다.
        plan = grasp_plan(geometry)
        result["grasp_plan"] = plan
        self.grasp_xy = plan["position_m"][:2]
        self.grasp_z = plan["position_m"][2]
        self.opening = plan["opening_per_finger_m"]
        self.orientation = p.getQuaternionFromEuler([math.pi, 0, plan["yaw_rad"]])
        self.gripper(self.opening)
        # 아래 정답 조회는 오차 기록 전용이며 이동 목표를 바꾸지 않는다.
        actual = p.getBasePositionAndOrientation(self.block)[0]
        result["perception_error_xy_m"] = math.dist(self.grasp_xy, actual[:2])
        result["perception_error_top_m"] = abs(geometry["top_z_m"] - p.getAABB(self.block)[1][2])
        self.log.write(json.dumps({"event": "perception", **result["perception"],
                                   "error_xy_m": result["perception_error_xy_m"]}) + "\n")

    def create_bins(self):
        # 상자 하나를 고정된 바닥 1개와 벽 4개로 만든다.
        half = CONFIG["bin_inner_half_size_m"]
        thickness = CONFIG["bin_wall_thickness_m"]
        floor_thickness = CONFIG["bin_floor_thickness_m"]
        floor_top = CONFIG["table_top_m"] + floor_thickness
        height = CONFIG["bin_wall_height_m"]
        for name, (x, y) in CONFIG["bin_centers_xy_m"].items():
            color = [0.15, 0.35, 0.9, 1] if name == "blue" else [0.15, 0.7, 0.3, 1]
            floor = self.box(
                [half + thickness, half + thickness, floor_thickness / 2],
                [x, y, CONFIG["table_top_m"] + floor_thickness / 2], 0, color,
            )
            parts = [floor]
            wall_z = floor_top + height / 2
            for sign in (-1, 1):
                parts.append(self.box(
                    [thickness / 2, half + thickness, height / 2],
                    [x + sign * (half + thickness / 2), y, wall_z], 0, color,
                ))
                parts.append(self.box(
                    [half, thickness / 2, height / 2],
                    [x, y + sign * (half + thickness / 2), wall_z], 0, color,
                ))
            self.supports.extend(parts)
            self.bins[name] = {"floor": floor, "parts": parts, "center": [x, y], "floor_top": floor_top}

    def hold_arm(self):
        # 목표 관절 각도를 모터에 전달한다. 실제 움직임은 물리 계산으로 진행된다.
        p.setJointMotorControlArray(
            self.robot, self.arms, p.POSITION_CONTROL,
            targetPositions=self.arm_targets, forces=self.arm_forces,
        )

    def gripper(self, opening_per_finger):
        # 한쪽 손가락의 열림 거리다. 0으로 닫아도 물체와 충돌하면 그 앞에서 멈춘다.
        p.setJointMotorControlArray(
            self.robot, self.fingers, p.POSITION_CONTROL,
            targetPositions=[opening_per_finger] * 2,
            forces=[CONFIG["finger_force_n"]] * 2,
        )

    def measure(self):
        # 검증용 시뮬레이터 값을 읽는다. 카메라로 추정한 값이 아니다.
        tcp_pose = p.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)[4:6]
        block_pose = p.getBasePositionAndOrientation(self.block)
        inv = p.invertTransform(*tcp_pose)
        # 손끝 기준 물체 위치를 구해, 팔 이동과 물체 미끄러짐을 구분한다.
        relative = p.multiplyTransforms(*inv, *block_pose)[0]
        contacts = p.getContactPoints(self.block, self.robot)
        finger_contacts = sorted({c[4] for c in contacts if c[4] in self.fingers and c[9] > 0.001})
        other_link_support = any(c[4] not in self.fingers and c[9] > 0.001 for c in contacts)
        support = other_link_support or any(
            c[9] > 0.001 for body in self.supports for c in p.getContactPoints(self.block, body)
        )
        arm_table = any(c[3] in self.arms and c[9] > 0.05 for c in p.getContactPoints(self.robot, self.table))
        arm_bin = any(
            (c[3] in self.arms or c[3] == self.hand) and c[9] > 0.05
            for bin_info in self.bins.values() for part in bin_info["parts"]
            for c in p.getContactPoints(self.robot, part)
        )
        linear, angular = p.getBaseVelocity(self.block)
        aabb = p.getAABB(self.block)
        shapes = p.getCollisionShapeData(self.block, -1)
        if len(shapes) == 1 and shapes[0][2] == p.GEOM_CYLINDER:
            # 검증 전용: 회전 원기둥의 PyBullet AABB는 느슨해서 바닥 아래로 과하게 나올 수 있다.
            # 원기둥 축의 각 방향 성분으로 실제 표면을 감싸는 경계를 계산한다.
            height, radius, _ = shapes[0][3]
            rotation = p.getMatrixFromQuaternion(block_pose[1])
            axis = [rotation[i] for i in (2, 5, 8)]
            half = [abs(v) * height / 2 + radius * math.sqrt(max(0, 1 - v * v)) for v in axis]
            aabb = ([block_pose[0][i] - half[i] for i in range(3)],
                    [block_pose[0][i] + half[i] for i in range(3)])
        # AABB는 물체를 감싸는 상자다. 물체 전체가 상자 안에 정착했는지 확인한다.
        bin_arrivals = {}
        for name, bin_info in self.bins.items():
            # 새 장면의 직사각형 상자도 검사한다. 기존 정사각형 실험의 기본값은 유지한다.
            halves = bin_info.get("inner_half_size_m", [CONFIG["bin_inner_half_size_m"]] * 2)
            inside = all(
                aabb[0][axis] >= bin_info["center"][axis] - halves[axis] + CONFIG["bin_inside_margin_m"]
                and aabb[1][axis] <= bin_info["center"][axis] + halves[axis] - CONFIG["bin_inside_margin_m"] for axis in (0, 1)
            )
            bin_arrivals[name] = (
                inside and abs(aabb[0][2] - bin_info["floor_top"]) < 0.005
                and aabb[1][2] <= bin_info["floor_top"] + CONFIG["bin_wall_height_m"] - CONFIG["bin_inside_margin_m"]
                and math.dist(linear, (0, 0, 0)) < 0.01
                and math.dist(angular, (0, 0, 0)) < 0.01
                and any(c[9] > 0.001 for c in p.getContactPoints(self.block, bin_info["floor"]))
            )
        return {
            "step": self.step_index, "stage": self.stage,
            "bottom_height_m": aabb[0][2] - CONFIG["table_top_m"],
            "block_position_m": block_pose[0], "tcp_position_m": tcp_pose[0],
            "relative_position_m": relative, "finger_contact_links": finger_contacts,
            "other_support": support, "arm_table_collision": arm_table,
            "arm_bin_collision": arm_bin, "bin_arrivals": bin_arrivals,
            "arm_max_joint_error_rad": max(
                abs(p.getJointState(self.robot, index)[0] - target)
                for index, target in zip(self.arms, self.arm_targets)
            ),
            "finger_positions_m": [p.getJointState(self.robot, index)[0] for index in self.fingers],
            "finger_normal_forces_n": [
                sum(c[9] for c in contacts if c[4] == index) for index in self.fingers
            ],
            "linear_speed_m_s": math.dist(linear, (0, 0, 0)),
            "angular_speed_rad_s": math.dist(angular, (0, 0, 0)),
        }

    def step(self):
        if not p.isConnected():
            raise KeyboardInterrupt
        p.stepSimulation()
        self.step_index += 1
        measurement = self.measure()
        # 운반 중 손가락 접촉이 사라지고 상대 위치도 크게 바뀌면 낙하로 기록한다.
        if (
            self.carry_reference is not None
            and not measurement["finger_contact_links"]
            and math.dist(measurement["relative_position_m"], self.carry_reference)
                > CONFIG["relative_slip_limit_m"]
        ):
            self.drop_via_relative_motion = True
        if self.step_index % 12 == 0:
            # 판정은 매 계산마다 하고, 파일 기록은 12번에 한 번만 남긴다.
            self.log.write(json.dumps(measurement) + "\n")
        if measurement["arm_table_collision"]:
            raise RuntimeError("arm_table_collision")
        if measurement["arm_bin_collision"]:
            raise RuntimeError("arm_bin_collision")
        if self.gui:
            time.sleep(CONFIG["dt_s"])
        return measurement

    def start_stage(self, stage):
        self.stage = stage
        print(f"  {stage}", flush=True)

    def wait(self, stage, seconds):
        self.start_stage(stage)
        return [self.step() for _ in range(round(seconds / CONFIG["dt_s"]))]

    def move(self, stage, position, seconds):
        self.start_stage(stage)
        # 역기구학: 손끝의 목표 위치·방향을 팔 관절 각도로 바꾼다.
        solution = p.calculateInverseKinematics(
            self.robot, self.tcp, position, self.orientation,
            maxNumIterations=500, residualThreshold=1e-7,
        )
        by_index = {info[0]: value for info, value in zip(self.movable, solution)}
        target = [by_index[index] for index in self.arms]
        if any(not info[8] <= by_index[info[0]] <= info[9] for info in self.movable if info[0] in self.arms):
            raise RuntimeError("ik_joint_limit")
        initial = [p.getJointState(self.robot, index)[0] for index in self.arms]
        count = round(seconds / CONFIG["dt_s"])
        for tick in range(1, count + 1):
            t = tick / count
            # 시작과 끝에서 천천히 움직이도록 목표 관절 각도를 조금씩 바꾼다.
            blend = t * t * (3 - 2 * t)
            self.arm_targets = [start + blend * (end - start) for start, end in zip(initial, target)]
            self.hold_arm()
            self.step()
        self.wait(stage + "_SETTLE", 0.25)
        actual = p.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)[4:6]
        # 명령만 보냈다고 성공은 아니다. 실제 손끝 위치·방향의 오차를 확인한다.
        error = math.dist(actual[0], position)
        angular_error = 2 * math.acos(min(1, abs(sum(a * b for a, b in zip(actual[1], self.orientation)))))
        self.max_tcp_error = max(self.max_tcp_error, error)
        if error > CONFIG["tcp_position_tolerance_m"] or angular_error > CONFIG["tcp_orientation_tolerance_rad"]:
            raise RuntimeError(f"tcp_not_reached: position={error:.6f}, orientation={angular_error:.6f}")

    def verify_hold(self, result):
        # 1초 동안 충분히 들렸고, 받침이 없으며, 상대 위치가 안정적인지 확인한다.
        samples = self.wait("VERIFY_LIFT", CONFIG["hold_duration_s"])
        initial_relative = samples[0]["relative_position_m"]
        min_height = min(s["bottom_height_m"] for s in samples)
        supports = sum(s["other_support"] for s in samples)
        slip = max(math.dist(s["relative_position_m"], initial_relative) for s in samples)
        bilateral = sum(len(s["finger_contact_links"]) == 2 for s in samples)
        result.update({
            "hold_duration_s": len(samples) * CONFIG["dt_s"],
            "hold_min_bottom_height_m": min_height, "support_contact_steps": supports,
            "hold_max_relative_slip_m": slip, "bilateral_contact_steps": bilateral,
            "lift_success": min_height >= CONFIG["lift_bottom_threshold_m"]
                and supports == 0 and slip <= CONFIG["relative_slip_limit_m"],
        })
        if result["lift_success"]:
            self.carry_reference = samples[-1]["relative_position_m"]

    def transport(self, scenario, result):
        # 정답 상자는 파란색이다. wrong_bin은 일부러 초록색으로 보내는 대조 실험이다.
        commanded = "green" if scenario == "wrong_bin" else "blue"
        result.update({"expected_bin": "blue", "commanded_bin": commanded})
        height = CONFIG["transport_tcp_z_m"]
        self.move("CLEARANCE", [*self.grasp_xy, height], 1.0)
        destination = self.bins[commanded]
        if scenario == "transport_drop":
            self.move("TRANSPORT_HALF", [0.425, 0.125, height], 1.2)
            self.gripper(0.04)
            self.wait("FORCED_TRANSPORT_DROP", 1.5)
            result["drop_detected"] = self.drop_via_relative_motion
            result["failure_reason"] = "drop_during_transport"
        else:
            x, y = destination["center"]
            self.move("TRANSPORT", [x, y, height], 2.0)
            if self.drop_via_relative_motion:
                result["failure_reason"] = "drop_during_transport"
            else:
                # 낮은 물체는 손바닥이 벽에 걸린다. 최소 여유 높이에서 의도적으로 놓고 정착을 확인한다.
                nominal_z = self.grasp_z + CONFIG["bin_floor_thickness_m"]
                minimum_z = destination["floor_top"] + CONFIG["bin_release_tcp_floor_clearance_m"]
                place_z = max(nominal_z, minimum_z)
                result["placement_plan"] = {"nominal_tcp_z_m": nominal_z, "release_tcp_z_m": place_z}
                self.move("BIN_PLACE", [x, y, place_z], 1.5)
                # 내려놓으려고 손을 여는 동작은 운반 중 낙하로 처리하지 않는다.
                self.carry_reference = None
                self.gripper(0.04)
                self.wait("BIN_RELEASE", 0.6)
                self.move("BIN_RETREAT", [x, y, height], 1.2)
                self.wait("ARRIVAL_SETTLE", 0.5)
        samples = self.wait("VERIFY_ARRIVAL", CONFIG["arrival_hold_s"])
        # 한순간 들어간 것만으로 성공 처리하지 않고, 0.5초 내내 정착했는지 확인한다.
        result["bin_arrivals"] = {
            name: all(s["bin_arrivals"][name] for s in samples) for name in self.bins
        }
        result["arrival_hold_s"] = len(samples) * CONFIG["dt_s"]
        result["final_block_position_m"] = samples[-1]["block_position_m"]
        result["commanded_bin_arrival"] = result["bin_arrivals"][commanded]
        result["arrival_success"] = result["bin_arrivals"]["blue"] and not self.drop_via_relative_motion
        result["place_success"] = any(result["bin_arrivals"].values())
        if not result["failure_reason"] and not result["arrival_success"]:
            result["failure_reason"] = "wrong_bin_arrival" if result["bin_arrivals"]["green"] else "arrival_not_verified"

    def run(self, scenario, trial_index):
        started = time.monotonic()
        result = {
            "scenario": scenario, "trial_index": trial_index, "pose_source": self.pose_source,
            "object_shape": self.object_shape,
            "lift_success": False, "place_success": False, "drop_detected": False,
            "hold_duration_s": 0.0, "hold_min_bottom_height_m": None,
            "support_contact_steps": 0, "failure_reason": "",
            "bin_arrivals": {}, "arrival_success": False,
        }
        try:
            self.setup(scenario)
            if self.pose_source == "camera":
                self.observe(scenario, result)
            if scenario == "support":
                self.verify_hold(result)
                result["failure_reason"] = "supported_by_pedestal"
            else:
                x, y = self.grasp_xy
                if scenario == "miss":
                    y += CONFIG["miss_offset_y_m"]
                # 접근 → 내려가기 → 닫기 → 들어올리기 → 유지 확인 순서로 진행한다.
                self.move("APPROACH", [x, y, 0.48], 1.5)
                self.move("DESCEND", [x, y, self.grasp_z], 1.2)
                self.gripper(0)
                self.wait("CLOSE", 0.8)
                self.move("LIFT", [x, y, self.grasp_z + CONFIG["lift_distance_m"]], 1.5)
                self.verify_hold(result)
                if not result["lift_success"]:
                    result["failure_reason"] = "lift_not_verified"
                elif scenario == "open":
                    self.gripper(0.04)
                    samples = self.wait("FORCED_OPEN", 1.5)
                    result["drop_detected"] = any(s["bottom_height_m"] < 0.02 for s in samples)
                    result["failure_reason"] = "drop_after_open"
                elif scenario in ("transport", "transport_drop", "wrong_bin"):
                    self.transport(scenario, result)
                else:
                    self.move("PLACE", [*self.grasp_xy, self.grasp_z], 1.5)
                    # 상판에 내려놓은 뒤 의도적으로 여는 동작은 낙하 감시에서 제외한다.
                    self.carry_reference = None
                    self.gripper(0.04)
                    self.wait("RELEASE", 0.6)
                    self.move("RETREAT", [*self.grasp_xy, 0.48], 1.2)
                    final = self.wait("VERIFY_PLACE", 0.5)[-1]
                    result["place_success"] = (
                        abs(final["bottom_height_m"]) < 0.005
                        and final["linear_speed_m_s"] < 0.01
                        and final["angular_speed_rad_s"] < 0.01
                        and math.dist(final["block_position_m"][:2], self.grasp_xy) < 0.03
                        and any(c[9] > 0.001 for c in p.getContactPoints(self.block, self.table))
                    )
                    if not result["place_success"]:
                        result["failure_reason"] = "place_not_verified"
        except RuntimeError as error:
            result["failure_reason"] = str(error)
        result.update({
            "sim_duration_s": self.step_index * CONFIG["dt_s"],
            "wall_duration_s": time.monotonic() - started,
            "max_tcp_target_error_m": self.max_tcp_error,
            "drop_via_relative_motion": self.drop_via_relative_motion,
        })
        result["drop_detected"] = result["drop_detected"] or self.drop_via_relative_motion
        if self.drop_via_relative_motion and not result["failure_reason"]:
            result["failure_reason"] = "drop_during_motion"
        # 실패 대조 실험은 실패를 제대로 감지해야 expectation_met가 참이다.
        if scenario == "camera_empty":
            expected = result["failure_reason"] == "no_object" and not result["lift_success"]
        elif scenario == "transport":
            expected = result["lift_success"] and result["arrival_success"] and not result["drop_detected"]
        elif scenario == "transport_drop":
            expected = result["lift_success"] and result["drop_detected"] and not result["arrival_success"]
        elif scenario == "wrong_bin":
            expected = result["bin_arrivals"].get("green", False) and not result["arrival_success"] and result["failure_reason"] == "wrong_bin_arrival"
        elif scenario == "normal":
            expected = result["lift_success"] and result["place_success"] and not result["drop_detected"]
        elif scenario == "open":
            expected = result["lift_success"] and result["drop_detected"]
        elif scenario == "miss":
            expected = result["failure_reason"] == "lift_not_verified" and not result["lift_success"]
        else:
            expected = result["support_contact_steps"] > 0 and not result["lift_success"]
        result["expectation_met"] = expected
        self.log.write(json.dumps({"event": "result", **result}) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["direct", "gui"], default="direct")
    parser.add_argument("--scenario", choices=[
        "normal", "miss", "open", "support", "all", "transport", "transport_drop", "wrong_bin", "all_transport", "camera_empty",
    ], default="normal")
    parser.add_argument("--pose-source", choices=["oracle", "camera"], default="oracle")
    parser.add_argument("--object-shape", choices=["cuboid", "cylinder"], default="cuboid")
    parser.add_argument("--cylinder-diameter", type=float, default=0.05, help="생성할 원기둥 지름(m)")
    parser.add_argument("--cylinder-height", type=float, default=0.06, help="생성할 원기둥 높이(m)")
    parser.add_argument("--block-x", type=float, default=0.5)
    parser.add_argument("--block-y", type=float, default=0.0)
    parser.add_argument("--block-yaw", type=float, default=0.0, help="초기 배치의 수평 회전각(rad)")
    parser.add_argument("--camera-angle", type=int, choices=[50, 65, 90], default=50)
    parser.add_argument("--trials", type=int, default=1, help="그룹 실행은 정상 시나리오만 반복, 대조군 각각 1회")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    # 원기둥은 영상 기반 실험으로만 실행한다. 고정 직육면체 oracle 기준과 섞지 않는다.
    if args.object_shape == "cylinder" and args.pose_source != "camera":
        parser.error("cylinder requires --pose-source camera")
    if args.object_shape == "cuboid" and (args.cylinder_diameter, args.cylinder_height) != (0.05, 0.06):
        parser.error("cylinder dimensions require --object-shape cylinder")
    if not all(math.isfinite(value) and value > 0 for value in (args.cylinder_diameter, args.cylinder_height)):
        parser.error("cylinder dimensions must be positive and finite")
    if args.pose_source == "camera" and args.scenario in ("support", "all"):
        parser.error("support/all are oracle probes; camera supports all_transport or individual grasp scenarios")
    if args.pose_source == "oracle" and (args.scenario == "camera_empty" or
            (args.block_x, args.block_y, args.block_yaw) != (0.5, 0, 0)):
        parser.error("camera_empty and varied block pose require --pose-source camera")
    if not all(math.isfinite(value) for value in (args.block_x, args.block_y, args.block_yaw)):
        parser.error("block pose must be finite")
    if args.mode == "gui" and sys.platform.startswith("linux") and not os.getenv("DISPLAY"):
        parser.error("GUI needs DISPLAY; use --mode direct")
    now = datetime.now(KST)
    run_id = now.strftime("%Y%m%dT%H%M%S_%f")
    output = args.output_dir or Path(__file__).resolve().parents[1] / "outputs" / "contact-grasp" / run_id
    output.mkdir(parents=True, exist_ok=True)
    # 각도 비교를 위해 같은 거리·관찰 중심에서 50/65/90도로 촬영한다.
    camera = camera_for_angle(args.camera_angle)
    # 설정·코드 해시·개별 측정·요약을 저장해 나중에 같은 실험을 확인할 수 있게 한다.
    (output / "config.json").write_text(json.dumps({
        **CONFIG, "pybullet_version": importlib.metadata.version("pybullet"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mode": args.mode, "scenario": args.scenario, "trials": args.trials,
        "pose_source": args.pose_source, "spawn_xy_yaw": [args.block_x, args.block_y, args.block_yaw],
        "object_shape": args.object_shape,
        "cylinder_diameter_m": args.cylinder_diameter if args.object_shape == "cylinder" else None,
        "cylinder_height_m": args.cylinder_height if args.object_shape == "cylinder" else None,
        "shape_source": "manual",
        "camera": camera, "numpy_version": importlib.metadata.version("numpy"),
        "pillow_version": importlib.metadata.version("Pillow"),
        "camera_script_sha256": hashlib.sha256(Path(__file__).with_name("rgbd_camera.py").read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    client = p.connect(p.GUI if args.mode == "gui" else p.DIRECT)
    if client < 0:
        print("PyBullet 연결 실패", file=sys.stderr)
        return 2
    results = []
    try:
        if args.scenario == "all":
            scenarios = ["normal", "miss", "open", "support"]
        elif args.scenario == "all_transport":
            scenarios = ["transport", "transport_drop", "wrong_bin"]
        else:
            scenarios = [args.scenario]
        for scenario in scenarios:
            count = args.trials if scenario in ("normal", "transport") or args.scenario not in ("all", "all_transport") else 1
            for index in range(count):
                print(f"[{scenario} #{index + 1}] pose_source={args.pose_source}, shape={args.object_shape}", flush=True)
                with (output / f"{scenario}_{index + 1}.jsonl").open("w") as log:
                    # 각 시도의 관측 이미지는 개별 폴더에 보관한다.
                    observation_output = output / f"{scenario}_{index + 1}"
                    if args.pose_source == "camera":
                        observation_output.mkdir(exist_ok=True)
                    results.append(Probe(args.mode == "gui", log, args.pose_source, observation_output,
                                         (args.block_x, args.block_y, args.block_yaw), camera,
                                         args.object_shape, (args.cylinder_diameter, args.cylinder_height)).run(scenario, index + 1))
                print("TRIAL_JSON=" + json.dumps(results[-1]), flush=True)
        summary = {
            "run_id": run_id, "recorded_at_kst": now.isoformat(),
            "config_ref": str(output / "config.json"), "trials": results,
            "all_expectations_met": all(r["expectation_met"] for r in results),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY_PATH=" + str(output / "summary.json"), flush=True)
        return 0 if summary["all_expectations_met"] else 1
    except KeyboardInterrupt:
        print("중단했습니다. 전체 실험을 통과했다고 기록하지 않습니다.", flush=True)
        return 130
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    raise SystemExit(main())

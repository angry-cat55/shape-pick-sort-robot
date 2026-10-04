"""로봇 앞 최대 5개 낙하 배치와 뒤쪽 두 상자를 확인한다. 순차 운반은 다음 단계다."""

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

from contact_grasp_probe import CONFIG
from rgbd_camera import camera_for_angle, capture, components, depth_metres, rgb_crop, save_observation, world_points


LAYOUT = {
    "version": "multi-scene-v2-camera-display",
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
    args = parser.parse_args()
    try:
        validate_counts(args.cuboids, args.cylinders)
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    if args.mode == "gui" and not os.getenv("DISPLAY"):
        parser.error("GUI needs DISPLAY; use --mode direct")
    output = args.output_dir or Path("outputs/multi-object-scene") / datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({
        "layout": LAYOUT, "cuboids": args.cuboids, "cylinders": args.cylinders, "initial_seed": args.seed,
        "dt_s": CONFIG["dt_s"], "block_size_m": CONFIG["block_size_m"], "cylinder_size_m": [0.05, 0.06],
        "mass_kg": 0.05, "lateral_friction": CONFIG["lateral_friction"], "mode": args.mode,
        "camera": scene_camera(LAYOUT), "pybullet_version": importlib.metadata.version("pybullet"),
        "numpy_version": importlib.metadata.version("numpy"), "pillow_version": importlib.metadata.version("Pillow"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
            summary["reachability"] = check_reachability(scene)
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print("SUMMARY_PATH=" + str(output / "summary.json"), flush=True)
        print("낙하·정착 확인: " + ("PASS" if result["settled"] else "FAIL"), flush=True)
        if result["settled"]:
            print(f"관측 영역: {summary['observation']['observed_regions']}개 (분류·운반은 다음 단계)", flush=True)
            print("뒤쪽 자리 정적 IK: " + ("PASS" if summary["reachability"]["all_valid"] else "FAIL"), flush=True)
        if args.mode == "gui" and result["settled"]:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.9, 70, -35, [0.15, 0, 0.55])
            add_camera_visual(scene["camera"])
            for _ in range(round(args.duration / CONFIG["dt_s"])):
                if not p.isConnected():
                    return 130
                p.stepSimulation()
                time.sleep(CONFIG["dt_s"])
        return 0 if result["settled"] and summary["reachability"]["all_valid"] else 1
    except KeyboardInterrupt:
        return 130
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    raise SystemExit(main())

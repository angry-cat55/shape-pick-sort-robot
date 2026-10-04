"""로봇 앞 최대 5개 낙하 배치와 뒤쪽 두 상자를 확인한다. 순차 운반은 다음 단계다."""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import pybullet as p
import pybullet_data

from contact_grasp_probe import CONFIG


LAYOUT = {
    "version": "multi-scene-v1",
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
        bins[shape] = {"center": [x, y], "floor": floor, "parts": parts, "slots": slots, "floor_top": floor_top}
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
    return {"robot": robot, "table": table, "objects": objects, "bins": bins, "layout": layout, "home": home}


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
                   "layout": LAYOUT, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if result["settled"]:
            summary["objects"] = [{**o, "actual_position": p.getBasePositionAndOrientation(o["body"])[0]} for o in scene["objects"]]
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print("SUMMARY_PATH=" + str(output / "summary.json"), flush=True)
        print("낙하·정착 확인: " + ("PASS" if result["settled"] else "FAIL"), flush=True)
        if args.mode == "gui" and result["settled"]:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.6, 70, -50, [0.1, 0, 0.3])
            for _ in range(round(args.duration / CONFIG["dt_s"])):
                if not p.isConnected():
                    return 130
                p.stepSimulation()
                time.sleep(CONFIG["dt_s"])
        return 0 if result["settled"] else 1
    except KeyboardInterrupt:
        return 130
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    raise SystemExit(main())

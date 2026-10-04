"""바닥, Panda, 직육면체로 중력과 충돌을 확인하는 첫 PyBullet 실행."""

import argparse
import json
import math
import os
import sys
import time


# 물리 계산을 1초에 240번 진행한다.
TIME_STEP_S = 1.0 / 240.0


def positive_steps(value):
    steps = int(value)
    if steps <= 0:
        raise argparse.ArgumentTypeError("steps must be a positive integer")
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "gui"), default="direct")
    parser.add_argument("--steps", type=positive_steps, default=480)
    args = parser.parse_args()
    # Linux의 PyBullet GUI는 X11을 사용한다. Wayland에서는 XWayland DISPLAY 필요.
    if args.mode == "gui" and sys.platform.startswith("linux") and not os.getenv("DISPLAY"):
        parser.error("GUI needs DISPLAY; use --mode direct or run in a desktop terminal")

    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("PyBullet이 없습니다. .venv/bin/python -m pip install -r requirements-sim.txt", file=sys.stderr)
        return 2

    client = p.connect(p.GUI if args.mode == "gui" else p.DIRECT)
    if client < 0:
        print("PyBullet 연결 실패", file=sys.stderr)
        return 2

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setRealTimeSimulation(0, physicsClientId=client)
        p.setTimeStep(TIME_STEP_S, physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        ground = p.loadURDF("plane.urdf", physicsClientId=client)
        robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=client)

        # 초기 배치에서만 관절을 설정한다. 이후에는 모터로 같은 자세를 유지한다.
        home = {
            "panda_joint1": 0.0, "panda_joint2": -0.6, "panda_joint3": 0.0,
            "panda_joint4": -2.0, "panda_joint5": 0.0, "panda_joint6": 1.6,
            "panda_joint7": 0.8, "panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04,
        }
        joints = []
        for index in range(p.getNumJoints(robot, physicsClientId=client)):
            info = p.getJointInfo(robot, index, physicsClientId=client)
            name, link = info[1].decode(), info[12].decode()
            joints.append({"index": index, "joint": name, "link": link, "type": info[2]})
            if name in home:
                p.resetJointState(robot, index, home[name], physicsClientId=client)
                p.setJointMotorControl2(
                    robot, index, p.POSITION_CONTROL, targetPosition=home[name],
                    force=20 if "finger" in name else 200, physicsClientId=client,
                )

        # 상자 크기는 가로·세로·높이의 절반을 미터 단위로 넣는다.
        half_extents = [0.02, 0.015, 0.02]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client)
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.9, 0.15, 0.1, 1], physicsClientId=client,
        )
        block = p.createMultiBody(
            baseMass=0.05, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
            basePosition=[0.5, 0, 0.2], physicsClientId=client,
        )
        initial_z = p.getBasePositionAndOrientation(block, physicsClientId=client)[0][2]

        if args.mode == "gui":
            # 사람이 보는 화면의 시점이다. 학습용 RGBD 카메라는 아직 없다.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.4, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0.3, 0, 0.3], physicsClientId=client,
            )
        print(f"mode={args.mode}, requested_steps={args.steps}, dt={TIME_STEP_S:.6f}s", flush=True)
        print("PANDA_JOINTS_JSON=" + json.dumps(joints), flush=True)
        print("로봇은 대기 자세를 유지합니다. 빨간 물체의 낙하·바닥 충돌을 확인합니다.", flush=True)

        started = time.monotonic()
        # 한 번씩 물리를 계산한다. GUI에서는 눈으로 볼 수 있게 속도를 맞춘다.
        for _ in range(args.steps):
            if not p.isConnected(client):
                print("창이 닫혀 종료했습니다. 전체 step 검증은 완료되지 않았습니다.", flush=True)
                return 130
            p.stepSimulation(physicsClientId=client)
            if args.mode == "gui":
                time.sleep(TIME_STEP_S)

        position, _ = p.getBasePositionAndOrientation(block, physicsClientId=client)
        linear, angular = p.getBaseVelocity(block, physicsClientId=client)
        contacts = p.getContactPoints(block, ground, physicsClientId=client)
        # 바닥 접촉, 높이, 움직임을 함께 확인해 정착 여부를 판단한다.
        settled = (
            abs(position[2] - half_extents[2]) <= 0.005
            and len(contacts) > 0
            and math.dist(linear, (0, 0, 0)) < 0.01
            and math.dist(angular, (0, 0, 0)) < 0.01
        )
        result = {
            "mode": args.mode, "steps_completed": args.steps,
            "sim_duration_s": args.steps * TIME_STEP_S,
            "wall_duration_s": time.monotonic() - started,
            "initial_z_m": initial_z, "final_z_m": position[2],
            "ground_contact_count": len(contacts), "settled_on_ground": settled,
        }
        print("RESULT_JSON=" + json.dumps(result), flush=True)
        print("바닥 정착 확인: " + ("PASS" if settled else "FAIL"), flush=True)
        return 0 if settled else 1
    except KeyboardInterrupt:
        print("Ctrl+C로 중단했습니다. 연결을 해제합니다.", flush=True)
        return 130
    finally:
        if p.isConnected(client):
            p.disconnect(physicsClientId=client)


if __name__ == "__main__":
    raise SystemExit(main())

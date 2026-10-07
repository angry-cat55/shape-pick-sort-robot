"""PyBullet을 소유하고 촬영·접촉 운반 명령을 실행하는 노드."""

import json
import queue
import threading
from pathlib import Path

import pybullet as p
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile

from shape_sort_interfaces.msg import CameraObservation, TaskStatus
from shape_sort_interfaces.srv import RobotCommand
from multi_object_scene import (
    LAYOUT,
    add_camera_visual,
    build_scene,
    execute_pick,
    grasp_plan,
    recover_pick,
    sample_spawns,
    scene_probe,
    settle_scene,
    validate_counts,
    verify_final_arrivals,
)
from rgbd_camera import capture
from terminal_ko import failure_name
from shape_sort_ros.protocol import candidate_dict, image_message, validate_target


class SimulationNode(Node):
    def __init__(self):
        super().__init__("simulation_node")

        # 장면 설정은 ROS 파라미터로 받는다. 기본값은 기존 혼합 장면과 같다.
        self.declare_parameter("mode", "gui")
        self.declare_parameter("cuboids", 3)
        self.declare_parameter("cylinders", 2)
        self.declare_parameter("seed", 0)
        self.declare_parameter("size_mode", "random")
        self.declare_parameter("output_dir", "outputs/ros2-run")
        self.output = (
            Path(self.get_parameter("output_dir").value).expanduser().resolve()
        )
        self.output.mkdir(parents=True, exist_ok=True)

        # 상태의 최신 한 건은 늦게 켜진 작업 관리 노드도 받을 수 있게 유지한다.
        status_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(
            TaskStatus, "/shape_sort/robot_status", status_qos
        )
        self.camera_pub = self.create_publisher(
            CameraObservation, "/shape_sort/observation", 1
        )
        self.service = self.create_service(
            RobotCommand, "/shape_sort/robot_command", self.accept_command
        )

        # ROS 콜백은 접수만 하고, PyBullet은 이 작업 스레드 한 곳에서만 호출한다.
        self.commands = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.stop_event = threading.Event()
        self.ready = False
        self.busy = False
        self.last_command_id = 0
        self.last_scan_id = 0
        self.scan_used = True
        self.remaining_count = 0
        self.active_request = None
        self.failed_pick = None
        self.worker = threading.Thread(target=self.run_worker, name="pybullet-worker")
        self.worker.start()

    def publish_status(
        self, stage, completed=False, success=False, reason="", result=None
    ):
        # command_id를 그대로 돌려줘야 관리 노드가 어느 명령의 결과인지 구분한다.
        if not self.context.ok():
            return
        message = TaskStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.source = "simulation"
        if self.active_request is not None:
            message.command_id = self.active_request.command_id
            message.scan_id = self.active_request.scan_id
        message.stage = stage
        message.completed = completed
        message.success = success
        message.failure_reason = reason
        message.result_json = json.dumps(result or {}, allow_nan=False)
        self.status_pub.publish(message)

    def accept_command(self, request, response):
        # 정지는 동작 중에도 접수한다. 다음 물리 스텝에서 팔 이동을 중단한다.
        # 집게를 자동으로 열거나 물체를 원위치로 돌리는 명령은 아니다.
        with self.lock:
            if request.command_id <= self.last_command_id:
                response.reason = "중복되거나 오래된 명령 번호입니다"
                return response
            if request.command == "stop":
                self.last_command_id = request.command_id
                self.stop_event.set()
                response.accepted = True
                response.reason = "중단 요청을 접수했습니다"
                return response

            # 두 동작이 동시에 PyBullet을 만지지 않도록 하나가 끝나야 다음 명령을 받는다.
            if not self.ready or self.busy or self.stop_event.is_set():
                response.reason = "초기화 중이거나 동작 중 또는 중단된 장면입니다"
                return response
            if request.command not in ("scan", "pick", "recover", "wait", "finalize"):
                response.reason = "지원하지 않는 명령입니다"
                return response

            try:
                if request.command == "scan":
                    if request.scan_id <= self.last_scan_id:
                        raise ValueError("촬영 번호는 이전 번호보다 커야 합니다")
                if request.command == "pick":
                    if request.scan_id != self.last_scan_id or self.scan_used:
                        raise ValueError("현재 촬영의 미사용 목표만 집을 수 있습니다")
                    validate_target(candidate_dict(request.target))
                if request.command == "recover":
                    if self.failed_pick is None:
                        raise ValueError("복구할 집기 실패가 없습니다")
                    if request.failure_reason not in (
                        "lift_not_verified",
                        "drop_during_transport",
                    ):
                        raise ValueError("이 오류는 복구하지 않습니다")
                    if request.failure_reason != self.failed_pick["failure_reason"]:
                        raise ValueError("실제 실패 원인과 복구 요청이 다릅니다")
                if request.command == "finalize" and self.remaining_count:
                    raise ValueError("아직 운반하지 않은 물체가 있습니다")
            except (ValueError, KeyError, TypeError) as error:
                response.reason = str(error)
                return response

            self.last_command_id = request.command_id
            self.busy = True
            self.commands.put_nowait(request)
            response.accepted = True
            response.reason = "명령을 접수했습니다. 완료는 상태 토픽으로 전달합니다"
            return response

    def initialize_scene(self):
        # 초기 낙하·정착이 실패하면 기존 CLI와 같은 방식으로 다음 seed를 시도한다.
        mode = self.get_parameter("mode").value
        if mode not in ("gui", "direct"):
            raise ValueError("mode는 gui 또는 direct여야 합니다")
        self.gui = mode == "gui"
        size_mode = self.get_parameter("size_mode").value
        if size_mode not in ("fixed", "random"):
            raise ValueError("size_mode는 fixed 또는 random이어야 합니다")
        self.size_range = (0.03, 0.064, 0.03, 0.07) if size_mode == "random" else None
        cuboids = self.get_parameter("cuboids").value
        cylinders = self.get_parameter("cylinders").value
        validate_counts(cuboids, cylinders)
        seed = self.get_parameter("seed").value
        p.connect(p.GUI if self.gui else p.DIRECT)

        attempts = []
        for attempt in range(LAYOUT["max_scene_attempts"]):
            if self.shutdown_event.is_set() or self.stop_event.is_set():
                raise RuntimeError("task_stopped")
            try:
                spawns = sample_spawns(
                    cuboids,
                    cylinders,
                    seed + attempt,
                    LAYOUT["region"],
                    self.size_range,
                )
                self.scene = build_scene(spawns, LAYOUT)
                settled = settle_scene(self.scene)
            except ValueError as error:
                settled = {"settled": False, "failure_reason": str(error)}
            attempts.append({"seed": seed + attempt, **settled})
            if settled["settled"]:
                break
        else:
            raise RuntimeError("initial_scene_not_settled")
        if self.size_range is not None:
            p.setPhysicsEngineParameter(numSolverIterations=200)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.9, 70, -35, [0.15, 0, 0.55])
            add_camera_visual(self.scene["camera"])

        # 원본 정답은 시뮬레이션 내부 평가에만 둔다. 카메라 메시지에는 보내지 않는다.
        self.remaining = {obj["body"] for obj in self.scene["objects"]}
        self.remaining_count = len(self.remaining)
        self.placed_counts = {"cuboid": 0, "cylinder": 0}
        self.motion_log = (self.output / "motion.jsonl").open("w")
        self.probe = scene_probe(self.scene, self.motion_log, self.gui)
        self.probe.block = next(iter(self.remaining))

        # 기존 제어 함수를 재사용하되 매 물리 스텝 전에 정지 요청을 확인한다.
        original_step = self.probe.step

        def interruptible_step():
            if self.stop_event.is_set() or self.shutdown_event.is_set():
                raise RuntimeError("task_stopped")
            return original_step()

        self.probe.step = interruptible_step

        # 이동 세부 단계도 토픽으로 보낸다. 동작 완료와 단계 알림을 구분한다.
        original_start = self.probe.start_stage

        def announce_stage(stage):
            original_start(stage)
            self.publish_status(stage)

        self.probe.start_stage = announce_stage
        (self.output / "config.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "size_mode": size_mode,
                    "camera": self.scene["camera"],
                    "initial_seed": seed,
                    "attempts": attempts,
                },
                indent=2,
            )
            + "\n"
        )

    def scan(self, request):
        # 물체 하나를 옮기거나 복구한 뒤 대기 자세에서 한 번만 촬영한다.
        observation = capture(self.scene["camera"])
        message = CameraObservation()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera_optical"
        message.scan_id = request.scan_id
        message.rgb = image_message(observation["rgb"], message.header, "rgb8")
        message.depth = image_message(observation["depth"], message.header, "64FC1")
        message.empty_depth = image_message(
            self.scene["empty"]["depth"], message.header, "64FC1"
        )
        message.view = observation["view"]
        message.projection = observation["projection"]
        message.camera_json = json.dumps(self.scene["camera"])
        message.all_candidates = request.all_candidates
        message.random_size = self.size_range is not None
        message.size_range_m = list(self.size_range or (0.0, 0.0, 0.0, 0.0))
        with self.lock:
            self.last_scan_id = request.scan_id
            self.scan_used = False
        self.camera_pub.publish(message)
        return {"remaining_count": len(self.remaining)}

    def pick(self, request):
        # 목표 좌표는 인식 노드의 추정값이다. 평가 정답으로 좌표를 보정하지 않는다.
        geometry = candidate_dict(request.target)
        validate_target(geometry)
        pick = {
            "grasp_plan": grasp_plan(geometry),
            "lift_success": False,
            "arrival_success": False,
            "predicted_class": geometry["object_shape"],
            "class_score": geometry["class_score"],
            "target_bin": geometry["object_shape"],
            "scan_index": request.scan_id,
            "retry_index": request.retry_index,
            "failure_reason": "",
        }
        self.scan_used = True
        self.failed_pick = None
        try:
            execute_pick(
                self.probe,
                self.scene,
                geometry,
                pick,
                self.placed_counts,
                self.remaining,
            )
        except RuntimeError as error:
            pick["failure_reason"] = str(error)
            self.failed_pick = pick
            self.save_pick(pick)
            raise
        self.save_pick(pick)
        self.remaining_count = len(self.remaining)
        return {"pick": pick, "remaining_count": self.remaining_count}

    def save_pick(self, pick):
        # 실패한 시도도 남겨야 재시도 후 성공만 보고 성능을 과장하지 않는다.
        with (self.output / "picks.jsonl").open("a") as output:
            output.write(json.dumps(pick, allow_nan=False) + "\n")

    def execute_command(self, request):
        if request.command == "scan":
            return self.scan(request)
        if request.command == "pick":
            return self.pick(request)
        if request.command == "recover":
            recover_pick(
                self.probe,
                self.scene,
                self.failed_pick["grasp_plan"],
                request.failure_reason,
            )
            self.failed_pick = None
        elif request.command == "wait":
            self.probe.wait("REOBSERVE_WAIT", 0.5)
        elif request.command == "finalize":
            final = verify_final_arrivals(self.probe, self.scene)
            if not all(final.values()):
                raise RuntimeError("final_arrival_not_verified")
            return {"final_arrivals": final, "remaining_count": 0}
        return {"remaining_count": len(self.remaining)}

    def run_worker(self):
        try:
            self.initialize_scene()
            with self.lock:
                self.ready = True
            self.publish_status(
                "READY",
                completed=True,
                success=True,
                result={"remaining_count": self.remaining_count},
            )
            self.get_logger().info("장면 준비 완료: 시작 요청을 기다립니다")

            while not self.shutdown_event.is_set():
                try:
                    request = self.commands.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.active_request = request
                result = {}
                reason = ""
                try:
                    result = self.execute_command(request)
                except Exception as error:
                    reason = str(error)
                    result = {
                        "pick": self.failed_pick,
                        "remaining_count": len(self.remaining),
                    }
                    self.get_logger().error("물리 명령 실패: " + failure_name(reason))
                with self.lock:
                    self.busy = False
                self.publish_status(
                    request.command.upper() + "_DONE",
                    completed=True,
                    success=not reason,
                    reason=reason,
                    result=result,
                )
        except Exception as error:
            self.publish_status(
                "INITIALIZATION_FAILED", completed=True, reason=str(error)
            )
            self.get_logger().error("시뮬레이션 초기화 실패: " + failure_name(str(error)))
        finally:
            if p.isConnected():
                p.disconnect()
            if hasattr(self, "motion_log"):
                self.motion_log.close()

    def close(self):
        # 프로세스를 끝낼 때 작업 스레드도 멈추고 PyBullet 연결을 정리한다.
        self.shutdown_event.set()
        self.worker.join(timeout=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

"""대상 선택·순차 운반·제한된 재시도를 관리하는 노드."""

import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger
from shape_sort_interfaces.msg import PerceptionResult, TaskStatus
from shape_sort_interfaces.srv import RobotCommand

from terminal_ko import failure_name
from shape_sort_ros.protocol import candidate_dict, candidate_message, choose_target


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__("task_manager_node")
        self.declare_parameter("auto_start", False)
        self.declare_parameter("max_retries", 2)
        self.declare_parameter("timeout_s", 90.0)
        self.declare_parameter("output_dir", "outputs/ros2-run")
        self.max_retries = self.get_parameter("max_retries").value
        self.timeout = float(self.get_parameter("timeout_s").value)
        if (
            type(self.max_retries) is not int
            or not 0 <= self.max_retries <= 2
            or self.timeout <= 0
        ):
            raise ValueError("재시도는 0~2회, 통신 제한 시간은 양수여야 합니다")
        self.output = (
            Path(self.get_parameter("output_dir").value).expanduser().resolve()
        )
        self.output.mkdir(parents=True, exist_ok=True)

        # 노드마다 할 일을 나눈다. 관리 노드는 PyBullet이나 CNN을 실행하지 않는다.
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(
            TaskStatus, "/shape_sort/task_status", qos
        )
        self.robot_sub = self.create_subscription(
            TaskStatus, "/shape_sort/robot_status", self.on_robot_status, qos
        )
        self.perception_sub = self.create_subscription(
            PerceptionResult, "/shape_sort/perception", self.on_perception, 1
        )
        self.robot_client = self.create_client(
            RobotCommand, "/shape_sort/robot_command"
        )
        self.perception_client = self.create_client(
            Trigger, "/shape_sort/perception_ready"
        )
        self.start_service = self.create_service(
            Trigger, "/shape_sort/start", self.start_task
        )
        self.stop_service = self.create_service(
            Trigger, "/shape_sort/stop", self.stop_task
        )

        # 한 번에 한 명령만 기다린다. 번호가 다른 결과는 새 작업에 섞지 않는다.
        self.robot_ready = False
        self.all_ready = False
        self.phase = "idle"
        self.used = False
        self.pending = None
        self.command_id = 0
        self.scan_id = 0
        self.retry_count = 0
        self.retry_xy = None
        self.selected = None
        self.remaining_count = 0
        self.early_perception = None
        self.deadline = time.monotonic() + self.timeout
        self.report = {
            "success": False,
            "failure_reason": "",
            "scans": [],
            "picks": [],
            "recoveries": [],
            "max_retries": self.max_retries,
        }
        self.timer = self.create_timer(0.1, self.check_progress)

    def publish_status(self, stage, completed=False, success=False, reason=""):
        if not self.context.ok():
            return
        message = TaskStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.source = "task_manager"
        message.command_id = self.command_id
        message.scan_id = self.scan_id
        message.stage = stage
        message.completed = completed
        message.success = success
        message.failure_reason = reason
        message.result_json = json.dumps(
            {"retry_index": self.retry_count, "remaining_count": self.remaining_count}
        )
        self.status_pub.publish(message)

    def start_task(self, request, response):
        # 서비스 응답은 시작 접수다. 전체 운반 성공은 task_status에서 따로 확인한다.
        if self.used or self.phase != "idle":
            response.message = (
                "이미 시작한 장면입니다. 새 작업은 노드를 다시 실행하세요"
            )
        elif not self.robot_ready or not self.perception_client.service_is_ready():
            response.message = "시뮬레이션 또는 CNN 노드가 아직 준비되지 않았습니다"
        else:
            self.used = True
            response.success = True
            response.message = "작업 시작을 접수했습니다"
            self.request_scan()
        return response

    def stop_task(self, request, response):
        if self.phase == "finished":
            response.message = "이미 종료된 작업입니다"
            return response
        response.success = True
        response.message = (
            "중단 요청을 접수했습니다. 집게 자동 개방이나 복귀는 하지 않습니다"
        )
        self.used = True
        self.finish(False, "task_stopped")
        return response

    def check_progress(self):
        if self.phase == "finished":
            return
        prepared_idle = (
            self.phase == "idle"
            and self.robot_ready
            and self.perception_client.service_is_ready()
        )
        # 장면만 준비된 상태와 CNN까지 준비된 상태를 구분해 시작 가능 여부를 알린다.
        if prepared_idle and not self.all_ready:
            self.all_ready = True
            self.publish_status("READY")
        if not prepared_idle and time.monotonic() > self.deadline:
            self.finish(False, "communication_timeout")
            return
        if self.phase == "idle" and self.get_parameter("auto_start").value:
            if (
                self.robot_ready
                and self.perception_client.service_is_ready()
                and self.robot_client.service_is_ready()
            ):
                self.start_task(Trigger.Request(), Trigger.Response())

    def send_command(self, command, cause=""):
        # call_async로 접수를 요청하고 즉시 돌아온다. 콜백 안에서 운반 완료를 기다리지 않는다.
        if not self.robot_client.service_is_ready():
            self.finish(False, "simulation_unavailable")
            return
        self.command_id += 1
        request = RobotCommand.Request()
        request.command_id = self.command_id
        request.scan_id = self.scan_id
        request.command = command
        request.retry_index = self.retry_count
        request.all_candidates = self.retry_xy is not None
        request.failure_reason = cause
        if self.selected is not None:
            request.target = candidate_message(self.selected)
        self.pending = request
        self.phase = command
        self.deadline = time.monotonic() + self.timeout
        self.publish_status(command.upper())
        future = self.robot_client.call_async(request)
        future.add_done_callback(
            lambda future: self.on_accepted(request.command_id, future)
        )

    def on_accepted(self, command_id, future):
        # 완료 상태가 응답보다 먼저 도착할 수 있다. 이미 넘어간 명령이면 무시한다.
        if (
            self.phase == "finished"
            or self.pending is None
            or self.pending.command_id != command_id
        ):
            return
        try:
            response = future.result()
            if not response.accepted:
                self.finish(False, "command_rejected: " + response.reason)
        except Exception as error:
            self.finish(False, "command_service_failed: " + str(error))

    def request_scan(self):
        self.scan_id += 1
        self.early_perception = None
        self.send_command("scan")

    def on_perception(self, message):
        # 이전 영상의 늦은 결과가 운반 뒤 새 좌표로 사용되지 않도록 번호를 확인한다.
        if self.phase not in ("scan", "perception") or message.scan_id != self.scan_id:
            return
        if self.phase == "scan":
            self.early_perception = message
        else:
            self.handle_perception(message)

    def handle_perception(self, message):
        if not message.valid:
            self.finish(False, "perception_failed: " + message.failure_reason)
            return
        try:
            candidates = [candidate_dict(candidate) for candidate in message.candidates]
            self.report["scans"].append(
                {
                    "scan_id": message.scan_id,
                    "retry_index": self.retry_count,
                    "cnn_evaluations": sum(
                        candidate["cnn_evaluated"] for candidate in candidates
                    ),
                    "regions": candidates,
                }
            )
            if any(
                candidate.get("failure_reason") == "invalid_depth"
                for candidate in candidates
            ):
                self.finish(False, "invalid_depth")
                return
            geometry = choose_target(candidates, self.retry_xy)
            if geometry is not None:
                self.selected = geometry
                self.send_command("pick")
            elif not self.remaining_count:
                self.send_command("finalize")
            elif self.retry_xy is not None:
                self.finish(False, "retry_target_not_found")
            elif self.retry_count >= self.max_retries:
                reason = (
                    "retry_limit_reached" if self.max_retries else "no_valid_target"
                )
                self.finish(False, reason)
            else:
                self.retry_count += 1
                self.report["recoveries"].append(
                    {
                        "cause": "no_valid_target",
                        "action": "wait_and_rescan",
                        "retry_index": self.retry_count,
                    }
                )
                self.send_command("wait")
        except (RuntimeError, ValueError, KeyError) as error:
            self.finish(False, str(error))

    def on_robot_status(self, message):
        # 준비 상태는 명령 번호0이다. 이후에는 현재 기다리는 명령의 완료만 처리한다.
        if message.stage == "INITIALIZATION_FAILED":
            self.finish(False, "initialization_failed: " + message.failure_reason)
            return
        if message.stage == "READY":
            self.robot_ready = message.success
            self.remaining_count = json.loads(message.result_json)["remaining_count"]
            self.deadline = time.monotonic() + self.timeout
            self.publish_status("ROBOT_READY")
            return
        if self.phase == "finished" or self.pending is None:
            return
        if (
            message.command_id != self.pending.command_id
            or message.scan_id != self.pending.scan_id
            or not message.completed
        ):
            return

        command = self.pending.command
        self.pending = None
        result = json.loads(message.result_json or "{}")
        self.remaining_count = result.get("remaining_count", self.remaining_count)
        if command == "pick":
            self.report["picks"].append(result.get("pick"))
        if not message.success:
            self.handle_failure(command, message.failure_reason)
        elif command == "scan":
            self.phase = "perception"
            if self.early_perception is not None:
                self.handle_perception(self.early_perception)
        elif command == "pick":
            self.retry_count = 0
            self.retry_xy = None
            self.request_scan()
        elif command == "recover":
            self.report["recoveries"][-1]["status"] = "recovered"
            if self.stop_after_recovery:
                self.finish(False, "retry_limit_reached")
            else:
                self.retry_xy = self.selected["center_xy_m"]
                self.request_scan()
        elif command == "wait":
            self.request_scan()
        elif command == "finalize":
            self.report["final_arrivals"] = result["final_arrivals"]
            self.finish(True)

    def handle_failure(self, command, reason):
        # 충돌 등은 즉시 중단한다. 집기 실패·운반 중 낙하 두 경우만 복구를 허용한다.
        allowed = ("lift_not_verified", "drop_during_transport")
        if command != "pick" or reason not in allowed or self.max_retries == 0:
            if command == "recover":
                self.report["recoveries"][-1].update(
                    status="failed", failure_reason=reason
                )
                reason = "recovery_failed"
            self.finish(False, reason)
            return
        self.stop_after_recovery = self.retry_count >= self.max_retries
        if not self.stop_after_recovery:
            self.retry_count += 1
        self.report["recoveries"].append(
            {"cause": reason, "retry_index": self.retry_count, "status": "recovering"}
        )
        self.send_command("recover", reason)

    def finish(self, success, reason=""):
        if self.phase == "finished":
            return
        self.phase = "finished"
        self.pending = None
        self.report.update(success=success, failure_reason=reason)
        (self.output / "sort_result.json").write_text(
            json.dumps(self.report, indent=2) + "\n"
        )
        self.publish_status("FINISHED", completed=True, success=success, reason=reason)
        self.get_logger().info(
            "순차 운반 성공" if success else "작업 종료: " + failure_name(reason)
        )
        if not success and self.context.ok() and self.robot_client.service_is_ready():
            # 통신 시간 초과로 관리 노드가 포기했을 때도 팔이 계속 움직이지 않도록 중단한다.
            self.command_id += 1
            stop = RobotCommand.Request(command_id=self.command_id, command="stop")
            self.robot_client.call_async(stop)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TaskManagerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node is not None and node.used and node.phase != "finished":
            node.finish(False, "task_stopped")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

"""촬영 메시지에서 객체를 분리하고 CNN·기하 계산 결과를 보내는 노드."""

import json
from pathlib import Path

from PIL import Image
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_srvs.srv import Trigger
from shape_sort_interfaces.msg import CameraObservation, PerceptionResult

from cnn_inference import CnnClassifier
from rgbd_camera import estimate_many, rgb_crop, save_observation
from shape_sort_ros.protocol import candidate_message, image_array

from shape_sort_ros.protocol import log_communication


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.declare_parameter("model_dir", "checkpoints/shape_cnn_v1")
        self.declare_parameter("output_dir", "outputs/ros2-run")
        self.declare_parameter("min_class_score", 0.8)
        self.output = (
            Path(self.get_parameter("output_dir").value).expanduser().resolve()
        )
        self.min_class_score = self.get_parameter("min_class_score").value
        if not 0.5 <= self.min_class_score <= 1.0:
            raise ValueError("분류 점수 기준은 0.5~1이어야 합니다")

        # 가중치는 시작할 때 한 번만 불러온다. 매 촬영마다 학습하거나 다시 로드하지 않는다.
        self.classifier = CnnClassifier(self.get_parameter("model_dir").value)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "inference_config.json").write_text(
            json.dumps(
                {
                    "model_sha256": self.classifier.model_sha256,
                    "params_sha256": self.classifier.params_sha256,
                    "metadata": self.classifier.metadata,
                    "min_class_score": self.min_class_score,
                },
                indent=2,
            )
            + "\n"
        )
        self.publisher = self.create_publisher(
            PerceptionResult, "/shape_sort/perception", 1
        )
        self.subscription = self.create_subscription(
            CameraObservation,
            "/shape_sort/observation",
            self.on_observation,
            1,
        )
        self.last_scan_id = 0
        # 모델 로드와 구독 준비가 끝난 뒤에만 준비 서비스를 공개한다.
        self.ready_service = self.create_service(
            Trigger, "/shape_sort/perception_ready", self.on_ready
        )
        self.get_logger().info("CNN 준비 완료: 카메라 관측을 기다립니다")

    def on_ready(self, request, response):
        log_communication(
            self, "서비스", "/shape_sort/perception_ready", "요청 수신", "CNN 준비 확인"
        )
        response.success = True
        response.message = "CNN과 카메라 구독이 준비되었습니다"
        log_communication(
            self,
            "서비스",
            "/shape_sort/perception_ready",
            "응답 준비",
            response.message,
        )
        return response

    def on_observation(self, message):
        log_communication(
            self, "토픽", "/shape_sort/observation", "수신", f"촬영 {message.scan_id}"
        )
        # 중복 관측을 다시 분류하지 않는다. 물체 이동 뒤에는 반드시 새 번호를 받는다.
        if message.scan_id <= self.last_scan_id:
            return
        self.last_scan_id = message.scan_id
        result = PerceptionResult()
        result.header = message.header
        result.header.frame_id = "world"
        result.scan_id = message.scan_id

        try:
            # ROS 영상 바이트를 기존 함수가 받던 RGB·깊이 배열로 되돌린다.
            camera = json.loads(message.camera_json)
            observation = {
                "rgb": image_array(message.rgb, "rgb8"),
                "depth": image_array(message.depth, "64FC1"),
                "view": list(message.view),
                "projection": list(message.projection),
            }
            empty = {"depth": image_array(message.empty_depth, "64FC1")}
            self.classifier.validate_camera(camera)
            size_range = tuple(message.size_range_m) if message.random_size else None

            # 기존 탐지·분류 순서를 그대로 쓴다. 재시도 때만 전체 후보를 확인한다.
            candidates = estimate_many(
                observation,
                empty,
                camera,
                None,
                classifier=self.classifier,
                min_class_score=self.min_class_score,
                first_valid_only=not message.all_candidates,
                size_range_m=size_range,
            )
            result.candidates = [
                candidate_message(candidate) for candidate in candidates
            ]
            result.valid = True

            # 어떤 영상을 어떻게 판단했는지 나중에 확인할 수 있게 관측별로 저장한다.
            folder = self.output / f"scan_{message.scan_id:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            save_observation(folder, observation, empty, {"valid": False}, camera)
            for index, candidate in enumerate(candidates):
                if "mask" in candidate:
                    rgb_crop(observation["rgb"], candidate["mask"]).save(
                        folder / f"crop_{index + 1}.png"
                    )
                    Image.fromarray(candidate["mask"].astype("uint8") * 255).save(
                        folder / f"mask_{index + 1}.png"
                    )
            details = [
                {key: value for key, value in candidate.items() if key != "mask"}
                for candidate in candidates
            ]
            (folder / "regions.json").write_text(json.dumps(details, indent=2) + "\n")
            evaluations = sum(
                candidate.get("cnn_evaluated", False) for candidate in candidates
            )
            self.get_logger().info(
                f"촬영 {message.scan_id}: 후보 {len(candidates)}개, CNN 분류 {evaluations}회"
            )
        except Exception as error:
            # 변환·카메라·CNN 오류를 유효한 좌표로 위장하지 않고 관리 노드에 전달한다.
            result.valid = False
            result.failure_reason = str(error)
            self.get_logger().error("인식 실패: " + str(error))
        self.publisher.publish(result)
        log_communication(
            self,
            "토픽",
            "/shape_sort/perception",
            "발행",
            f"촬영 {result.scan_id} · 후보 {len(result.candidates)}개",
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

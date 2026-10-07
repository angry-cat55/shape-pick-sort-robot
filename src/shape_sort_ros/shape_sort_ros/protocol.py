"""영상과 후보를 ROS 메시지로 바꾸고 동작 입력을 확인한다."""

import json
import math
import sys

import numpy as np
from sensor_msgs.msg import Image
from shape_sort_interfaces.msg import ObjectCandidate

from multi_object_scene import LAYOUT

# depth는 미터 거리가 아니다. 기존 카메라의 0~1 buffer 값을 그대로 전달한다.
IMAGE_FORMATS = {
    "rgb8": (np.dtype("uint8"), 3),
    "64FC1": (np.dtype("float64"), 1),
}


def image_message(array, header, encoding):
    # 픽셀 자료형과 메모리 순서를 맞춘 뒤 ROS가 전달할 바이트로 바꾼다.
    dtype, channels = IMAGE_FORMATS[encoding]
    array = np.ascontiguousarray(array, dtype=dtype)
    expected_dims = 3 if channels == 3 else 2
    if array.ndim != expected_dims or (channels == 3 and array.shape[2] != 3):
        raise ValueError("영상 배열의 차원이 맞지 않습니다")

    message = Image()
    message.header = header
    message.height, message.width = array.shape[:2]
    message.encoding = encoding
    message.is_bigendian = int(sys.byteorder == "big")
    message.step = message.width * channels * dtype.itemsize
    message.data = array.tobytes()
    return message


def image_array(message, encoding):
    # 영상 형식·한 줄 길이·전체 바이트 수를 확인한다. 깨진 영상은 좌표 계산에 넘기지 않는다.
    dtype, channels = IMAGE_FORMATS[encoding]
    row_bytes = message.width * channels * dtype.itemsize
    if (
        message.encoding != encoding
        or message.height == 0
        or message.width == 0
        or message.step != row_bytes
        or len(message.data) != row_bytes * message.height
    ):
        raise ValueError("영상 형식 또는 데이터 길이가 맞지 않습니다")

    byteorder = ">" if message.is_bigendian else "<"
    dtype = dtype.newbyteorder(byteorder)
    shape = (message.height, message.width)
    if channels == 3:
        shape += (3,)
    return np.frombuffer(bytes(message.data), dtype=dtype).reshape(shape).copy()


def candidate_message(geometry):
    # 좌표·종류는 자료형이 정해진 필드에, 부가 측정값은 기록용 JSON에 담는다.
    details = {key: value for key, value in geometry.items() if key != "mask"}
    message = ObjectCandidate()
    message.valid = bool(geometry["valid"])
    message.failure_reason = geometry.get("failure_reason", "")
    message.object_shape = geometry.get("object_shape", "")
    message.center_xy_m = geometry.get("center_xy_m", [0.0, 0.0])
    message.top_z_m = float(geometry.get("top_z_m", 0.0))
    message.width_m = float(geometry.get("width_m", 0.0))
    message.yaw_rad = float(geometry.get("yaw_rad", 0.0))
    message.top_pixels = int(geometry.get("top_pixels", 0))
    message.cnn_evaluated = bool(geometry.get("cnn_evaluated", False))
    message.class_score = float(geometry.get("class_score", 0.0))
    message.details_json = json.dumps(details, allow_nan=False)
    return message


def candidate_dict(message):
    # 로봇이 쓰는 주요 값은 기록용 JSON보다 메시지의 좌표·종류 필드를 우선한다.
    geometry = json.loads(message.details_json or "{}")
    if not isinstance(geometry, dict):
        raise ValueError("후보 부가 정보는 JSON 객체여야 합니다")
    geometry.update(
        valid=message.valid,
        failure_reason=message.failure_reason,
        object_shape=message.object_shape,
        center_xy_m=list(message.center_xy_m),
        top_z_m=message.top_z_m,
        width_m=message.width_m,
        yaw_rad=message.yaw_rad,
        top_pixels=message.top_pixels,
        cnn_evaluated=message.cnn_evaluated,
        class_score=message.class_score,
    )
    return geometry


def validate_target(geometry, min_class_score=0.8):
    # 외부 서비스에서 받은 목표도 검사한다. NaN·범위 밖·모르는 종류면 팔을 움직이지 않는다.
    if not geometry["valid"] or geometry["object_shape"] not in ("cuboid", "cylinder"):
        raise ValueError("유효한 도형 목표가 아닙니다")
    values = [
        *geometry["center_xy_m"],
        geometry["top_z_m"],
        geometry["width_m"],
        geometry["yaw_rad"],
        geometry["class_score"],
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("목표 좌표 또는 점수가 유한수가 아닙니다")
    x, y = geometry["center_xy_m"]
    if (
        not LAYOUT["region"]["x"][0] <= x <= LAYOUT["region"]["x"][1]
        or not LAYOUT["region"]["y"][0] <= y <= LAYOUT["region"]["y"][1]
        or not 0.315 <= geometry["top_z_m"] <= 0.38
        or not 0.01 <= geometry["width_m"] <= 0.065
        or not min_class_score <= geometry["class_score"] <= 1.0
    ):
        raise ValueError("목표 위치·크기 또는 분류 점수가 허용 범위를 벗어났습니다")


def choose_target(candidates, retry_xy=None):
    # 평소에는 윗면이 잘 보이는 유효 후보를 고른다. 재시도는 이전 중심에서 3cm 이내로 제한한다.
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        return None
    if retry_xy is None:
        return max(valid, key=lambda candidate: candidate["top_pixels"])

    nearby = sorted(
        valid,
        key=lambda candidate: math.dist(candidate["center_xy_m"], retry_xy),
    )
    if math.dist(nearby[0]["center_xy_m"], retry_xy) > 0.03:
        raise RuntimeError("retry_target_not_found")
    if len(nearby) > 1 and math.dist(nearby[1]["center_xy_m"], retry_xy) <= 0.03:
        raise RuntimeError("retry_target_ambiguous")
    return nearby[0]


def log_communication(node, kind, name, event, detail):
    # 영상·좌표 원문 대신 식별 번호와 동작만 기록해 GUI에서 통신 흐름을 읽게 한다.
    node.get_logger().info(f"[ROS통신] [{kind}] {event} {name} | {detail}")

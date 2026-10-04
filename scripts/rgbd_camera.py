"""고정 RGB-D 촬영과 도형별 윗면 기하 추정. 정답 pose·크기는 받지 않는다."""

from collections import deque
import math

import numpy as np
from PIL import Image
import pybullet as p


# 대각선 위에서 물체 상면과 옆면을 본다. 학습과 추론에서도 같은 설정을 사용한다.
CAMERA = {
    "width": 640, "height": 480, "eye": [0.8, -0.55, 1.05],
    "target": [0.5, 0, 0.30], "up": [0, 0, 1],
    "fov_deg": 50, "near_m": 0.05, "far_m": 2.0,
    "roi_xy_m": [[0.44, 0.56], [-0.065, 0.065]], "table_top_m": 0.30,
    "depth_difference_m": 0.005, "top_band_m": 0.002,
    "renderer": "ER_TINY_RENDERER", "crop_size": 64,
}


def camera_for_angle(angle=50):
    # 관찰 중심까지의 거리는 1m로 유지하고 위에서 내려다보는 각도만 바꾼다.
    camera = dict(CAMERA)
    elevation = math.radians(angle)
    direction = math.atan2(-0.55, 0.3)
    camera["eye"] = [0.5 + math.cos(elevation) * math.cos(direction),
                     math.cos(elevation) * math.sin(direction), 0.30 + math.sin(elevation)]
    if angle == 90:
        camera["up"] = [0, 1, 0]
    return camera


# 단독 촬영·테스트·집기 실행 모두 같은 기본 카메라를 사용한다.
CAMERA = camera_for_angle()


def capture(camera):
    # GUI 화면 시점과 별개로, 항상 이 view/projection으로 촬영한다.
    view = p.computeViewMatrix(camera["eye"], camera["target"], camera["up"])
    projection = p.computeProjectionMatrixFOV(
        camera["fov_deg"], camera["width"] / camera["height"], camera["near_m"], camera["far_m"],
    )
    frame = p.getCameraImage(
        camera["width"], camera["height"], view, projection,
        renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK,
    )
    # RGB·깊이만 사용한다. 시뮬레이터의 물체 ID 마스크는 요청하지 않는다.
    return {
        "rgb": np.asarray(frame[2], dtype=np.uint8).reshape(camera["height"], camera["width"], 4)[..., :3],
        "depth": np.asarray(frame[3], dtype=np.float64).reshape(camera["height"], camera["width"]),
        "view": view, "projection": projection,
    }


def world_points(observation):
    # OpenGL의 0~1 depth buffer와 픽셀 좌표를 -1~1 좌표로 바꾼다.
    depth = observation["depth"]
    height, width = depth.shape
    v, u = np.indices(depth.shape)
    # TinyRenderer는 정수 픽셀을 쓴다. 아래쪽 원점으로 행을 뒤집으면 height-1-v가 된다.
    ndc = np.stack((2 * u / width - 1, 1 - 2 * (v + 1) / height,
                    2 * depth - 1, np.ones_like(depth)), axis=-1)
    # PyBullet 행렬은 열 우선이다. 투영·카메라 변환을 거꾸로 적용하면 월드 좌표가 된다.
    view = np.asarray(observation["view"]).reshape(4, 4, order="F")
    projection = np.asarray(observation["projection"]).reshape(4, 4, order="F")
    homogeneous = ndc @ np.linalg.inv(projection @ view).T
    return homogeneous[..., :3] / homogeneous[..., 3:4]


def depth_metres(buffer, camera):
    # depth buffer는 미터가 아니다. 카메라 앞쪽 방향의 거리로 변환한다.
    near, far = camera["near_m"], camera["far_m"]
    return near * far / (far - (far - near) * buffer)


def components(mask):
    # 연결된 픽셀을 묶는다. 물체가 둘 이상이면 단일 물체 집기를 중단한다.
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        first = remaining.pop()
        queue, group = deque([first]), [first]
        while queue:
            v, u = queue.popleft()
            for neighbour in ((v - 1, u), (v + 1, u), (v, u - 1), (v, u + 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
                    group.append(neighbour)
        if len(group) >= 30:
            groups.append(group)
    return groups


def cuboid_geometry(xy):
    # 직육면체는 긴 방향에 맞춰 손끝을 돌려야 한다. PCA로 그 방향을 구한다.
    _, vectors = np.linalg.eigh(np.cov(xy.T))
    major = vectors[:, -1]
    yaw = float(np.arctan2(major[1], major[0]) % np.pi)
    if yaw > np.pi / 2:
        yaw -= np.pi
    axes = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    local = xy @ axes
    # 실제 물체를 돌리지 않고 좌표의 축만 맞춘 뒤 양 끝의 중앙·짧은 폭을 구한다.
    low, high = local.min(axis=0), local.max(axis=0)
    center = ((low + high) / 2) @ axes.T
    return {"valid": True, "center_xy_m": center.tolist(),
            "width_m": float(high[1] - low[1]), "yaw_rad": yaw}


def cylinder_geometry(xy):
    # 수직 원기둥의 원형 윗면에는 긴 방향이 없다. X·Y 양 끝의 중앙을 쓴다.
    low, high = xy.min(axis=0), xy.max(axis=0)
    diameters = high - low
    diameter = float(diameters.mean())
    # 원형이면 X·Y 지름이 비슷해야 한다. 큰 차이는 잘림·기울어짐 등의 후보로 거부한다.
    if diameter <= 0 or abs(diameters[0] - diameters[1]) > 0.20 * diameter:
        return {"valid": False, "failure_reason": "non_circular_top"}
    # ponytail: 전체 윗면이 보이는 수직 원기둥에 한정한다. 가려진 원은 별도 추정이 필요하다.
    return {"valid": True, "center_xy_m": ((low + high) / 2).tolist(),
            "width_m": diameter, "diameter_xy_m": diameters.tolist(), "yaw_rad": 0.0}


def estimate(observation, empty, camera, object_shape="cuboid"):
    # 종류는 현재 명령에서 지정한다. 이후 CNN 예측을 이 입력에 연결할 수 있다.
    if object_shape not in ("cuboid", "cylinder"):
        return {"valid": False, "failure_reason": "unsupported_shape"}
    depth = observation["depth"]
    # 고장난 깊이나 다른 크기의 기준 영상으로 로봇을 움직이지 않는다.
    if (depth.shape != (camera["height"], camera["width"])
            or depth.shape != empty["depth"].shape
            or not np.isfinite(depth).all() or not np.isfinite(empty["depth"]).all()
            or np.any((depth < 0) | (depth > 1))):
        return {"valid": False, "failure_reason": "invalid_depth"}
    points = world_points(observation)
    difference = depth_metres(empty["depth"], camera) - depth_metres(depth, camera)
    mask = difference > camera["depth_difference_m"]
    for axis, (low, high) in enumerate(camera["roi_xy_m"]):
        mask &= (points[..., axis] > low) & (points[..., axis] < high)
    mask &= (points[..., 2] > camera["table_top_m"] + 0.008) & (depth < 1)
    groups = components(mask)
    if len(groups) != 1:
        return {"valid": False, "failure_reason": "no_object" if not groups else "multiple_objects"}
    mask[:] = False
    pixels = np.asarray(groups[0])
    mask[pixels[:, 0], pixels[:, 1]] = True
    surface = points[mask]
    # 옆면까지 평균 내면 중심이 카메라 쪽으로 치우친다. 가장 높은 상면만 따로 쓴다.
    top_z = float(np.percentile(surface[:, 2], 95))
    top = surface[np.abs(surface[:, 2] - top_z) <= camera["top_band_m"]]
    if len(top) < 30:
        return {"valid": False, "failure_reason": "insufficient_top_surface"}
    # 공통 물체 영역·윗면 선택 뒤 도형별 계산만 나눈다.
    geometry = cylinder_geometry(top[:, :2]) if object_shape == "cylinder" else cuboid_geometry(top[:, :2])
    if not geometry["valid"]:
        return geometry
    if not 0.01 <= geometry["width_m"] <= 0.065 or not 0.015 <= top_z - camera["table_top_m"] <= 0.08:
        return {"valid": False, "failure_reason": "unsupported_geometry"}
    # 영역 경계에 닿으면 잘린 물체일 수 있으므로 완전한 중심을 구했다고 주장하지 않는다.
    for axis, (roi_low, roi_high) in enumerate(camera["roi_xy_m"]):
        if surface[:, axis].min() < roi_low + 0.002 or surface[:, axis].max() > roi_high - 0.002:
            return {"valid": False, "failure_reason": "object_at_roi_edge"}
    return {
        **geometry, "object_shape": object_shape, "top_z_m": top_z,
        "height_m": top_z - camera["table_top_m"],
        "object_pixels": int(mask.sum()), "top_pixels": len(top), "mask": mask,
    }


def estimate_many(observation, empty, camera, object_shape):
    # 생성 목록 없이 깊이 차이로 영역을 분리한다. 좌표 계산은 기존 함수를 재사용한다.
    depth = observation["depth"]
    if (depth.shape != (camera["height"], camera["width"]) or depth.shape != empty["depth"].shape
            or any(not np.isfinite(d).all() or np.any((d < 0) | (d > 1)) for d in (depth, empty["depth"]))):
        return [{"valid": False, "failure_reason": "invalid_depth"}]
    points = world_points(observation)
    mask = depth_metres(empty["depth"], camera) - depth_metres(depth, camera) > camera["depth_difference_m"]
    for axis, (low, high) in enumerate(camera["roi_xy_m"]):
        mask &= (points[..., axis] > low) & (points[..., axis] < high)
    mask &= (points[..., 2] > camera["table_top_m"] + 0.008) & (depth < 1)
    results = []
    for group in components(mask):
        pixels = np.asarray(group)
        region = np.zeros_like(mask)
        region[pixels[:, 0], pixels[:, 1]] = True
        isolated = dict(observation)
        isolated["depth"] = np.where(region, depth, empty["depth"])
        geometry = estimate(isolated, empty, camera, object_shape)
        if geometry["valid"]:
            top = points[region]
            top = top[np.abs(top[:, 2] - geometry["top_z_m"]) <= camera["top_band_m"]]
            yaw = geometry["yaw_rad"]
            axes = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            spans = np.ptp(top[:, :2] @ axes, axis=0)
            # 이번 장면의 지원 크기와 비교한다. 잘린 윗면·붙은 영역의 일부를 보류할 수 있다.
            # 개별 생성 정답은 읽지 않는다. 임의 크기와 모든 가림을 판별하는 검사는 아니다.
            expected = np.array([0.06, 0.04] if object_shape == "cuboid" else [0.05, 0.05])
            if np.any(np.abs(spans - expected) > expected * 0.15):
                geometry = {"valid": False, "failure_reason": "incomplete_or_merged_surface"}
        results.append({**geometry, "mask": region})
    return results


def rgb_crop(rgb, mask, size=64):
    # 같은 영역·여백·크기 변환을 이후 학습 데이터와 로컬 추론에서 재사용한다.
    v, u = np.where(mask)
    side = max(int(u.max() - u.min() + 1), int(v.max() - v.min() + 1)) + 12
    x = int((u.min() + u.max() - side) / 2)
    y = int((v.min() + v.max() - side) / 2)
    return Image.fromarray(rgb).crop((x, y, x + side, y + side)).resize((size, size), Image.Resampling.BILINEAR)


def save_observation(output, observation, empty, result, camera):
    # 원본 RGB와 깊이를 보관해 추정 결과를 다시 계산할 수 있게 한다.
    Image.fromarray(observation["rgb"]).save(output / "rgb.png")
    # 보기용 깊이 그림은 가까울수록 밝다. 계산에는 원본 float depth를 사용한다.
    metres = depth_metres(observation["depth"], camera)
    grey = np.clip(255 * (camera["far_m"] - metres) / (camera["far_m"] - camera["near_m"]), 0, 255)
    Image.fromarray(grey.astype(np.uint8)).save(output / "depth.png")
    np.savez_compressed(output / "rgbd.npz", depth=observation["depth"], empty_depth=empty["depth"],
                        view=observation["view"], projection=observation["projection"])
    if result["valid"]:
        Image.fromarray(result["mask"].astype(np.uint8) * 255).save(output / "mask.png")
        rgb_crop(observation["rgb"], result["mask"], camera["crop_size"]).save(output / "crop.png")

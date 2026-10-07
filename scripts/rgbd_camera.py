"""고정 RGB-D 촬영과 도형별 윗면 기하 추정. 정답 pose·크기는 받지 않는다."""

from collections import deque
import math

import numpy as np
from PIL import Image
import pybullet as p

# 대각선 위에서 물체 상면과 옆면을 본다. 학습과 추론에서도 같은 설정을 사용한다.
CAMERA = {
    "width": 640,
    "height": 480,
    "eye": [0.8, -0.55, 1.05],
    "target": [0.5, 0, 0.30],
    "up": [0, 0, 1],
    "fov_deg": 50,
    "near_m": 0.05,
    "far_m": 2.0,
    "roi_xy_m": [[0.44, 0.56], [-0.065, 0.065]],
    "table_top_m": 0.30,
    "depth_difference_m": 0.005,
    "top_band_m": 0.002,
    "renderer": "ER_TINY_RENDERER",
    "crop_size": 64,
}


def camera_for_angle(angle=50):
    # 관찰 중심까지의 거리는 1m로 유지하고 위에서 내려다보는 각도만 바꾼다.
    camera = dict(CAMERA)
    elevation = math.radians(angle)
    direction = math.atan2(-0.55, 0.3)
    camera["eye"] = [
        0.5 + math.cos(elevation) * math.cos(direction),
        math.cos(elevation) * math.sin(direction),
        0.30 + math.sin(elevation),
    ]
    if angle == 90:
        camera["up"] = [0, 1, 0]
    return camera


# 단독 촬영·테스트·집기 실행 모두 같은 기본 카메라를 사용한다.
CAMERA = camera_for_angle()


def capture(camera):
    # GUI 화면 시점과 별개로, 항상 이 view/projection으로 촬영한다.
    view = p.computeViewMatrix(camera["eye"], camera["target"], camera["up"])
    projection = p.computeProjectionMatrixFOV(
        camera["fov_deg"],
        camera["width"] / camera["height"],
        camera["near_m"],
        camera["far_m"],
    )
    frame = p.getCameraImage(
        camera["width"],
        camera["height"],
        view,
        projection,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_NO_SEGMENTATION_MASK,
    )
    # RGB·깊이만 사용한다. 시뮬레이터의 물체 ID 마스크는 요청하지 않는다.
    return {
        "rgb": np.asarray(frame[2], dtype=np.uint8).reshape(
            camera["height"], camera["width"], 4
        )[..., :3],
        "depth": np.asarray(frame[3], dtype=np.float64).reshape(
            camera["height"], camera["width"]
        ),
        "view": view,
        "projection": projection,
    }


def world_points(observation):
    # OpenGL의 0~1 depth buffer와 픽셀 좌표를 -1~1 좌표로 바꾼다.
    depth = observation["depth"]
    height, width = depth.shape
    v, u = np.indices(depth.shape)
    # TinyRenderer는 정수 픽셀을 쓴다. 아래쪽 원점으로 행을 뒤집으면 height-1-v가 된다.
    ndc = np.stack(
        (
            2 * u / width - 1,
            1 - 2 * (v + 1) / height,
            2 * depth - 1,
            np.ones_like(depth),
        ),
        axis=-1,
    )
    # PyBullet 행렬은 열 우선이다. 투영·카메라 변환을 거꾸로 적용하면 월드 좌표가 된다.
    view = np.asarray(observation["view"]).reshape(4, 4, order="F")
    projection = np.asarray(observation["projection"]).reshape(4, 4, order="F")
    # 화면으로 보내는 변환을 역으로 풀어 각 픽셀의 3차원 위치를 되찾는다.
    homogeneous = ndc @ np.linalg.inv(projection @ view).T
    # 네 번째 성분으로 나누면 앞의 세 값이 월드의 X·Y·Z 좌표가 된다.
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
    # 정사각형은 긴 축이 없어 PCA 방향이 흔들린다. 점을 감싸는 최소 면적 사각형을 찾는다.
    # 0~90도를 0.1도씩 비교한다. 크기·방향은 생성 정답 없이 카메라 상면 점으로만 구한다.
    best = None
    for yaw in np.arange(900) * (np.pi / 1800):
        axes = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        local = xy @ axes
        low, high = local.min(axis=0), local.max(axis=0)
        spans = high - low
        area = float(np.prod(spans))
        if best is None or area < best[0]:
            best = (area, float(yaw), spans, ((low + high) / 2) @ axes.T)
    _, yaw, spans, center = best
    # 긴 변에 손끝 방향을 맞추고 짧은 변을 집는다. 정사각형의 90도 동등 방향은 0도 가까이 통일한다.
    if abs(spans[0] - spans[1]) <= 0.0005:
        if yaw > np.pi / 4:
            yaw -= np.pi / 2
    elif spans[1] > spans[0]:
        yaw -= np.pi / 2
    return {
        "valid": True,
        "center_xy_m": center.tolist(),
        "width_m": float(min(spans)),
        "yaw_rad": yaw,
    }


def cylinder_geometry(xy):
    low, high = xy.min(axis=0), xy.max(axis=0)
    diameters = high - low
    diameter = float(diameters.mean())
    if diameter <= 0 or abs(diameters[0] - diameters[1]) > 0.20 * diameter:
        return {"valid": False, "failure_reason": "non_circular_top"}
    # XY 양 끝만 쓰면 한 픽셀 차이로 폭 상한을 넘을 수 있다. 여러 외곽 점에 원을 맞춘다.
    ordered = sorted(set(map(tuple, xy)))

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower, upper = [], []
    # 내부 점을 제외하고 상면을 둘러싸는 테두리인 볼록 껍질만 남긴다.
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    boundary = np.asarray(lower[:-1] + upper[:-1])
    if len(boundary) < 12:
        return {"valid": False, "failure_reason": "non_circular_top"}
    origin = boundary.mean(axis=0)
    local = boundary - origin
    # (x-cx)²+(y-cy)²=r²를 정리한 선형식을 최소제곱으로 풀어 중심과 반지름을 구한다.
    solution = np.linalg.lstsq(
        np.column_stack((2 * local, np.ones(len(local)))),
        np.sum(local**2, axis=1),
        rcond=None,
    )[0]
    center = solution[:2]
    radius_squared = solution[2] + np.dot(center, center)
    if radius_squared <= 0:
        return {"valid": False, "failure_reason": "non_circular_top"}
    radius = np.sqrt(radius_squared)
    residual = float(np.max(np.abs(np.linalg.norm(local - center, axis=1) - radius)))
    # 원과 맞지 않는 경계는 보류한다. 가려진 부분을 무조건 완전한 원으로 복원하지 않는다.
    if residual > 0.0015:
        return {"valid": False, "failure_reason": "non_circular_top"}
    return {
        "valid": True,
        "center_xy_m": (center + origin).tolist(),
        "width_m": float(2 * radius),
        "diameter_xy_m": diameters.tolist(),
        "circle_boundary_error_m": residual,
        "yaw_rad": 0.0,
    }


def estimate(observation, empty, camera, object_shape="cuboid"):
    # 종류는 현재 명령에서 지정한다. 이후 CNN 예측을 이 입력에 연결할 수 있다.
    if object_shape not in ("cuboid", "cylinder"):
        return {"valid": False, "failure_reason": "unsupported_shape"}
    depth = observation["depth"]
    # 고장난 깊이나 다른 크기의 기준 영상으로 로봇을 움직이지 않는다.
    if (
        depth.shape != (camera["height"], camera["width"])
        or depth.shape != empty["depth"].shape
        or not np.isfinite(depth).all()
        or not np.isfinite(empty["depth"]).all()
        or np.any((depth < 0) | (depth > 1))
    ):
        return {"valid": False, "failure_reason": "invalid_depth"}
    points = world_points(observation)
    difference = depth_metres(empty["depth"], camera) - depth_metres(depth, camera)
    mask = difference > camera["depth_difference_m"]
    for axis, (low, high) in enumerate(camera["roi_xy_m"]):
        mask &= (points[..., axis] > low) & (points[..., axis] < high)
    mask &= (points[..., 2] > camera["table_top_m"] + 0.008) & (depth < 1)
    # 서로 이어진 물체 픽셀을 한 묶음으로 만든다. 각 묶음이 객체 후보가 된다.
    groups = components(mask)
    if len(groups) != 1:
        return {
            "valid": False,
            "failure_reason": "no_object" if not groups else "multiple_objects",
        }
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
    geometry = (
        cylinder_geometry(top[:, :2])
        if object_shape == "cylinder"
        else cuboid_geometry(top[:, :2])
    )
    if not geometry["valid"]:
        return geometry
    if (
        not 0.01 <= geometry["width_m"] <= 0.065
        or not 0.015 <= top_z - camera["table_top_m"] <= 0.08
    ):
        return {"valid": False, "failure_reason": "unsupported_geometry"}
    # 영역 경계에 닿으면 잘린 물체일 수 있으므로 완전한 중심을 구했다고 주장하지 않는다.
    for axis, (roi_low, roi_high) in enumerate(camera["roi_xy_m"]):
        if (
            surface[:, axis].min() < roi_low + 0.002
            or surface[:, axis].max() > roi_high - 0.002
        ):
            return {"valid": False, "failure_reason": "object_at_roi_edge"}
    return {
        **geometry,
        "object_shape": object_shape,
        "top_z_m": top_z,
        "height_m": top_z - camera["table_top_m"],
        "object_pixels": int(mask.sum()),
        "top_pixels": len(top),
        "mask": mask,
    }


def validate_size_range(size_range_m):
    values = np.asarray(size_range_m, dtype=float)
    if (
        values.shape != (4,)
        or not np.isfinite(values).all()
        or not 0.03 <= values[0] <= values[1] <= 0.064
        or not 0.03 <= values[2] <= values[3] <= 0.07
    ):
        raise ValueError(
            "폭은3~6.4cm, 높이는3~7cm 범위에서 최솟값이 최댓값 이하여야 합니다"
        )
    return tuple(float(n) for n in values)


def variable_surface_check(
    points, region, foreground, geometry, spans, camera, size_range_m
):
    minimum, maximum, min_height, max_height = size_range_m
    # 영상 경계 오차1.5mm만 허용한다. 생성된 개별 치수는 읽지 않는다.
    tolerance = 0.0015
    observed_size = (
        np.full(2, geometry['width_m'])
        if geometry['object_shape'] == 'cylinder'
        else spans
    )
    if (
        np.any(observed_size < minimum - tolerance)
        or np.any(observed_size > maximum + tolerance)
        or not min_height - tolerance <= geometry['height_m'] <= max_height + tolerance
    ):
        return 'unsupported_geometry'
    center = np.asarray(geometry['center_xy_m'])
    surface_xy = points[region][:, :2] - center
    yaw = geometry['yaw_rad']
    axes = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    if geometry['object_shape'] == 'cuboid':
        outside = np.any(np.abs(surface_xy @ axes) > spans / 2 + 0.003, axis=1)
    else:
        outside = np.linalg.norm(surface_xy, axis=1) > geometry['width_m'] / 2 + 0.003
    # 하나의 수직 물체라면 옆면도 윗면 테두리 안에 있다. 붙은 다른 물체는 보류한다.
    if np.any(outside):
        return 'incomplete_or_merged_surface'
    top_z = geometry['top_z_m']
    higher = points[foreground & (points[..., 2] > top_z + 0.003)]
    if len(higher):
        eye = np.asarray(camera['eye'])
        rays = higher - eye
        # 높은 물체를 지나는 카메라 광선을 후보 윗면 높이까지 연장한다.
        # 그 위치가 후보의 가능한 최대 범위와 겹치면 가림 가능성이 있어 뒤로 미룬다.
        scale = (top_z - eye[2]) / rays[:, 2]
        plane_xy = (eye + scale[:, None] * rays)[:, :2]
        possible_radius = np.sqrt(2) * maximum + 0.003
        if np.any(np.linalg.norm(plane_xy - center, axis=1) < possible_radius):
            return 'possible_top_occlusion'
    return ''


def estimate_many(
    observation,
    empty,
    camera,
    object_shape,
    supported_top_m=None,
    classifier=None,
    min_class_score=0.8,
    first_valid_only=False,
    size_range_m=None,
):
    if size_range_m is not None:
        size_range_m = validate_size_range(size_range_m)
        if supported_top_m is not None:
            raise ValueError(
                "고정한 윗면 크기와 랜덤 크기 범위는 함께 지정하지 않습니다"
            )
    if not 0.5 <= min_class_score <= 1:
        raise ValueError("분류 점수 기준은0.5~1이어야 합니다")
    if classifier is not None:
        classifier.validate_camera(camera)
        if supported_top_m is not None:
            raise ValueError("혼합 CNN 모드에서는 종류별 기본 크기 검사를 사용합니다")
    elif object_shape not in ("cuboid", "cylinder"):
        raise ValueError("수동 모드에는 도형 종류가 필요합니다")
    # 별도 지정 크기는 수동 실험에서만 사용한다. 개별 생성 정답을 읽지 않는다.
    if supported_top_m is not None:
        expected = np.asarray(supported_top_m, dtype=float)
        if (
            expected.shape != (2,)
            or not np.isfinite(expected).all()
            or np.any(expected <= 0)
        ):
            raise ValueError("지원 윗면 크기는 양수·유한수 두 개여야 합니다")
    # 생성 목록 없이 깊이 차이로 영역을 분리한다. 좌표 계산은 기존 함수를 재사용한다.
    depth = observation["depth"]
    if (
        depth.shape != (camera["height"], camera["width"])
        or depth.shape != empty["depth"].shape
        or any(
            not np.isfinite(d).all() or np.any((d < 0) | (d > 1))
            for d in (depth, empty["depth"])
        )
    ):
        return [{"valid": False, "failure_reason": "invalid_depth"}]
    points = world_points(observation)
    mask = (
        depth_metres(empty["depth"], camera) - depth_metres(depth, camera)
        > camera["depth_difference_m"]
    )
    for axis, (low, high) in enumerate(camera["roi_xy_m"]):
        mask &= (points[..., axis] > low) & (points[..., axis] < high)
    mask &= (points[..., 2] > camera["table_top_m"] + 0.008) & (depth < 1)
    # 서로 이어진 물체 픽셀을 한 묶음으로 만든다. 각 묶음이 객체 후보가 된다.
    groups = components(mask)
    ranked = []
    for group in groups:
        pixels = np.asarray(group)
        surface = points[pixels[:, 0], pixels[:, 1]]
        # 종류를 몰라도 깊이의 높은 점을 골라 윗면 후보 수를 셀 수 있다.
        top_z = float(np.percentile(surface[:, 2], 95))
        count = int(
            np.count_nonzero(np.abs(surface[:, 2] - top_z) <= camera["top_band_m"])
        )
        ranked.append((count, group))
    if first_valid_only:
        ranked.sort(key=lambda item: item[0], reverse=True)
    results = []
    selected = False
    for top_count, group in ranked:
        pixels = np.asarray(group)
        region = np.zeros_like(mask)
        region[pixels[:, 0], pixels[:, 1]] = True
        details = {
            "mask": region,
            "priority_top_pixels": top_count,
            "cnn_evaluated": False,
        }
        # 유효한 한 대상을 찾았으면 남은 영역은 분류하지 않고 탐지 기록만 남긴다.
        if first_valid_only and selected:
            results.append(
                {"valid": False, "failure_reason": "not_evaluated", **details}
            )
            continue
        isolated = dict(observation)
        isolated["depth"] = np.where(region, depth, empty["depth"])
        # 우선순위대로 필요한 후보만 RGB crop으로 잘라 분류한다.

        prediction = (
            classifier.predict(rgb_crop(observation["rgb"], region))
            if classifier is not None
            else {}
        )
        details["cnn_evaluated"] = classifier is not None
        shape = prediction.get("predicted_class", object_shape)
        if prediction and prediction["class_score"] < min_class_score:
            results.append(
                {
                    "valid": False,
                    "failure_reason": "low_class_confidence",
                    **prediction,
                    **details,
                }
            )
            continue
        geometry = estimate(isolated, empty, camera, shape)
        if geometry["valid"]:
            top = points[region]
            top = top[np.abs(top[:, 2] - geometry["top_z_m"]) <= camera["top_band_m"]]
            yaw = geometry["yaw_rad"]
            axes = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            spans = np.ptp(top[:, :2] @ axes, axis=0)
            # 기본 크기는 기존15% 비교를 유지하고, 랜덤 크기는 별도 영상 검사를 한다.
            if size_range_m is not None:
                reason = variable_surface_check(
                    points, region, mask, geometry, spans, camera, size_range_m
                )
                if reason:
                    geometry = {"valid": False, "failure_reason": reason}
            else:
                expected = np.sort(
                    np.asarray(
                        supported_top_m
                        if supported_top_m is not None
                        else ([0.06, 0.04] if shape == "cuboid" else [0.05, 0.05])
                    )
                )
                if np.any(np.abs(np.sort(spans) - expected) > expected * 0.15):
                    geometry = {
                        "valid": False,
                        "failure_reason": "incomplete_or_merged_surface",
                    }
        results.append({**geometry, **prediction, **details})
        # 점수가 낮거나 기하 검사가 실패하면 다음 후보를 계속 확인한다.
        selected = selected or geometry["valid"]
    return results


def rgb_crop(rgb, mask, size=64):
    # 같은 영역·여백·크기 변환을 이후 학습 데이터와 로컬 추론에서 재사용한다.
    v, u = np.where(mask)
    side = max(int(u.max() - u.min() + 1), int(v.max() - v.min() + 1)) + 12
    x = int((u.min() + u.max() - side) / 2)
    y = int((v.min() + v.max() - side) / 2)
    return (
        Image.fromarray(rgb)
        .crop((x, y, x + side, y + side))
        .resize((size, size), Image.Resampling.BILINEAR)
    )


def save_observation(output, observation, empty, result, camera):
    # 원본 RGB와 깊이를 보관해 추정 결과를 다시 계산할 수 있게 한다.
    Image.fromarray(observation["rgb"]).save(output / "rgb.png")
    # 보기용 깊이 그림은 가까울수록 밝다. 계산에는 원본 float depth를 사용한다.
    metres = depth_metres(observation["depth"], camera)
    grey = np.clip(
        255 * (camera["far_m"] - metres) / (camera["far_m"] - camera["near_m"]), 0, 255
    )
    Image.fromarray(grey.astype(np.uint8)).save(output / "depth.png")
    np.savez_compressed(
        output / "rgbd.npz",
        depth=observation["depth"],
        empty_depth=empty["depth"],
        view=observation["view"],
        projection=observation["projection"],
    )
    if result["valid"]:
        Image.fromarray(result["mask"].astype(np.uint8) * 255).save(output / "mask.png")
        rgb_crop(observation["rgb"], result["mask"], camera["crop_size"]).save(
            output / "crop.png"
        )

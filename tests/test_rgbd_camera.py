"""카메라 영상만으로 좌표를 찾는지 실제 렌더링으로 확인한다."""

import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class CameraTests(unittest.TestCase):
    def test_square_and_nearly_square_top_from_real_camera(self):
        import math
        from multi_object_scene import LAYOUT, build_scene, settle_scene
        from rgbd_camera import capture, estimate
        client = p.connect(p.DIRECT)
        try:
            # 정답 크기·각도는 장면 생성과 오차 평가에만 사용한다.
            for size in ([0.064, 0.064, 0.03], [0.064, 0.064, 0.07], [0.064, 0.062, 0.07]):
                for yaw in (-0.9, 0, 0.7):
                    with self.subTest(size=size, yaw=yaw):
                        scene = build_scene([{"shape": "cuboid", "xy": [0.5, 0], "yaw": yaw, "size_m": size}], LAYOUT)
                        self.assertTrue(settle_scene(scene)["settled"])
                        result = estimate(capture(scene["camera"]), scene["empty"], scene["camera"])
                        self.assertTrue(result["valid"], result)
                        self.assertLess(math.dist(result["center_xy_m"], [0.5, 0]), 0.002)
                        self.assertAlmostEqual(result["width_m"], min(size[:2]), delta=0.001)
                        # 정사각형은 90도 돌아도 같은 면을 잡는다. 거의 정사각형은 긴 축을 따른다.
                        period = math.pi / 2 if size[0] == size[1] else math.pi
                        error = abs((result["yaw_rad"] - yaw + period / 2) % period - period / 2)
                        self.assertLess(error, 0.02)
        finally:
            p.disconnect(client)

    def test_cylinder_top_center_diameter_and_height_from_rendered_depth(self):
        from rgbd_camera import CAMERA, capture, estimate

        self.assertIn("object_shape", inspect.signature(estimate).parameters,
                      "도형별 기하 추정 입력이 아직 없습니다")
        client = p.connect(p.DIRECT)
        try:
            # 실제 좌표·지름은 오차 평가에만 쓴다. 추정 함수에는 영상·종류만 전달한다.
            for x, y, diameter, height in ((0.5, 0, 0.03, 0.04), (0.48, -0.025, 0.025, 0.03)):
                p.resetSimulation()
                table = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.4, 0.02])
                p.createMultiBody(0, table, basePosition=[0.3, 0, 0.28])
                empty = capture(CAMERA)
                cylinder = p.createCollisionShape(p.GEOM_CYLINDER, radius=diameter / 2, height=height)
                p.createMultiBody(0, cylinder, basePosition=[x, y, 0.30 + height / 2])
                result = estimate(capture(CAMERA), empty, CAMERA, object_shape="cylinder")
                self.assertTrue(result["valid"], result)
                self.assertEqual(result["object_shape"], "cylinder")
                self.assertAlmostEqual(result["center_xy_m"][0], x, delta=0.002)
                self.assertAlmostEqual(result["center_xy_m"][1], y, delta=0.002)
                self.assertAlmostEqual(result["width_m"], diameter, delta=0.003)
                self.assertAlmostEqual(result["height_m"], height, delta=0.002)
                self.assertEqual(result["yaw_rad"], 0)
        finally:
            p.disconnect(client)

    def test_empty_frame_and_rotated_box_geometry(self):
        # 모듈 부재도 기능 미구현으로 명확하게 보고한다.
        self.assertIsNotNone(importlib.util.find_spec("rgbd_camera"), "카메라 모듈이 아직 없습니다")
        from rgbd_camera import CAMERA, capture, estimate, world_points

        client = p.connect(p.DIRECT)
        try:
            table = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.4, 0.02])
            p.createMultiBody(0, table, basePosition=[0.3, 0, 0.28])
            empty = capture(CAMERA)
            # 실제 높이가 알려진 평면으로 픽셀→월드 변환 자체의 오차를 먼저 검사한다.
            points = world_points(empty)
            selected = points[(abs(points[..., 0] - 0.5) < 0.04) & (abs(points[..., 1]) < 0.04)]
            self.assertGreater(len(selected), 100)
            self.assertLess(abs(float(selected[:, 2].mean()) - 0.30), 0.00002)
            self.assertFalse(estimate(empty, empty, CAMERA)["valid"])
            # 크기와 정답 좌표는 평가에만 사용한다. estimate에는 영상과 카메라 설정만 준다.
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.015, 0.02])
            block = p.createMultiBody(0, shape, basePosition=[0.51, 0.025, 0.32],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, 0.35]))
            result = estimate(capture(CAMERA), empty, CAMERA)
            self.assertTrue(result["valid"], result)
            self.assertAlmostEqual(result["center_xy_m"][0], 0.51, delta=0.003)
            self.assertAlmostEqual(result["center_xy_m"][1], 0.025, delta=0.003)
            self.assertAlmostEqual(result["top_z_m"], 0.34, delta=0.002)
            self.assertAlmostEqual(result["width_m"], 0.03, delta=0.004)
            self.assertAlmostEqual(result["yaw_rad"], 0.35, delta=0.12)
            # 물체가 두 개이거나 영역 경계에서 잘리면 로봇에 애매한 좌표를 전달하지 않는다.
            second = p.createMultiBody(0, shape, basePosition=[0.47, -0.025, 0.32])
            self.assertEqual(estimate(capture(CAMERA), empty, CAMERA)["failure_reason"], "multiple_objects")
            p.removeBody(block)
            p.removeBody(second)
            p.createMultiBody(0, shape, basePosition=[0.555, 0, 0.32])
            self.assertEqual(estimate(capture(CAMERA), empty, CAMERA)["failure_reason"], "object_at_roi_edge")
            broken = dict(empty)
            broken["depth"] = empty["depth"].copy()
            broken["depth"][0, 0] = float("nan")
            self.assertEqual(estimate(broken, empty, CAMERA)["failure_reason"], "invalid_depth")
        finally:
            p.disconnect(client)


if __name__ == "__main__":
    unittest.main()

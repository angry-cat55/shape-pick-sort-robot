"""새 장면의 개수 제한·랜덤 배치·실제 낙하를 확인한다."""

import importlib.util
import math
import tempfile
from pathlib import Path
import sys
import unittest

import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class MultiSceneTests(unittest.TestCase):
    def test_camera_marker_matches_view_and_leaves_rgb_unchanged(self):
        m = self.module()
        self.assertTrue(hasattr(m, "add_camera_visual"), "카메라 표시 기능이 아직 없습니다")
        from rgbd_camera import capture
        import numpy as np
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene([], m.LAYOUT)
            camera = scene["camera"]
            before = capture(camera)
            visual = m.add_camera_visual(camera)
            self.assertEqual(len(visual["footprint"]), 4)
            for corner in visual["footprint"]:
                self.assertAlmostEqual(corner[2], 0.30)
            np.testing.assert_allclose(p.getBasePositionAndOrientation(visual["bodies"][0])[0], camera["eye"])
            for body in visual["bodies"]:
                self.assertFalse(p.getCollisionShapeData(body, -1))
            after = capture(camera)
            np.testing.assert_array_equal(before["rgb"], after["rgb"])
            np.testing.assert_array_equal(before["depth"], after["depth"])
        finally:
            p.disconnect(client)

    def test_rear_slots_have_valid_static_ik_and_restore_waiting_pose(self):
        m = self.module()
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene([], m.LAYOUT)
            robot = scene["robot"]
            before = [p.getJointState(robot, i)[0] for i in range(p.getNumJoints(robot))]
            report = m.check_reachability(scene)
            self.assertTrue(report["all_valid"], report)
            self.assertEqual(len(report["checks"]), 20)
            self.assertFalse(report["transport_verified"])
            after = [p.getJointState(robot, i)[0] for i in range(p.getNumJoints(robot))]
            self.assertEqual(before, after)
        finally:
            p.disconnect(client)

    def test_one_camera_observes_regions_and_empty_scene(self):
        m = self.module()
        self.assertTrue(hasattr(m, "observe_scene"), "새 구도 촬영 기능이 아직 없습니다")
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene(m.sample_spawns(3, 2, 0, m.LAYOUT["region"]), m.LAYOUT)
            self.assertTrue(m.settle_scene(scene)["settled"])
            with tempfile.TemporaryDirectory() as directory:
                result = m.observe_scene(scene, Path(directory))
                self.assertEqual(result["observed_regions"], 5)
                self.assertEqual(len(list(Path(directory).glob("crop_*.png"))), 5)
                # 빈 촬영에서 생성 개수로 관측 개수를 대신하지 않는지 확인한다.
                for o in scene["objects"]:
                    p.removeBody(o["body"])
                empty = m.observe_scene(scene, Path(directory))
                self.assertEqual(empty["observed_regions"], 0)
                self.assertEqual(len(list(Path(directory).glob("crop_*.png"))), 0)
                self.assertEqual(len(list(Path(directory).glob("mask_*.png"))), 0)
            self.assertTrue(all(len(b["slots"]) == 5 for b in scene["bins"].values()))
        finally:
            p.disconnect(client)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("multi_object_scene"), "새 장면 프로그램이 아직 없습니다")
        import multi_object_scene
        return multi_object_scene

    def test_counts_seed_bounds_and_separation(self):
        m = self.module()
        for counts in ((0, 0), (4, 2), (-1, 2), (2, -1)):
            with self.assertRaises(ValueError):
                m.validate_counts(*counts)
        # 같은 seed의 생성값이 같아야 실패 장면도 재현할 수 있다.
        successful_samples = 0
        for seed in range(10):
            try:
                spawns = m.sample_spawns(3, 2, seed, m.LAYOUT["region"])
            except ValueError as error:
                # 랜덤 시도 횟수 안에 공간을 못 찾으면 재생성 대상으로 기록해야 한다.
                self.assertEqual(str(error), "placement_exhausted")
                with self.assertRaisesRegex(ValueError, "placement_exhausted"):
                    m.sample_spawns(3, 2, seed, m.LAYOUT["region"])
                continue
            successful_samples += 1
            self.assertEqual(spawns, m.sample_spawns(3, 2, seed, m.LAYOUT["region"]))
            self.assertEqual(len(spawns), 5)
            self.assertEqual(sum(s["shape"] == "cuboid" for s in spawns), 3)
            for s in spawns:
                for axis, key in enumerate(("x", "y")):
                    low, high = m.LAYOUT["region"][key]
                    self.assertGreaterEqual(s["xy"][axis] - s["half_xy"][axis], low)
                    self.assertLessEqual(s["xy"][axis] + s["half_xy"][axis], high)
            for i, a in enumerate(spawns):
                for b in spawns[i + 1:]:
                    self.assertTrue(any(abs(a["xy"][axis] - b["xy"][axis]) >=
                                        a["half_xy"][axis] + b["half_xy"][axis] + m.LAYOUT["gap_m"]
                                        for axis in (0, 1)))
        self.assertGreater(successful_samples, 0)
        # 빈 공간이 없으면 횟수 제한 후 이유를 알려야 한다.
        with self.assertRaisesRegex(ValueError, "placement_exhausted"):
            m.sample_spawns(3, 2, 0, {"x": [0.5, 0.51], "y": [0, 0.01]})

    def test_drop_settles_on_table_with_upright_cylinders(self):
        m = self.module()
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene(m.sample_spawns(3, 2, 0, m.LAYOUT["region"]), m.LAYOUT)
            self.assertTrue("spawn_area_visual" in scene, "생성 구역 바닥의 색상 표시가 아직 없습니다")
            self.assertFalse(p.getCollisionShapeData(scene["spawn_area_visual"], -1))
            self.assertNotEqual(p.getVisualShapeData(scene["spawn_area_visual"])[0][7][:3],
                                p.getVisualShapeData(scene["table"])[0][7][:3])
            initial = [p.getAABB(o["body"])[0][2] for o in scene["objects"]]
            self.assertTrue(all(z > 0.34 for z in initial))
            result = m.settle_scene(scene)
            self.assertTrue(result["settled"], result)
            for o in scene["objects"]:
                body = o["body"]
                self.assertAlmostEqual(p.getAABB(body)[0][2], 0.30, delta=0.002)
                self.assertTrue(p.getContactPoints(body, scene["table"]))
                if o["shape"] == "cylinder":
                    q = p.getBasePositionAndOrientation(body)[1]
                    self.assertGreater(p.getMatrixFromQuaternion(q)[8], math.cos(math.radians(5)))
            self.assertTrue(all(b["center"][0] < 0 for b in scene["bins"].values()))
            self.assertTrue(all(len(b["slots"]) >= 5 for b in scene["bins"].values()))
            self.assertFalse(m.settle_scene(scene, timeout_s=0)["settled"])
        finally:
            p.disconnect(client)


if __name__ == "__main__":
    unittest.main()

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
    def test_cylinder_surface_height_uses_tight_rotated_bounds(self):
        import io
        m = self.module()
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene(m.sample_spawns(0, 1, 0, m.LAYOUT["region"]), m.LAYOUT)
            body = scene["objects"][0]["body"]
            # 대조군 초기 배치: 상자 바닥에 누운 원기둥. 로봇 동작 중 reset은 아니다.
            p.resetBasePositionAndOrientation(body, [-0.28, -0.30, 0.335], p.getQuaternionFromEuler([math.pi / 2, 0, 0]))
            for _ in range(480):
                p.stepSimulation()
            probe = m.scene_probe(scene, io.StringIO(), False)
            probe.block = body
            self.assertAlmostEqual(probe.measure()["bottom_height_m"], 0.01, delta=0.001)
            self.assertLess(p.getAABB(body)[0][2], 0.305)
        finally:
            p.disconnect(client)

    def test_five_cylinders_and_manual_mixed_scene_rejection(self):
        m = self.module()
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene(m.sample_spawns(0, 5, 0, m.LAYOUT["region"]), m.LAYOUT)
            self.assertTrue(m.settle_scene(scene)["settled"])
            with tempfile.TemporaryDirectory() as directory:
                result = m.sort_scene(scene, "cylinder", Path(directory))
                self.assertTrue(result["success"], result)
                self.assertEqual(len(result["picks"]), 5)
                self.assertTrue(all(result["final_arrivals"].values()))
                mixed = m.build_scene(m.sample_spawns(1, 1, 0, m.LAYOUT["region"]), m.LAYOUT)
                with self.assertRaises(ValueError):
                    m.sort_scene(mixed, "cuboid", Path(directory))
        finally:
            p.disconnect(client)

    def test_hidden_object_is_not_completed_and_reappears_after_occluder_removal(self):
        from rgbd_camera import capture, estimate_many
        import numpy as np
        m = self.module()
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene(m.sample_spawns(1, 0, 0, m.LAYOUT["region"]), m.LAYOUT)
            self.assertTrue(m.settle_scene(scene)["settled"])
            center = np.array(p.getBasePositionAndOrientation(scene["objects"][0]["body"])[0])
            eye = np.array(scene["camera"]["eye"])
            # 진단용 고정판으로 시야를 가린다. 운반 중 순간이동하는 처리는 아니다.
            point = center + 0.22 * (eye - center)
            screen = m.box([0.075, 0.075, 0.015], point.tolist(), 0, [0.1, 0.1, 0.1, 1])
            hidden = estimate_many(capture(scene["camera"]), scene["empty"], scene["camera"], "cuboid")
            self.assertFalse(any(r["valid"] for r in hidden), hidden)
            with tempfile.TemporaryDirectory() as directory:
                result = m.sort_scene(scene, "cuboid", Path(directory))
                self.assertFalse(result["success"])
                self.assertEqual(result["failure_reason"], "no_valid_target")
            # 관측 대조 실험에서만 판을 제거하고, 재촬영으로 물체가 드러나는지 확인한다.
            p.removeBody(screen)
            visible = estimate_many(capture(scene["camera"]), scene["empty"], scene["camera"], "cuboid")
            self.assertEqual(sum(r["valid"] for r in visible), 1)
        finally:
            p.disconnect(client)

    def test_five_cuboids_reach_all_rear_slots(self):
        m = self.module()
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene(m.sample_spawns(5, 0, 0, m.LAYOUT["region"]), m.LAYOUT)
            self.assertTrue(m.settle_scene(scene)["settled"])
            with tempfile.TemporaryDirectory() as directory:
                result = m.sort_scene(scene, "cuboid", Path(directory))
                self.assertTrue(result["success"], result)
                self.assertEqual(len(result["picks"]), 5)
        finally:
            p.disconnect(client)

    def test_sequential_cuboids_arrive_with_a_scan_after_each_pick(self):
        m = self.module()
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene(m.sample_spawns(2, 0, 1, m.LAYOUT["region"]), m.LAYOUT)
            self.assertTrue(m.settle_scene(scene)["settled"])
            with tempfile.TemporaryDirectory() as directory:
                result = m.sort_scene(scene, "cuboid", Path(directory), False)
                self.assertTrue(result["success"], result)
                self.assertEqual(len(result["picks"]), 2)
                self.assertEqual(len(result["scans"]), 3)
                self.assertEqual(result["scans"][-1]["valid_targets"], 0)
        finally:
            p.disconnect(client)

    def test_multiple_geometries_use_camera_and_reject_cut_surface(self):
        from rgbd_camera import capture, estimate_many, world_points
        import numpy as np
        m = self.module()
        client = p.connect(p.DIRECT)
        try:
            scene = m.build_scene(m.sample_spawns(3, 0, 0, m.LAYOUT["region"]), m.LAYOUT)
            self.assertTrue(m.settle_scene(scene)["settled"])
            observation = capture(scene["camera"])
            results = estimate_many(observation, scene["empty"], scene["camera"], "cuboid")
            self.assertEqual(len(results), 3)
            self.assertTrue(all(r["valid"] for r in results), results)
            for r in results:
                error = min(math.dist(r["center_xy_m"], p.getBasePositionAndOrientation(o["body"])[0][:2])
                            for o in scene["objects"])
                self.assertLess(error, 0.002)
            # 깊이 영상에서 절반을 가린 대조군으로 잘린 윗면을 보류하는지 확인한다.
            masked = dict(observation)
            masked["depth"] = observation["depth"].copy()
            region = results[0]["mask"]
            yaw = results[0]["yaw_rad"]
            points = world_points(observation)
            short_axis = points[..., 0] * -math.sin(yaw) + points[..., 1] * math.cos(yaw)
            center = results[0]["center_xy_m"]
            midpoint = center[0] * -math.sin(yaw) + center[1] * math.cos(yaw)
            cut = region & (short_axis < midpoint)
            masked["depth"][cut] = scene["empty"]["depth"][cut]
            partial = estimate_many(masked, scene["empty"], scene["camera"], "cuboid")
            self.assertTrue(any(not r["valid"] for r in partial))
            self.assertEqual(estimate_many(scene["empty"], scene["empty"], scene["camera"], "cuboid"), [])
            broken_empty = dict(scene["empty"])
            broken_empty["depth"] = scene["empty"]["depth"].copy()
            broken_empty["depth"][0, 0] = np.nan
            invalid = estimate_many(scene["empty"], broken_empty, scene["camera"], "cuboid")
            self.assertTrue(any(r.get("failure_reason") == "invalid_depth" for r in invalid))
        finally:
            p.disconnect(client)

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

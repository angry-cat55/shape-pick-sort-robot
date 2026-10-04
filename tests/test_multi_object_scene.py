"""새 장면의 개수 제한·랜덤 배치·실제 낙하를 확인한다."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class MultiSceneTests(unittest.TestCase):
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

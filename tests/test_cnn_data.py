"""크기 범위·장면 분할·실제 사진 라벨을 검증한다."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import pybullet as p
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multi_object_scene as scene

class CnnDataTests(unittest.TestCase):
    def test_explicit_sizes_change_actual_geometry_without_changing_defaults(self):
        client = p.connect(p.DIRECT)
        try:
            spawns = [{"shape": "cuboid", "xy": [0.5, 0.1], "yaw": 0, "size_m": [0.03, 0.05, 0.04]},
                      {"shape": "cylinder", "xy": [0.5, -0.1], "yaw": 0, "size_m": [0.04, 0.03]}]
            world = scene.build_scene(spawns, scene.LAYOUT)
            self.assertTrue(scene.settle_scene(world)["settled"])
            low, high = p.getAABB(world["objects"][0]["body"])
            self.assertAlmostEqual(high[0] - low[0], 0.03, delta=0.001)
            self.assertAlmostEqual(high[2] - low[2], 0.04, delta=0.001)
            self.assertAlmostEqual(p.getBasePositionAndOrientation(world["objects"][1]["body"])[0][2], 0.315, delta=0.001)
            # 충돌 원기둥에 실제로 전달된 높이·반지름도 확인한다.
            dimensions = p.getCollisionShapeData(world["objects"][1]["body"], -1)[0][3]
            self.assertAlmostEqual(dimensions[0], 0.03)
            self.assertAlmostEqual(dimensions[1], 0.02)
        finally:
            p.disconnect(client)

    def test_touching_or_hidden_objects_are_not_given_single_object_labels(self):
        import collect_cnn_data as data
        client = p.connect(p.DIRECT)
        try:
            world = scene.build_scene([
                {"shape": "cuboid", "xy": [0.5, 0], "yaw": 0, "size_m": [0.06, 0.06, 0.06]},
                {"shape": "cylinder", "xy": [0.5, 0], "yaw": 0, "size_m": [0.03, 0.03]},
            ], scene.LAYOUT)
            # 같은 위치에 겹친 초기 장면은 수집 대상이 아니다. 잘못 라벨링하지 않는지 확인한다.
            with self.assertRaises(ValueError):
                data.labelled_crops(world)
        finally:
            p.disconnect(client)

    def test_ranges_and_rotated_bounds(self):
        import collect_cnn_data as data
        with self.assertRaises(ValueError):
            data.sample_scene(1, 0.07, 0.03)
        with self.assertRaises(ValueError):
            data.sample_scene(1, float("nan"), 0.06)
        for seed in range(20):
            objects = data.sample_scene(seed, 0.03, 0.06)
            self.assertTrue(1 <= len(objects) <= 5)
            for obj in objects:
                self.assertTrue(all(0.03 <= value <= 0.06 for value in obj["size_m"][:-1]) and 0.03 <= obj["size_m"][-1] <= 0.07)
                for axis, key in enumerate(("x", "y")):
                    self.assertGreaterEqual(obj["xy"][axis] - obj["half_xy"][axis], scene.LAYOUT["region"][key][0])
                    self.assertLessEqual(obj["xy"][axis] + obj["half_xy"][axis], scene.LAYOUT["region"][key][1])

    def test_dataset_has_disjoint_scenes_both_classes_and_shared_rgb_crop(self):
        import collect_cnn_data as data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            result = data.collect(root, scenes=6, seed=17, minimum=0.03, maximum=0.06, show_progress=False)
            self.assertEqual(result["status"], "complete")
            rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
            groups = {split: {row["scene_id"] for row in rows if row["split"] == split} for split in ("train", "val", "test")}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                self.assertFalse(groups[a] & groups[b])
            for split in groups:
                self.assertEqual({r["label"] for r in rows if r["split"] == split}, {"cuboid", "cylinder"})
            for row in rows:
                with Image.open(root / row["path"]) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (64, 64))
                self.assertGreaterEqual(row["annotation_purity"], 0.99)
            self.assertTrue(root.with_suffix(".zip").is_file())
            with self.assertRaises(FileExistsError):
                data.collect(root, scenes=6, seed=17, show_progress=False)

if __name__ == "__main__":
    unittest.main()

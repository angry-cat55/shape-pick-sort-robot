"""실제 접촉 파지와 잘못된 성공 판정을 대조하는 통합 실험."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "contact_grasp_probe.py"


class ContactGraspTests(unittest.TestCase):
    def test_larger_defaults_fit_inside_bin_after_camera_transport(self):
        # 새 기본 크기가 실제 영상에 반영되고, 운반 후 물체 전체가 상자 안에 정착해야 한다.
        for shape, width in (("cuboid", 0.04), ("cylinder", 0.05)):
            with self.subTest(shape=shape):
                result = self.scenario("transport", "--pose-source", "camera", "--object-shape", shape)
                self.assertAlmostEqual(result["perception"]["height_m"], 0.06, delta=0.002)
                self.assertAlmostEqual(result["perception"]["width_m"], width, delta=0.003)
                self.assertTrue(result["arrival_success"])
                self.assertFalse(result["drop_detected"])

    def test_camera_cylinder_transport_uses_estimated_geometry(self):
        result = self.scenario("transport", "--pose-source", "camera", "--object-shape", "cylinder")
        self.assertEqual(result["object_shape"], "cylinder")
        self.assertEqual(result["perception"]["object_shape"], "cylinder")
        self.assertLess(result["perception_error_xy_m"], 0.002)
        self.assertTrue(result["lift_success"])
        self.assertTrue(result["arrival_success"])
        self.assertFalse(result["drop_detected"])
        self.assertAlmostEqual(result["grasp_plan"]["position_m"][2],
                               result["perception"]["top_z_m"] - 0.015)

    def test_camera_cylinder_different_size_and_position(self):
        result = self.scenario("transport", "--pose-source", "camera", "--object-shape", "cylinder",
                               "--block-x", "0.52", "--block-y", "0.025",
                               "--cylinder-diameter", "0.04", "--cylinder-height", "0.05")
        self.assertTrue(result["lift_success"])
        self.assertTrue(result["arrival_success"])
        self.assertAlmostEqual(result["perception"]["width_m"], 0.04, delta=0.003)

    def test_short_cylinder_releases_without_palm_hitting_bin_wall(self):
        # 낮은 물체를 바닥까지 내리다가 손바닥이 상자 벽에 걸린 사례를 재현한다.
        result = self.scenario("transport", "--pose-source", "camera", "--object-shape", "cylinder",
                               "--block-x", "0.48", "--block-y", "-0.025",
                               "--cylinder-diameter", "0.025", "--cylinder-height", "0.03")
        self.assertTrue(result["arrival_success"])
        self.assertFalse(result["drop_detected"])
        self.assertAlmostEqual(result["grasp_plan"]["position_m"][2],
                               result["perception"]["top_z_m"] - 0.015)

    def scenario(self, name, *extra):
        # 실제 물리 실험을 실행하고 요약을 읽는다. 테스트 로그는 임시 폴더에 둔다.
        self.assertTrue(SCRIPT.is_file(), "접촉 파지 실행 프로그램이 아직 없습니다")
        with tempfile.TemporaryDirectory() as output:
            process = subprocess.run(
                [sys.executable, str(SCRIPT), "--scenario", name, "--output-dir", output, *extra],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            summary = json.loads((Path(output) / "summary.json").read_text())
        # 실패 실험도 예상한 실패를 감지하면 테스트 통과다.
        self.assertTrue(summary["all_expectations_met"], summary)
        return summary["trials"][0]

    def test_camera_coordinates_drive_offset_rotated_grasp(self):
        result = self.scenario("transport", "--pose-source", "camera", "--block-x", "0.51",
                               "--block-y", "0.025", "--block-yaw", "0.35")
        self.assertEqual(result["pose_source"], "camera")
        self.assertLess(result["perception_error_xy_m"], 0.003)
        self.assertTrue(result["lift_success"])
        self.assertTrue(result["arrival_success"])

    def test_camera_empty_frame_stops_before_grasp(self):
        result = self.scenario("camera_empty", "--pose-source", "camera")
        self.assertEqual(result["failure_reason"], "no_object")
        self.assertFalse(result["lift_success"])
        self.assertLessEqual(result["sim_duration_s"], 1.01)

    def test_normal_grasp_holds_one_second_and_places(self):
        result = self.scenario("normal")
        self.assertTrue(result["lift_success"])
        self.assertGreaterEqual(result["hold_duration_s"], 1.0)
        self.assertGreaterEqual(result["hold_min_bottom_height_m"], 0.05)
        self.assertTrue(result["place_success"])
        self.assertFalse(result["drop_detected"])

    def test_missed_grasp_has_no_false_positive_lift(self):
        result = self.scenario("miss")
        self.assertFalse(result["lift_success"])
        self.assertFalse(result["place_success"])

    def test_opening_after_lift_detects_actual_drop(self):
        result = self.scenario("open")
        self.assertTrue(result["lift_success"])
        self.assertTrue(result["drop_detected"])
        self.assertTrue(result.get("drop_via_relative_motion", False))
        self.assertFalse(result["place_success"])

    def test_elevated_object_on_support_is_not_a_grasp(self):
        result = self.scenario("support")
        self.assertGreaterEqual(result["hold_min_bottom_height_m"], 0.05)
        self.assertFalse(result["lift_success"])
        self.assertGreater(result["support_contact_steps"], 0)

    def test_transport_reaches_inside_designated_bin(self):
        result = self.scenario("transport")
        self.assertTrue(result["lift_success"])
        self.assertFalse(result["drop_detected"])
        self.assertTrue(result["bin_arrivals"]["blue"])
        self.assertTrue(result["arrival_success"])

    def test_drop_during_transport_is_not_arrival(self):
        result = self.scenario("transport_drop")
        self.assertTrue(result["lift_success"])
        self.assertTrue(result["drop_detected"])
        self.assertFalse(result["arrival_success"])

    def test_other_bin_is_not_designated_arrival(self):
        result = self.scenario("wrong_bin")
        self.assertTrue(result["lift_success"])
        self.assertTrue(result["bin_arrivals"]["green"])
        self.assertFalse(result["arrival_success"])
        self.assertEqual(result["failure_reason"], "wrong_bin_arrival")


if __name__ == "__main__":
    unittest.main()

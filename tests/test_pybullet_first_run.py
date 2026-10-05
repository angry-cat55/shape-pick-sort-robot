"""첫 장면의 실제 낙하와 실패 보고를 확인하는 통합 테스트."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pybullet_first_run.py"


class FirstRunTests(unittest.TestCase):
    def run_scene(self, *args, env=None):
        # 현재 Python 환경에서 첫 장면을 실제로 실행하고 출력과 종료 코드를 받는다.
        self.assertTrue(SCRIPT.is_file(), "첫 장면 실행 프로그램이 아직 없습니다")
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

    def result(self, process):
        # 여러 출력 중 실행 결과 줄에서 판정 결과만 꺼낸다.
        lines = process.stdout.splitlines()
        payload = next(line for line in lines if line.startswith("실행 결과="))
        return json.loads(payload.removeprefix("실행 결과="))

    def test_gravity_and_ground_contact_after_two_simulation_seconds(self):
        process = self.run_scene("--mode", "direct", "--steps", "480")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = self.result(process)
        self.assertEqual(result["완료한 물리 계산 수"], 480)
        self.assertAlmostEqual(result["시뮬레이션 시간(초)"], 2.0)
        self.assertAlmostEqual(result["물체 초기 높이(m)"], 0.2)
        self.assertLess(result["물체 최종 높이(m)"], result["물체 초기 높이(m)"] - 0.1)
        self.assertAlmostEqual(result["물체 최종 높이(m)"], 0.02, delta=0.005)
        self.assertGreater(result["바닥 접촉 수"], 0)
        self.assertEqual(result["바닥 정착 여부"], "성공")

    def test_short_run_does_not_report_settled(self):
        process = self.run_scene("--steps", "1")
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        self.assertEqual(self.result(process)["바닥 정착 여부"], "실패")

    def test_nonpositive_steps_are_rejected(self):
        process = self.run_scene("--steps", "0")
        self.assertEqual(process.returncode, 2)
        self.assertIn("양의 정수", process.stderr)

    def test_gui_without_display_reports_how_to_run_direct(self):
        env = dict(os.environ)
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        process = self.run_scene("--mode", "gui", env=env)
        self.assertEqual(process.returncode, 2)
        self.assertIn("--mode direct", process.stderr)


if __name__ == "__main__":
    unittest.main()

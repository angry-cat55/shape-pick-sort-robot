"""단계 표시를 바꾸어도 저장 식별자와 실패 판정은 유지되는지 확인한다."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from terminal_ko import stage_name, failure_name


class TerminalKoreanTests(unittest.TestCase):
    def test_every_motion_stage_has_korean_display_and_keeps_internal_name(self):
        from contact_grasp_probe import Probe
        # 실제 코드에 쓰인 기본 단계와 이동 중 자동으로 붙는 접미사를 모두 확인한다.
        stages = {'SETUP'}
        for name in ['contact_grasp_probe.py', 'multi_object_scene.py']:
            for node in ast.walk(ast.parse((ROOT/'scripts'/name).read_text())):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ('start_stage', 'wait', 'move') and node.args
                        and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                    stages.add(node.args[0].value)
        probe = Probe.__new__(Probe)
        for stage in stages:
            for label in [stage, stage+'_FINAL', stage+'_FINAL_SETTLE']:
                display = io.StringIO()
                with redirect_stdout(display): probe.start_stage(label)
                self.assertEqual(probe.stage, label)
                self.assertNotIn(label, display.getvalue())
                self.assertNotEqual(stage_name(label), '동작 단계 확인 필요')
        self.assertEqual(stage_name('WAIT_SETTLE'), '대기 자세에서 안정화 대기')
        self.assertEqual(failure_name('drop_during_transport'), '운반 중 물체가 떨어짐')

    def test_all_cli_help_and_argument_errors_are_korean(self):
        for name in ['pybullet_first_run.py', 'contact_grasp_probe.py', 'multi_object_scene.py', 'collect_cnn_data.py']:
            script = ROOT/'scripts'/name
            help_result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn('사용법:', help_result.stdout)
            self.assertIn('실행 옵션', help_result.stdout)
            error = subprocess.run([sys.executable, str(script), '--unknown-option'], capture_output=True, text=True)
            self.assertEqual(error.returncode, 2)
            self.assertIn('알 수 없는 옵션:', error.stderr)
            self.assertNotIn('unrecognized arguments', error.stderr)

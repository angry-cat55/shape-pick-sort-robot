"""화면 설정·실행 인자·실제 프로세스 정리를 확인한다."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shape_sort_gui as gui


class GuiSettingsTests(unittest.TestCase):
    def test_count_limits_and_empty_scene(self):
        self.assertEqual(gui.parse_counts('3', '2'), (3, 2))
        self.assertEqual(gui.parse_counts('0', '5'), (0, 5))
        for cuboids, cylinders in [('0','0'), ('3','3'), ('-1','2'), ('2.5','1'), ('x','1')]:
            with self.subTest(cuboids=cuboids,cylinders=cylinders), self.assertRaises(ValueError):
                gui.parse_counts(cuboids, cylinders)

    def test_windows_fit_screen_budget(self):
        for width, height in [(2048,1280), (1366,768), (1024,768), (3840,2160)]:
            layout = gui.window_layout(width, height)
            cw, ch, x, y = layout['control']
            sw, sh = layout['simulation']
            self.assertLessEqual(x + cw, width)
            self.assertLessEqual(y + ch, height)
            self.assertLessEqual(sw, width)
            self.assertLessEqual(sh, height)
            if layout['side_by_side']:
                self.assertLessEqual(cw + sw + 24, width)
            else:
                self.assertLessEqual(ch + sh + 80, height)

    def test_launch_values_stay_arguments(self):
        root = Path('/tmp/project with spaces')
        command = gui.launch_command(root, 2, 1, 17, root/'outputs/a', (900,700))
        self.assertIn('cuboids:=2', command)
        self.assertIn('cylinders:=1', command)
        self.assertIn('seed:=17', command)
        self.assertIn('gui_width:=900', command)
        self.assertIn('gui_height:=700', command)
        self.assertIn('output_dir:=/tmp/project with spaces/outputs/a', command)
        # 경로와 값은 셸 본문에 넣지 않고 인용된 위치 인자로 전달한다.
        self.assertNotIn('/tmp/project with spaces', command[2])


class ProcessTests(unittest.TestCase):
    def wait_until(self, condition, seconds=8):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(condition(), '프로세스 정리 시간 초과')

    def test_collects_output_and_natural_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            run = gui.SimulationProcess(
                [sys.executable, '-u', '-c', "print('첫 로그'); print('둘째 로그')"],
                Path(folder), Path(folder),
            )
            run.start()
            self.wait_until(run.finished.is_set)
            content = (Path(folder)/'terminal.log').read_text()
            self.assertIn('첫 로그', content)
            self.assertIn('둘째 로그', content)
            self.assertEqual(run.process.returncode, 0)

    def test_stops_owned_children_but_keeps_unrelated_process(self):
        foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        with tempfile.TemporaryDirectory() as folder:
            code = "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(child.pid,flush=True); time.sleep(30)"
            run = gui.SimulationProcess([sys.executable,'-u','-c',code], Path(folder), Path(folder))
            try:
                run.start()
                self.wait_until(lambda:(Path(folder)/'terminal.log').exists() and (Path(folder)/'terminal.log').stat().st_size>0)
                child_pid = int((Path(folder)/'terminal.log').read_text().splitlines()[0])
                run.stop()
                self.wait_until(run.finished.is_set)
                self.assertIsNone(foreign.poll())
                stat = Path(f'/proc/{child_pid}/stat')
                if stat.exists():
                    self.assertEqual(stat.read_text().split()[2], 'Z')
            finally:
                run.stop()
                foreign.terminate()
                foreign.wait(timeout=5)

    def test_noncooperative_process_is_forcefully_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            code="import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); print('준비',flush=True); time.sleep(30)"
            run=gui.SimulationProcess([sys.executable,'-u','-c',code], Path(folder), Path(folder))
            run.start()
            self.wait_until(lambda:(Path(folder)/'terminal.log').exists() and (Path(folder)/'terminal.log').stat().st_size>0)
            run.stop()
            self.wait_until(run.finished.is_set)
            self.assertIsNotNone(run.process.poll())

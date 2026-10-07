"""도형 설정과 ROS2 시뮬레이션 실행·로그 확인을 위한 제어 화면."""
import json
import os
from pathlib import Path
import queue
import secrets
import signal
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_counts(cuboids, cylinders):
    # 입력칸에는 문자가 들어올 수도 있다. 정수 변환과 전체 개수를 함께 확인한다.
    try:
        counts = int(cuboids), int(cylinders)
    except (ValueError, TypeError) as error:
        raise ValueError("도형 개수는 0~5 사이의 정수로 입력하세요") from error
    if any(value < 0 or value > 5 for value in counts) or not 1 <= sum(counts) <= 5:
        raise ValueError("두 종류의 합계가 1~5개가 되도록 선택하세요")
    return counts


def window_layout(screen_width, screen_height):
    # 창 테두리·상단 바·독을 위한 여백을 남긴다. 제어 화면은 로그 공간도 확보한다.
    width = max(320, screen_width - 64)
    height = max(240, screen_height - 140)
    side_by_side = screen_width >= 1360
    if side_by_side:
        control_width = min(860, max(580, int(width * 0.43)))
        simulation_width = min(1200, width - control_width - 24)
        control_height = min(820, height)
        simulation_height = min(820, height, round(simulation_width * 0.72))
    else:
        # 작은 화면에서는 두 창을 위아래로 볼 수 있는 크기로 줄인다.
        control_width = min(900, width)
        simulation_width = min(1100, width)
        control_height = min(440, int(height * 0.48))
        simulation_height = min(720, height - control_height)
    return {
        "control": (control_width, control_height, screen_width - control_width - 24, 40),
        "simulation": (simulation_width, simulation_height),
        "side_by_side": side_by_side,
    }


def launch_command(root, cuboids, cylinders, seed, output, simulation_size):
    # 셸 본문에는 설정 경로·입력값을 끼워 넣지 않는다. 모두 별도 인자로 전달한다.
    width, height = simulation_size
    shell = 'set -e; source "$1"; source "$2"; shift 2; exec ros2 launch shape_sort_ros shape_sort.launch.py "$@"'
    return [
        "bash", "-c", shell, "shape-sort-gui",
        "/opt/ros/jazzy/setup.bash", str(root / "install/setup.bash"),
        "auto_start:=true", "mode:=gui", "size_mode:=random",
        f"cuboids:={cuboids}", f"cylinders:={cylinders}", f"seed:={seed}",
        f"gui_width:={width}", f"gui_height:={height}",
        f"output_dir:={output}",
        f"model_dir:={root / 'checkpoints/shape_cnn_v1'}",
        f"python_executable:={root / '.venv/bin/python'}",
    ]


class SimulationProcess:
    """한 번 실행한 프로세스와 로그를 관리한다. Tk 위젯은 여기서 만지지 않는다."""
    def __init__(self, command, root, output, environment=None):
        self.command = command
        self.root = root
        self.output = output
        self.environment = environment
        self.process = None
        self.reader = None
        self.events = queue.Queue()
        self.finished = threading.Event()
        self.stopping = threading.Event()
        self.stop_lock = threading.Lock()

    def start(self):
        # 새 세션을 만들면 종료할 때 이 실행의 프로세스 그룹만 정리할 수 있다.
        if self.process is not None:
            raise RuntimeError("이 실행은 이미 시작했습니다")
        self.output.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            self.command, cwd=self.root, env=self.environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, start_new_session=True,
        )
        self.reader = threading.Thread(target=self.read_output, name="simulation-log", daemon=True)
        self.reader.start()

    def read_output(self):
        # stdout 읽기는 기다릴 수 있으므로 별도 스레드에서 한다. GUI에는 큐로 전달한다.
        try:
            with (self.output / "terminal.log").open("w", encoding="utf-8") as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    self.events.put(("log", line))
        except OSError as error:
            self.events.put(("error", "실행 로그를 읽거나 저장하지 못했습니다: " + str(error)))
            self.stop()
        finally:
            self.process.stdout.close()
            code = self.process.wait()
            self.events.put(("exit", code))
            # 부모가 먼저 끝나도 남은 자식이 있으면 정리한다.
            self.stop()

    def stop(self):
        # 버튼 연타와 창 닫기가 겹쳐도 종료 스레드는 한 번만 만든다.
        with self.stop_lock:
            if self.stopping.is_set():
                return
            self.stopping.set()
            threading.Thread(target=self.stop_owned_group, name="simulation-stop", daemon=True).start()

    def signal_group(self, signal_number):
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal_number)
        except ProcessLookupError:
            pass

    def group_exists(self):
        if self.process is None:
            return False
        try:
            os.killpg(self.process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def stop_owned_group(self):
        try:
            if self.process is None:
                return
            # launch에 먼저 Ctrl+C와 같은 신호를 준다. launch가 노드들을 정상 종료시킨다.
            if self.process.poll() is None:
                try:
                    self.process.send_signal(signal.SIGINT)
                    self.process.wait(timeout=3.0)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    pass

            # 종료에 응답하지 않거나 남은 자식이 있는 경우에만 그룹 전체를 정리한다.
            if self.group_exists():
                self.signal_group(signal.SIGTERM)
                deadline = time.monotonic() + 1.0
                while self.group_exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
            if self.group_exists():
                self.signal_group(signal.SIGKILL)
            self.process.wait(timeout=2.0)
        finally:
            if self.reader is not None:
                self.reader.join(timeout=1.0)
            self.finished.set()
            self.events.put(("closed", None))


def main():
    # GUI가 필요한 경우에만 tkinter를 불러온다. 계산·프로세스 테스트는 창 없이 실행한다.
    import tkinter as tk
    root = tk.Tk()
    ShapeSortApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

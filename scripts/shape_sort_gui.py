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
        control_height = min(440, max(380, int(height * 0.52)))
        simulation_height = min(720, height - control_height)
    return {
        "control": (
            control_width,
            control_height,
            screen_width - control_width - 24,
            40,
        ),
        "simulation": (simulation_width, simulation_height),
        "side_by_side": side_by_side,
    }


def launch_command(root, cuboids, cylinders, seed, output, simulation_size):
    # 셸 본문에는 설정 경로·입력값을 끼워 넣지 않는다. 모두 별도 인자로 전달한다.
    width, height = simulation_size
    shell = 'set -e; source "$1"; source "$2"; shift 2; exec ros2 launch shape_sort_ros shape_sort.launch.py "$@"'
    return [
        "bash",
        "-c",
        shell,
        "shape-sort-gui",
        "/opt/ros/jazzy/setup.bash",
        str(root / "install/setup.bash"),
        "auto_start:=true",
        "mode:=gui",
        "size_mode:=random",
        f"cuboids:={cuboids}",
        f"cylinders:={cylinders}",
        f"seed:={seed}",
        f"gui_width:={width}",
        f"gui_height:={height}",
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
            self.command,
            cwd=self.root,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self.reader = threading.Thread(
            target=self.read_output, name="simulation-log", daemon=True
        )
        self.reader.start()
        # 남은 자식이 로그를 열고 있어도 부모 종료를 별도로 감지해 정리를 시작한다.
        threading.Thread(
            target=self.watch_parent, name="simulation-exit", daemon=True
        ).start()

    def watch_parent(self):
        self.process.wait()
        self.stop()

    def read_output(self):
        # stdout 읽기는 기다릴 수 있으므로 별도 스레드에서 한다. GUI에는 큐로 전달한다.
        try:
            with (self.output / "terminal.log").open("w", encoding="utf-8") as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    self.events.put(("log", line))
        except OSError as error:
            self.events.put(
                ("error", "실행 로그를 읽거나 저장하지 못했습니다: " + str(error))
            )
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
            threading.Thread(
                target=self.stop_owned_group, name="simulation-stop", daemon=True
            ).start()

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


class ShapeSortApp:
    """선택 화면을 그리고 큐로 받은 로그·실행 결과를 표시한다."""

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.run = None
        self.result = None
        self.closing = False
        self.user_stopped = False
        self.closed_handled = False
        self.layout = window_layout(root.winfo_screenwidth(), root.winfo_screenheight())
        width, height, x, y = self.layout["control"]
        root.title("도형 분류 로봇 · 실행 제어")
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.minsize(min(620, width), min(380, height))
        root.protocol("WM_DELETE_WINDOW", self.close_app)

        # 화면 값은 Tk 변수가 관리한다. 작업 스레드는 이 변수나 위젯을 수정하지 않는다.
        self.cuboids = tk.StringVar(value="3")
        self.cylinders = tk.StringVar(value="2")
        self.count_text = tk.StringVar(value="총 5개 / 최대 5개")
        self.status = tk.StringVar(value="대기 중")
        self.build_widgets()
        self.cuboids.trace_add("write", self.on_count_changed)
        self.cylinders.trace_add("write", self.on_count_changed)
        self.root.after(80, self.poll_run)

    def build_widgets(self):
        ttk = self.ttk
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f9")
        style.configure("TLabel", background="#f3f5f9", foreground="#243448")
        style.configure("Title.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure(
            "Small.TLabel", font=("TkDefaultFont", 10), foreground="#64748b"
        )
        style.configure("Status.TLabel", font=("TkDefaultFont", 11, "bold"))
        style.configure(
            "Start.TButton", background="#0f766e", foreground="white", padding=9
        )
        style.map(
            "Start.TButton",
            background=[("active", "#115e59"), ("disabled", "#cbd5e1")],
            foreground=[("disabled", "#64748b")],
        )
        self.root.configure(background="#f3f5f9")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # 위쪽에는 프로그램 목적만 짧게 보여준다. 세부 동작은 오른쪽 로그에서 확인한다.
        header = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="도형 분류 로봇", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="도형을 고르고 시작하면 별도 시뮬레이션 창이 열립니다.",
            style="Small.TLabel",
        ).pack(anchor="w", pady=(5, 0))

        body = ttk.Frame(self.root, padding=(18, 0, 18, 0))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        settings = ttk.Frame(body, padding=(0, 12, 16, 0))
        settings.grid(row=0, column=0, sticky="ns")
        ttk.Label(settings, text="도형 선택", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        self.count_inputs = []
        for row, (label, variable) in enumerate(
            (("직육면체", self.cuboids), ("원기둥", self.cylinders)), start=1
        ):
            ttk.Label(settings, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=5
            )
            field = ttk.Spinbox(
                settings,
                from_=0,
                to=5,
                width=4,
                textvariable=variable,
                justify="center",
            )
            field.grid(row=row, column=1, sticky="e", pady=5)
            self.count_inputs.append(field)
        ttk.Label(
            settings, textvariable=self.count_text, style="Small.TLabel", wraplength=165
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 14))

        # 실행 중에는 설정과 시작을 잠그고, 종료 버튼만 사용할 수 있도록 한다.
        self.start_button = ttk.Button(
            settings, text="시작", command=self.start_run, style="Start.TButton"
        )
        self.start_button.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.stop_button = ttk.Button(
            settings, text="시뮬레이션 종료", command=self.stop_run, state="disabled"
        )
        self.stop_button.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 14))
        ttk.Separator(settings).grid(row=6, column=0, columnspan=2, sticky="ew")
        ttk.Label(settings, text="현재 상태", style="Small.TLabel").grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(12, 4)
        )
        ttk.Label(
            settings, textvariable=self.status, style="Status.TLabel", wraplength=165
        ).grid(row=8, column=0, columnspan=2, sticky="w")

        # 로그는 선택·복사는 가능하지만 입력으로 수정할 수 없는 텍스트 영역이다.
        logs = ttk.Frame(body)
        logs.grid(row=0, column=1, sticky="nsew")
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(1, weight=1)
        ttk.Label(logs, text="실행 로그", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(12, 8)
        )
        self.log_text = self.tk.Text(
            logs,
            state="disabled",
            wrap="word",
            width=30,
            height=8,
            background="#172334",
            foreground="#dbe7f3",
            insertbackground="white",
            font=("TkFixedFont", 10),
            padx=12,
            pady=10,
            borderwidth=0,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.tag_configure("error", foreground="#fda4af")
        self.log_text.tag_configure("note", foreground="#67e8f9")
        ttk.Label(
            self.root,
            text="완료된 장면은 유지됩니다. 종료 후 새로 시작할 수 있습니다.",
            style="Small.TLabel",
            padding=(18, 8),
        ).grid(row=2, column=0, sticky="w")

    def append_log(self, text, tag=None):
        # 사용자가 지난 로그를 읽고 있으면 스크롤 위치를 강제로 맨 아래로 보내지 않는다.
        follow = self.log_text.yview()[1] >= 0.98
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text, tag or ())
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 2000:
            self.log_text.delete("1.0", f"{lines - 2000}.0")
        self.log_text.configure(state="disabled")
        if follow:
            self.log_text.see("end")

    def on_count_changed(self, *args):
        try:
            counts = parse_counts(self.cuboids.get(), self.cylinders.get())
            self.count_text.set(f"총 {sum(counts)}개 / 최대 5개")
            valid = True
        except ValueError as error:
            self.count_text.set(str(error))
            valid = False
        running = self.run is not None and not self.run.finished.is_set()
        self.start_button.configure(
            state="normal" if valid and not running and not self.closing else "disabled"
        )

    def check_environment(self):
        # 모델과 설치된 ROS2 실행 파일이 있어야 시작한다. GUI가 몰래 재학습·빌드하지 않는다.
        required = [
            Path("/opt/ros/jazzy/setup.bash"),
            ROOT / "install/setup.bash",
            ROOT / ".venv/bin/python",
            ROOT / "checkpoints/shape_cnn_v1/best_model.pt",
            ROOT / "checkpoints/shape_cnn_v1/best_params.json",
        ]
        for path in required:
            if not path.is_file():
                raise ValueError(
                    "필요한 파일이 없습니다: "
                    + str(path)
                    + "\nREADME의 설치·빌드 안내를 확인하세요."
                )
        installed_launch = (
            ROOT
            / "install/shape_sort_ros/share/shape_sort_ros/launch/shape_sort.launch.py"
        )
        if (
            not installed_launch.is_file()
            or "gui_width" not in installed_launch.read_text()
        ):
            raise ValueError(
                "ROS2 실행 파일을 다시 빌드해야 합니다. README의 빌드 명령을 실행하세요."
            )

    def start_run(self):
        from datetime import datetime

        # 버튼이 비활성화되어도 같은 이벤트가 들어올 수 있어 실행 중 여부를 다시 확인한다.
        if self.closing or (self.run is not None and not self.run.finished.is_set()):
            return
        try:
            cuboids, cylinders = parse_counts(self.cuboids.get(), self.cylinders.get())
            self.check_environment()
            output = (
                ROOT
                / "outputs"
                / ("gui-" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
            )
            seed = secrets.randbelow(2**31)
            command = launch_command(
                ROOT, cuboids, cylinders, seed, output, self.layout["simulation"]
            )

            # 기존 터미널에서 켜둔 기본 ROS 도메인과 섞이지 않도록 이 실행은 별도 영역을 쓴다.
            environment = dict(os.environ)
            environment.update(
                ROS_DOMAIN_ID=str(160 + secrets.randbelow(60)),
                ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
                PYTHONUNBUFFERED="1",
            )
            self.run = SimulationProcess(command, ROOT, output, environment)
            self.run.start()
        except (ValueError, OSError) as error:
            self.status.set("실행 준비 실패")
            self.append_log(str(error) + "\n", "error")
            self.run = None
            return

        self.result = None
        self.user_stopped = False
        self.closed_handled = False
        self.status.set("준비 중")
        self.append_log(
            f"\n── 새 시뮬레이션: 직육면체 {cuboids}개 · 원기둥 {cylinders}개 ──\n",
            "note",
        )
        for field in self.count_inputs:
            field.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def stop_run(self):
        if self.run is None or self.run.finished.is_set():
            return
        self.user_stopped = True
        self.status.set("종료 중")
        self.stop_button.configure(state="disabled")
        self.append_log("시뮬레이션 종료를 요청했습니다.\n", "note")
        self.run.stop()

    def poll_run(self):
        # 한 번에 처리할 로그 수를 제한해 출력이 많아도 버튼·창 이동이 계속 반응하게 한다.
        if self.run is not None:
            for _ in range(200):
                try:
                    kind, data = self.run.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "log":
                    self.append_log(
                        data,
                        "error" if "[ERROR]" in data or "Traceback" in data else None,
                    )
                    if (
                        "물체 위로 접근" in data
                        and self.result is None
                        and not self.user_stopped
                    ):
                        self.status.set("운반 중")
                elif kind == "error":
                    self.append_log(data + "\n", "error")

            # 성공은 로그 문구가 아니라 관리 노드가 기록한 결과 파일로 확인한다.
            result_path = self.run.output / "sort_result.json"
            if self.result is None and result_path.is_file():
                try:
                    self.result = json.loads(result_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
                else:
                    if not self.user_stopped:
                        self.status.set(
                            "완료 · 장면 유지 중"
                            if self.result["success"]
                            else "실패 · 로그 확인"
                        )

            if self.run.finished.is_set() and not self.closed_handled:
                self.closed_handled = True
                for field in self.count_inputs:
                    field.configure(state="normal")
                self.stop_button.configure(state="disabled")
                if self.result is None:
                    self.status.set(
                        "종료됨" if self.user_stopped else "실행 종료 · 로그 확인"
                    )
                elif self.user_stopped:
                    self.status.set("종료됨")
                self.on_count_changed()
                self.append_log(
                    "시뮬레이션 프로세스를 정리했습니다. 다시 시작할 수 있습니다.\n",
                    "note",
                )

        # 창 닫기도 작업 스레드가 정리될 때까지 GUI 이벤트를 유지한 뒤 끝낸다.
        if self.closing and (self.run is None or self.run.finished.is_set()):
            self.root.destroy()
            return
        self.root.after(80, self.poll_run)

    def close_app(self):
        self.closing = True
        self.start_button.configure(state="disabled")
        self.stop_run()
        if self.run is None or self.run.finished.is_set():
            self.root.destroy()


def main():
    # GUI가 필요한 경우에만 tkinter를 불러온다. 계산·프로세스 테스트는 창 없이 실행한다.
    import tkinter as tk

    root = tk.Tk()
    ShapeSortApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

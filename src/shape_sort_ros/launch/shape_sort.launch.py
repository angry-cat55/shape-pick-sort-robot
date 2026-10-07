"""세 노드를 같은 설정으로 실행한다. 기존 venv의 Python을 명시해서 사용한다."""

from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # 실행 위치가 달라도 되도록 모델·결과·Python 경로를 launch 인자로 받는다.
    root = Path.cwd()
    default_output = (
        root / "outputs" / ("ros2-" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    )
    defaults = {
        "python_executable": str(root / ".venv/bin/python"),
        "model_dir": str(root / "checkpoints/shape_cnn_v1"),
        "output_dir": str(default_output),
        "mode": "gui",
        "cuboids": "3",
        "cylinders": "2",
        "seed": "0",
        "size_mode": "random",
        "auto_start": "false",
        "max_retries": "2",
        "timeout_s": "90.0",
    }
    arguments = [
        DeclareLaunchArgument(name, default_value=value)
        for name, value in defaults.items()
    ]

    def value(name, value_type):
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    # Python 노드의 shebang 대신 지정한 venv Python으로 실행해 ROS2와 torch를 같이 쓴다.
    common = {
        "package": "shape_sort_ros",
        "output": "screen",
        "prefix": [LaunchConfiguration("python_executable"), " "],
    }
    simulation = Node(
        **common,
        executable="simulation_node",
        name="simulation_node",
        # ROS 노드 이름과 터미널의 프로세스 표시 이름을 함께 맞춘다.
        exec_name="simulation_node",
        output_format="[simulation_node] {line}",
        parameters=[
            {
                "mode": value("mode", str),
                "cuboids": value("cuboids", int),
                "cylinders": value("cylinders", int),
                "seed": value("seed", int),
                "size_mode": value("size_mode", str),
                "output_dir": value("output_dir", str),
            }
        ],
    )
    perception = Node(
        **common,
        executable="perception_node",
        name="perception_node",
        exec_name="perception_node",
        output_format="[perception_node] {line}",
        parameters=[
            {
                "model_dir": value("model_dir", str),
                "output_dir": value("output_dir", str),
            }
        ],
    )
    manager = Node(
        **common,
        executable="task_manager_node",
        name="task_manager_node",
        exec_name="task_manager_node",
        output_format="[task_manager_node] {line}",
        parameters=[
            {
                "auto_start": value("auto_start", bool),
                "max_retries": value("max_retries", int),
                "timeout_s": value("timeout_s", float),
                "output_dir": value("output_dir", str),
            }
        ],
    )
    return LaunchDescription(arguments + [simulation, perception, manager])

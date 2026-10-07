from glob import glob
from setuptools import setup

# 기존 스크립트를 설치할 때도 함께 넣는다. 소스 경로를 sys.path에 추가하지 않는다.
setup(
    name="shape_sort_ros",
    version="0.1.0",
    packages=["shape_sort_ros"],
    package_dir={"": "../../scripts", "shape_sort_ros": "shape_sort_ros"},
    py_modules=[
        "multi_object_scene",
        "contact_grasp_probe",
        "rgbd_camera",
        "cnn_inference",
        "terminal_ko",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/shape_sort_ros"]),
        ("share/shape_sort_ros", ["package.xml"]),
        ("share/shape_sort_ros/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Jihwan Yu",
    maintainer_email="applewlghks321@gmail.com",
    description="카메라 인식과 접촉 운반을 연결하는 세 ROS2 노드",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "simulation_node = shape_sort_ros.simulation_node:main",
            "perception_node = shape_sort_ros.perception_node:main",
            "task_manager_node = shape_sort_ros.task_manager_node:main",
        ],
    },
)

cmake_minimum_required(VERSION 3.8)
project(shape_sort_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# 노드들이 같은 형식으로 읽도록 메시지와 서비스의 코드를 생성한다.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraObservation.msg"
  "msg/ObjectCandidate.msg"
  "msg/PerceptionResult.msg"
  "msg/TaskStatus.msg"
  "srv/RobotCommand.srv"
  DEPENDENCIES std_msgs sensor_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()

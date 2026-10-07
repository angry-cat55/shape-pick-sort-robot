"""관리 노드의 재시도 한도·오래된 결과·통신 중단 처리를 확인한다."""

import importlib.util
import tempfile
import time
import unittest

ROS_AVAILABLE = importlib.util.find_spec('shape_sort_interfaces') is not None


@unittest.skipUnless(ROS_AVAILABLE, 'ROS2 빌드 환경에서 실행하는 테스트')
class TaskManagerTests(unittest.TestCase):
    def setUp(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from shape_sort_interfaces.srv import RobotCommand
        from std_srvs.srv import Trigger
        from shape_sort_ros.task_manager_node import TaskManagerNode

        # 실제 ROS 서비스를 준비해, 관리 노드가 비동기 명령을 접수하는 경로를 사용한다.
        self.folder = tempfile.TemporaryDirectory()
        rclpy.init(args=['--ros-args', '-p', 'output_dir:=' + self.folder.name])
        self.robot = rclpy.create_node('policy_test_robot')

        def accept(request, response):
            response.accepted = True
            return response

        def ready(request, response):
            response.success = True
            return response

        self.robot.create_service(RobotCommand, '/shape_sort/robot_command', accept)
        self.robot.create_service(Trigger, '/shape_sort/perception_ready', ready)
        self.manager = TaskManagerNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.robot)
        self.executor.add_node(self.manager)
        deadline = time.monotonic() + 3
        while not self.manager.robot_client.service_is_ready():
            self.executor.spin_once(timeout_sec=0.01)
            self.assertLess(time.monotonic(), deadline)
        self.manager.selected = dict(
            valid=True,
            object_shape='cuboid',
            center_xy_m=[0.5, 0.0],
            top_z_m=0.36,
            width_m=0.04,
            yaw_rad=0.0,
            class_score=0.95,
            top_pixels=100,
        )
        self.manager.remaining_count = 1

    def tearDown(self):
        import rclpy

        self.executor.shutdown()
        self.manager.destroy_node()
        self.robot.destroy_node()
        rclpy.shutdown()
        self.folder.cleanup()

    def complete_current(self, success=True, reason=''):
        from shape_sort_interfaces.msg import TaskStatus

        request = self.manager.pending
        status = TaskStatus(
            command_id=request.command_id,
            scan_id=request.scan_id,
            source='simulation',
            stage=request.command.upper() + '_DONE',
            completed=True,
            success=success,
            failure_reason=reason,
            result_json='{"remaining_count": 1}',
        )
        self.manager.on_robot_status(status)

    def test_two_retries_then_recover_and_stop(self):
        for expected_count in (1, 2):
            self.manager.handle_failure('pick', 'lift_not_verified')
            self.assertEqual(self.manager.retry_count, expected_count)
            self.assertEqual(self.manager.phase, 'recover')
            self.complete_current()
            self.assertEqual(self.manager.phase, 'scan')
            self.assertEqual(self.manager.retry_xy, [0.5, 0.0])
            self.assertTrue(self.manager.pending.all_candidates)
        self.manager.handle_failure('pick', 'lift_not_verified')
        self.assertEqual(self.manager.retry_count, 2)
        self.complete_current()
        self.assertEqual(self.manager.phase, 'finished')
        self.assertEqual(self.manager.report['failure_reason'], 'retry_limit_reached')
        self.assertEqual(len(self.manager.report['recoveries']), 3)

    def test_collision_does_not_retry(self):
        self.manager.handle_failure('pick', 'arm_bin_collision')
        self.assertEqual(self.manager.phase, 'finished')
        self.assertEqual(self.manager.retry_count, 0)
        self.assertEqual(self.manager.report['failure_reason'], 'arm_bin_collision')

    def test_zero_retries_stops_on_first_miss(self):
        self.manager.max_retries = 0
        self.manager.handle_failure('pick', 'lift_not_verified')
        self.assertEqual(self.manager.phase, 'finished')
        self.assertFalse(self.manager.report['recoveries'])

    def test_old_command_and_scan_results_are_ignored(self):
        from shape_sort_interfaces.msg import TaskStatus, PerceptionResult

        self.manager.request_scan()
        pending = self.manager.pending
        self.manager.on_robot_status(
            TaskStatus(command_id=999, scan_id=999, completed=True, success=True)
        )
        self.manager.on_perception(PerceptionResult(scan_id=999, valid=True))
        self.assertIs(self.manager.pending, pending)
        self.assertEqual(self.manager.phase, 'scan')
        self.assertFalse(self.manager.report['scans'])

    def test_command_timeout_finishes_failed(self):
        self.manager.send_command('pick')
        self.manager.deadline = time.monotonic() - 1
        self.manager.check_progress()
        self.assertEqual(self.manager.phase, 'finished')
        self.assertEqual(self.manager.report['failure_reason'], 'communication_timeout')

    def test_ready_manual_scene_can_wait_without_timeout(self):
        self.manager.robot_ready = True
        self.manager.deadline = time.monotonic() - 1
        self.manager.check_progress()
        self.assertEqual(self.manager.phase, 'idle')

    def test_recovery_failure_stops_without_more_attempts(self):
        self.manager.handle_failure('pick', 'drop_during_transport')
        self.complete_current(False, 'arm_bin_collision')
        self.assertEqual(self.manager.report['failure_reason'], 'recovery_failed')
        self.assertEqual(self.manager.retry_count, 1)

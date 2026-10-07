"""설치된 세 ROS2 노드의 실제 통신·접촉 운반·중단을 검사한다."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
ROS_AVAILABLE = (
    importlib.util.find_spec('shape_sort_interfaces') is not None
    and os.environ.get('SHAPE_SORT_ROS_INTEGRATION') == '1'
)


@unittest.skipUnless(
    ROS_AVAILABLE, '빌드·source 후 SHAPE_SORT_ROS_INTEGRATION=1로 실행'
)
class RosNodeIntegrationTests(unittest.TestCase):
    def run_scene(self, name, domain, extra_args, stop_during_pick=False):
        import rclpy
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from shape_sort_interfaces.msg import TaskStatus
        from shape_sort_interfaces.srv import RobotCommand
        from std_srvs.srv import Trigger

        # 테스트마다 독립 DDS 도메인을 써서 다른 장면의 토픽·서비스와 섞이지 않게 한다.
        environment = dict(
            os.environ,
            ROS_DOMAIN_ID=str(domain),
            ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
        )
        folder = ROOT / 'outputs/ros2-validation' / name
        folder.mkdir(parents=True, exist_ok=True)
        log = (folder / 'terminal.log').open('w')
        launch = subprocess.Popen(
            [
                'ros2',
                'launch',
                'shape_sort_ros',
                'shape_sort.launch.py',
                'mode:=direct',
                'auto_start:=false',
                'timeout_s:=10.0',
                f'output_dir:={folder}',
                *extra_args,
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        from rclpy.executors import SingleThreadedExecutor

        context = rclpy.Context()
        executor = None
        previous_domain = os.environ.get('ROS_DOMAIN_ID')
        os.environ['ROS_DOMAIN_ID'] = str(domain)
        rclpy.init(context=context)
        node = rclpy.create_node('integration_observer', context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        robot_status = []
        task_status = []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(
            TaskStatus, '/shape_sort/robot_status', robot_status.append, qos
        )
        node.create_subscription(
            TaskStatus, '/shape_sort/task_status', task_status.append, qos
        )
        ready = node.create_client(Trigger, '/shape_sort/perception_ready')
        start = node.create_client(Trigger, '/shape_sort/start')
        stop = node.create_client(Trigger, '/shape_sort/stop')
        command = node.create_client(RobotCommand, '/shape_sort/robot_command')

        def wait_for(condition, seconds=30):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
                if condition():
                    return
                if launch.poll() is not None:
                    self.fail(
                        'launch가 예상보다 먼저 종료됨: '
                        + (folder / 'terminal.log').read_text()[-3000:]
                    )
            self.fail(
                'ROS 응답 시간 초과: ' + (folder / 'terminal.log').read_text()[-3000:]
            )

        try:
            wait_for(
                lambda: start.service_is_ready()
                and ready.service_is_ready()
                # 명령 서비스도 이 관측 노드에서 발견된 뒤 요청한다.
                and command.service_is_ready()
                and any(status.stage == 'READY' for status in task_status)
            )
            # 잘못된 명령이 팔 이동으로 이어지지 않는지 실제 서비스로 확인한다.
            request = RobotCommand.Request(command_id=1, command='unknown')
            rejected = command.call_async(request)
            wait_for(rejected.done)
            self.assertFalse(rejected.result().accepted)
            request = RobotCommand.Request(command_id=1, scan_id=99, command='pick')
            rejected = command.call_async(request)
            wait_for(rejected.done)
            self.assertFalse(rejected.result().accepted)

            started = start.call_async(Trigger.Request())
            wait_for(started.done)
            self.assertTrue(started.result().success)
            if stop_during_pick:
                wait_for(
                    lambda: any(status.stage == 'APPROACH' for status in robot_status)
                )
                stopped = stop.call_async(Trigger.Request())
                wait_for(stopped.done)
                self.assertTrue(stopped.result().success)
            wait_for(
                lambda: any(status.stage == 'FINISHED' for status in task_status),
                seconds=60,
            )
            result = json.loads((folder / 'sort_result.json').read_text())
            if stop_during_pick:
                self.assertFalse(result['success'])
                self.assertEqual(result['failure_reason'], 'task_stopped')
                wait_for(
                    lambda: any(
                        status.completed and status.failure_reason == 'task_stopped'
                        for status in robot_status
                    )
                )
            else:
                self.assertTrue(result['success'], result)
                self.assertTrue(all(result['final_arrivals'].values()))
                self.assertTrue(
                    all(pick['correct_bin_arrival'] for pick in result['picks'])
                )
                self.assertTrue(
                    all(pick['classification_correct'] for pick in result['picks'])
                )
                # 보류된 후보는 다음 후보까지 분류할 수 있다. 첫 유효 후보 뒤에는 멈춰야 한다.
                for scan in result['scans']:
                    selected = False
                    for candidate in scan['regions']:
                        if selected:
                            self.assertFalse(candidate['cnn_evaluated'])
                        selected = selected or candidate['valid']
            return result
        finally:
            executor.shutdown()
            node.destroy_node()
            context.shutdown()
            if previous_domain is None:
                os.environ.pop('ROS_DOMAIN_ID', None)
            else:
                os.environ['ROS_DOMAIN_ID'] = previous_domain
            launch.send_signal(signal.SIGINT)
            try:
                launch.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(launch.pid, signal.SIGTERM)
                launch.wait(timeout=5)
            log.close()

    def test_random_mixed_five_objects_through_ros(self):
        result = self.run_scene(
            'integration-random-five',
            131,
            ['cuboids:=3', 'cylinders:=2', 'seed:=1', 'size_mode:=random'],
        )
        self.assertEqual(len(result['picks']), 5)
        self.assertEqual(len(result['final_arrivals']), 5)

    def test_stop_interrupts_actual_robot_motion(self):
        self.run_scene(
            'integration-stop',
            132,
            ['cuboids:=1', 'cylinders:=1', 'seed:=0', 'size_mode:=fixed'],
            stop_during_pick=True,
        )

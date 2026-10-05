"""실제 모터로 실패를 만들고 재촬영·복귀·횟수 제한을 확인한다."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import multi_object_scene as scene_code


class GraspRetryTests(unittest.TestCase):
    def setUp(self):
        self.client = p.connect(p.DIRECT)
        self.scene = scene_code.build_scene(
            [{'shape': 'cuboid', 'xy': [0.5, 0], 'yaw': 0}], scene_code.LAYOUT)
        self.assertTrue(scene_code.settle_scene(self.scene)['settled'])
        self.folder = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.folder.cleanup()
        p.disconnect(self.client)

    def run_sort(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return scene_code.sort_scene(self.scene, 'cuboid', Path(self.folder.name), **kwargs)

    def test_failed_close_rescans_then_physically_places_object(self):
        original = scene_code.SceneProbe.gripper
        closes = []
        def miss_first(probe, opening):
            if opening == 0:
                closes.append(opening)
                if len(closes) == 1:
                    return original(probe, 0.04)
            return original(probe, opening)
        with patch.object(scene_code.SceneProbe, 'gripper', miss_first):
            result = self.run_sort(max_retries=2)
        self.assertTrue(result['success'], result['failure_reason'])
        self.assertEqual(len(closes), 2)
        self.assertFalse(result['picks'][0]['lift_success'])
        self.assertTrue(result['picks'][1]['correct_bin_arrival'])
        self.assertGreater(result['picks'][1]['scan_index'], result['picks'][0]['scan_index'])
        self.assertEqual(result['recoveries'][0]['cause'], 'lift_not_verified')
        self.assertEqual(result['recoveries'][0]['status'], 'ready_to_rescan')

    def test_repeated_misses_stop_after_initial_plus_two_retries(self):
        original = scene_code.SceneProbe.gripper
        def miss_every_close(probe, opening):
            return original(probe, 0.04 if opening == 0 else opening)
        with patch.object(scene_code.SceneProbe, 'gripper', miss_every_close):
            result = self.run_sort(max_retries=2)
        self.assertFalse(result['success'])
        self.assertEqual(len(result['picks']), 3)
        self.assertEqual(sum(r['status'] == 'ready_to_rescan' for r in result['recoveries']), 2)
        self.assertEqual(result['recoveries'][-1]['status'], 'stopped_after_limit')
        actual = [p.getJointState(self.scene['robot'], i)[0] for i in range(7)]
        self.assertLess(max(abs(a-b) for a, b in zip(actual, self.scene['home'][:7])), 0.01)
        self.assertEqual(result['failure_reason'], 'retry_limit_reached')
        self.assertEqual(result['last_failure_reason'], 'lift_not_verified')
        self.assertFalse(any(pick['arrival_success'] for pick in result['picks']))

    def test_opening_during_clearance_recovers_from_real_drop(self):
        original = scene_code.SceneProbe.move
        injected = []
        def drop_once(probe, stage, position, seconds):
            value = original(probe, stage, position, seconds)
            if stage == 'CLEARANCE' and not injected:
                injected.append(True)
                # 실제로 손가락을 열어 낙하를 만들고 기존 상대 위치 검사로 확인한다.
                probe.gripper(0.04)
                probe.wait('FORCED_TRANSPORT_DROP', 1.0)
                self.assertTrue(probe.drop_via_relative_motion)
            return value
        with patch.object(scene_code.SceneProbe, 'move', drop_once):
            result = self.run_sort(max_retries=2)
        self.assertTrue(result['success'], result['failure_reason'])
        self.assertEqual(len(result['picks']), 2)
        self.assertEqual(result['recoveries'][0]['cause'], 'drop_during_transport')
        self.assertTrue(result['picks'][1]['correct_bin_arrival'])

    def test_collision_reason_stops_without_recovery(self):
        original = scene_code.SceneProbe.move
        def collision_stop(probe, stage, position, seconds):
            if stage == 'APPROACH':
                raise RuntimeError('arm_table_collision')
            return original(probe, stage, position, seconds)
        with patch.object(scene_code.SceneProbe, 'move', collision_stop):
            result = self.run_sort(max_retries=2)
        self.assertFalse(result['success'])
        self.assertEqual(result['failure_reason'], 'arm_table_collision')
        self.assertEqual(result['recoveries'], [])
        self.assertEqual(len(result['picks']), 1)

    def test_drop_stops_carry_before_completing_four_seconds(self):
        original_arc = scene_code.SceneProbe.carry_arc
        original_step = scene_code.SceneProbe.step
        arc_ticks = []
        injected = []
        def open_at_start(probe, direction, seconds=4.0, end_angle=None):
            if end_angle is None and not injected:
                injected.append(True)
                probe.gripper(0.04)
            return original_arc(probe, direction, seconds, end_angle=end_angle)
        def count_first_arc(probe):
            if probe.stage == 'CARRY_ARC' and len(arc_ticks) == 0:
                arc_ticks.append(0)
            if probe.stage == 'CARRY_ARC' and len(injected) == 1 and not getattr(probe, '_first_arc_finished', False):
                arc_ticks[0] += 1
            try:
                return original_step(probe)
            except RuntimeError:
                probe._first_arc_finished = True
                raise
        with patch.object(scene_code.SceneProbe, 'carry_arc', open_at_start), \
             patch.object(scene_code.SceneProbe, 'step', count_first_arc):
            result = self.run_sort(max_retries=2)
        self.assertLess(arc_ticks[0], round(4 / scene_code.CONFIG['dt_s']))
        self.assertEqual(result['recoveries'][0]['cause'], 'drop_during_transport')
        self.assertTrue(result['success'], result['failure_reason'])

    def test_retry_disabled_keeps_first_failure(self):
        original = scene_code.SceneProbe.gripper
        with patch.object(scene_code.SceneProbe, 'gripper', lambda probe, opening: original(probe, 0.04)):
            result = self.run_sort(max_retries=0)
        self.assertEqual(result['failure_reason'], 'lift_not_verified')
        self.assertEqual(len(result['picks']), 1)
        self.assertEqual(result['recoveries'], [])

    def test_drop_behind_robot_recovers_then_stops_if_target_is_outside_camera_roi(self):
        original = scene_code.SceneProbe.carry_arc
        injected = []
        def drop_behind(probe, direction, seconds=4.0, end_angle=None):
            value = original(probe, direction, seconds, end_angle=end_angle)
            if end_angle is None and not injected:
                injected.append(True)
                probe.gripper(0.04)
                probe.wait('FORCED_TRANSPORT_DROP', 1.0)
            return value
        with patch.object(scene_code.SceneProbe, 'carry_arc', drop_behind):
            result = self.run_sort(max_retries=2)
        self.assertFalse(result['success'])
        self.assertEqual(result['failure_reason'], 'retry_target_not_found')
        self.assertEqual(len(result['picks']), 1)
        self.assertEqual(result['recoveries'][0]['status'], 'ready_to_rescan')

    def test_recovery_error_stops_without_another_attempt(self):
        original = scene_code.SceneProbe.gripper
        with patch.object(scene_code.SceneProbe, 'gripper', lambda probe, opening: original(probe, 0.04)), \
             patch.object(scene_code, 'recover_pick', side_effect=RuntimeError('arm_bin_collision')):
            result = self.run_sort(max_retries=2)
        self.assertFalse(result['success'])
        self.assertEqual(result['failure_reason'], 'recovery_failed')
        self.assertEqual(len(result['picks']), 1)
        self.assertEqual(result['recoveries'][0]['failure_reason'], 'arm_bin_collision')

    def test_final_verification_collision_is_saved_as_failure(self):
        original = scene_code.SceneProbe.step
        def collision_at_final(probe):
            if probe.stage == 'VERIFY_FINAL_ARRIVAL':
                raise RuntimeError('arm_bin_collision')
            return original(probe)
        with patch.object(scene_code.SceneProbe, 'step', collision_at_final):
            result = self.run_sort(max_retries=2)
        self.assertFalse(result['success'])
        self.assertEqual(result['failure_reason'], 'arm_bin_collision')
        self.assertTrue((Path(self.folder.name)/'sort_result.json').exists())

    def test_invalid_retry_limit_is_rejected(self):
        for limit in (-1, 3, True, 1.5):
            with self.assertRaises(ValueError):
                self.run_sort(max_retries=limit)

"""ROS 환경에서 영상·후보 전달과 실패 후 대상 선택을 확인한다."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/shape_sort_ros'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
ROS_AVAILABLE = importlib.util.find_spec('shape_sort_interfaces') is not None


@unittest.skipUnless(ROS_AVAILABLE, 'ROS2 빌드 환경에서 실행하는 테스트')
class RosProtocolTests(unittest.TestCase):
    def setUp(self):
        from shape_sort_ros import protocol

        self.protocol = protocol

    def test_depth_round_trip_preserves_exact_values(self):
        # float32 반올림으로 좌표 계산이 달라지지 않게 원래 float64를 그대로 전달한다.
        from std_msgs.msg import Header

        values = np.array([[0.123456789012345, 0.99], [0.5, 1.0]])
        message = self.protocol.image_message(values, Header(), '64FC1')
        np.testing.assert_array_equal(
            self.protocol.image_array(message, '64FC1'), values
        )

    def test_corrupt_image_is_rejected(self):
        from sensor_msgs.msg import Image

        message = Image(height=2, width=2, encoding='64FC1', step=16, data=b'x')
        with self.assertRaises(ValueError):
            self.protocol.image_array(message, '64FC1')

    def test_rgb_round_trip_and_wrong_encoding(self):
        from std_msgs.msg import Header

        values = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        message = self.protocol.image_message(values, Header(), 'rgb8')
        np.testing.assert_array_equal(
            self.protocol.image_array(message, 'rgb8'), values
        )
        with self.assertRaises(ValueError):
            self.protocol.image_array(message, '64FC1')

    def test_candidate_geometry_survives_round_trip(self):
        geometry = dict(
            valid=True,
            object_shape='cuboid',
            center_xy_m=[0.5, 0.0],
            top_z_m=0.36,
            width_m=0.04,
            yaw_rad=-0.9,
            top_pixels=42,
            cnn_evaluated=True,
            class_score=0.95,
        )
        message = self.protocol.candidate_message(geometry)
        decoded = self.protocol.candidate_dict(message)
        for key, value in geometry.items():
            self.assertEqual(decoded[key], value)
        self.protocol.validate_target(decoded)
        decoded['center_xy_m'] = [float('nan'), 0.0]
        with self.assertRaises(ValueError):
            self.protocol.validate_target(decoded)

    def test_retry_target_missing_and_ambiguous(self):
        candidates = [dict(valid=True, center_xy_m=[0.5, 0.0], top_pixels=50)]
        self.assertEqual(
            self.protocol.choose_target(candidates, [0.51, 0.0]), candidates[0]
        )
        with self.assertRaisesRegex(RuntimeError, 'retry_target_not_found'):
            self.protocol.choose_target(candidates, [0.6, 0.0])
        candidates.append(dict(valid=True, center_xy_m=[0.51, 0.0], top_pixels=40))
        with self.assertRaisesRegex(RuntimeError, 'retry_target_ambiguous'):
            self.protocol.choose_target(candidates, [0.5, 0.0])

    def test_geometry_bounds_and_low_score_are_rejected(self):
        geometry = dict(
            valid=True,
            object_shape='cuboid',
            center_xy_m=[0.5, 0.0],
            top_z_m=0.36,
            width_m=0.04,
            yaw_rad=0.0,
            class_score=0.95,
        )
        for changes in (
            {'width_m': 0.2},
            {'object_shape': 'unknown'},
            {'class_score': 0.1},
            {'top_z_m': 0.9},
            {'center_xy_m': [-0.4, 0.0]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.protocol.validate_target({**geometry, **changes})

    def test_candidate_details_must_be_a_dictionary(self):
        from shape_sort_interfaces.msg import ObjectCandidate

        message = ObjectCandidate(details_json='[]')
        with self.assertRaises(ValueError):
            self.protocol.candidate_dict(message)

"""학습과 같은 입력·가중치 복원 및 영상만으로 혼합 물체를 분류하는지 확인한다."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class CnnInferenceTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('cnn_inference'), 'CNN 추론 모듈이 아직 없습니다')
        import cnn_inference
        return cnn_inference

    def test_normalization_and_logits_match_training_notebook(self):
        m = self.module()
        import torch
        # 모델·노트북 구조의 일치를 실제 출력으로 비교한다. 개인 가중치 없이도 실행되는 테스트다.
        nb = json.loads((ROOT / 'notebooks/shape_cnn_colab.ipynb').read_text())
        source = ''.join(next(c['source'] for c in nb['cells'] if 'model' in c.get('metadata', {}).get('tags', [])))
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'SmallCNN')
        ns = {'nn': torch.nn, 'CLASSES': ['cuboid', 'cylinder']}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<training model>', 'exec'), ns)
        model = m.SmallCNN(0.15).eval()
        reference = ns['SmallCNN'](0.15).eval()
        reference.load_state_dict(model.state_dict())
        raw = np.zeros((64, 64, 3), dtype=np.uint8)
        raw[:, :, 1] = 127; raw[:, :, 2] = 255
        x = m.preprocess(Image.fromarray(raw), [0.5]*3, [0.5]*3)
        torch.testing.assert_close(x, (torch.tensor(raw.copy()).permute(2,0,1).float()/255-0.5)/0.5)
        with torch.inference_mode(): torch.testing.assert_close(model(x[None]), reference(x[None]))
        with self.assertRaises(ValueError): m.preprocess(Image.new('RGB', (32,32)), [0.5]*3, [0.5]*3)

    def test_synthetic_checkpoint_rejects_mismatched_metadata_and_camera(self):
        m = self.module()
        import torch
        from multi_object_scene import scene_camera, LAYOUT
        camera = scene_camera(LAYOUT)
        metadata = {'model':'SmallCNN-v1','classes':['cuboid','cylinder'],'image_size':[64,64],
                    'mean':[0.5]*3,'std':[0.5]*3,'crop_total_margin_px':12,'resize':'PIL_BILINEAR',
                    'camera':camera,'params':{'dropout':0.15}}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'best_params.json').write_text(json.dumps(metadata))
            torch.save({'model_state_dict':m.SmallCNN().state_dict(),'metadata':metadata,'params':metadata['params']}, folder/'best_model.pt')
            classifier = m.CnnClassifier(folder)
            classifier.validate_camera(camera)
            with self.assertRaises(ValueError): classifier.validate_camera({**camera,'fov_deg':45})
            prediction = classifier.predict(Image.new('RGB',(64,64)))
            self.assertIn(prediction['predicted_class'], metadata['classes'])
            self.assertAlmostEqual(sum(prediction['class_probabilities'].values()),1,places=6)
            metadata['classes'].reverse()
            (folder/'best_params.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError): m.CnnClassifier(folder)

    def test_uncertain_prediction_and_invalid_depth_do_not_start_grasp(self):
        m = self.module()
        import torch
        from multi_object_scene import build_scene, sample_spawns, settle_scene, sort_scene, LAYOUT
        from rgbd_camera import capture, estimate_many
        client = p.connect(p.DIRECT)
        try:
            scene = build_scene(sample_spawns(1, 0, 0, LAYOUT['region']), LAYOUT)
            self.assertTrue(settle_scene(scene)['settled'])
            metadata = {'model': 'SmallCNN-v1', 'classes': ['cuboid', 'cylinder'], 'image_size': [64,64],
                        'mean': [0.5]*3, 'std': [0.5]*3, 'crop_total_margin_px': 12,
                        'resize': 'PIL_BILINEAR', 'camera': scene['camera'], 'params': {'dropout': 0.15}}
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                # 두 클래스에 같은 점수를 내는 실제 CNN으로 불확실한 입력을 재현한다.
                model = m.SmallCNN()
                with torch.no_grad():
                    for parameter in model.parameters(): parameter.zero_()
                (folder/'best_params.json').write_text(json.dumps(metadata))
                torch.save({'model_state_dict': model.state_dict(), 'metadata': metadata,
                            'params': metadata['params']}, folder/'best_model.pt')
                classifier = m.CnnClassifier(folder)
                observation = capture(scene['camera'])
                candidates = estimate_many(observation, scene['empty'], scene['camera'], None, classifier=classifier)
                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]['failure_reason'], 'low_class_confidence')
                self.assertFalse(candidates[0]['valid'])
                result = sort_scene(scene, None, folder/'sort', classifier=classifier)
                self.assertFalse(result['success'])
                self.assertEqual(result['picks'], [])
                # 같은 낮은 점수가 계속되면 재촬영 두 번 뒤 끝내고 집기를 시작하지 않는다.
                self.assertEqual(len(result['scans']), 3)
                self.assertEqual(len(result['recoveries']), 2)
                self.assertEqual(result['failure_reason'], 'retry_limit_reached')
                # 분류기를 연결해도 손상된 깊이값을 집기 좌표로 사용하면 안 된다.
                invalid = {**observation, 'depth': np.full_like(observation['depth'], np.nan)}
                rejected = estimate_many(invalid, scene['empty'], scene['camera'], None, classifier=classifier)
                self.assertEqual(rejected[0]['failure_reason'], 'invalid_depth')
                with self.assertRaises(ValueError):
                    estimate_many(observation, scene['empty'], scene['camera'], None,
                                  supported_top_m=[0.06,0.04], classifier=classifier)
        finally:
            p.disconnect(client)

    def test_priority_stops_after_first_valid_and_checks_next_after_rejection(self):
        from multi_object_scene import build_scene, sample_spawns, settle_scene, LAYOUT
        from rgbd_camera import capture, estimate_many
        from unittest.mock import Mock
        client = p.connect(p.DIRECT)
        try:
            scene = build_scene(sample_spawns(3, 0, 0, LAYOUT['region']), LAYOUT)
            self.assertTrue(settle_scene(scene)['settled'])
            observation = capture(scene['camera'])
            accepted = {'predicted_class': 'cuboid', 'class_score': 1.0,
                        'class_probabilities': {'cuboid': 1.0, 'cylinder': 0.0}}
            # 후보 선택 검증에는 점수와 호출 횟수를 확인할 수 있는 분류기를 쓴다.
            for rejected in [None, {**accepted, 'class_score': 0.5},
                             {**accepted, 'predicted_class': 'cylinder'}]:
                classifier = Mock()
                classifier.predict.side_effect = ([rejected, accepted] if rejected else [accepted])
                candidates = estimate_many(observation, scene['empty'], scene['camera'], None,
                                           classifier=classifier, first_valid_only=True)
                self.assertEqual(len(candidates), 3)
                self.assertEqual(classifier.predict.call_count, 2 if rejected else 1)
                self.assertEqual(sum(c['valid'] for c in candidates), 1)
                self.assertEqual(sum(c['cnn_evaluated'] for c in candidates), 2 if rejected else 1)
                self.assertEqual([c['priority_top_pixels'] for c in candidates],
                                 sorted((c['priority_top_pixels'] for c in candidates), reverse=True))
                if rejected:
                    self.assertFalse(candidates[0]['valid'])
                    self.assertTrue(candidates[1]['valid'])
                self.assertEqual(candidates[-1]['failure_reason'], 'not_evaluated')
        finally:
            p.disconnect(client)

    def test_real_weights_classify_both_shapes_and_transport_mixed_scene(self):
        m = self.module()
        folder = ROOT/'checkpoints/shape_cnn_v1'
        if not (folder/'best_model.pt').is_file(): self.skipTest('공개 학습 가중치 best_model.pt가 없습니다')
        from multi_object_scene import build_scene, sample_spawns, settle_scene, sort_scene, LAYOUT
        from rgbd_camera import capture, estimate_many
        import inspect
        self.assertIn('classifier', inspect.signature(estimate_many).parameters)
        self.assertIn('classifier', inspect.signature(sort_scene).parameters)
        classifier = m.CnnClassifier(folder)
        client = p.connect(p.DIRECT)
        try:
            scene = build_scene(sample_spawns(1,1,0,LAYOUT['region']),LAYOUT)
            self.assertTrue(settle_scene(scene)['settled'])
            classifier.validate_camera(scene['camera'])
            candidates = estimate_many(capture(scene['camera']),scene['empty'],scene['camera'],None,classifier=classifier)
            self.assertEqual({g['predicted_class'] for g in candidates},{'cuboid','cylinder'})
            self.assertTrue(all(g['valid'] for g in candidates),candidates)
            # 생성 정답은 평가에만 쓴다. 추정 API에는 물체 목록을 전달하지 않는다.
            with tempfile.TemporaryDirectory() as directory:
                result=sort_scene(scene,None,Path(directory),classifier=classifier)
                self.assertTrue(result['success'],result)
                self.assertEqual(result['classification_source'],'cnn')
                self.assertEqual(len(result['picks']),2)
                self.assertEqual(len(result['scans']),3)
                self.assertTrue(all(g['classification_correct'] and g['correct_bin_arrival'] for g in result['picks']))
                self.assertEqual(sum(c['cnn_evaluated'] for scan in result['scans'] for c in scan['regions']), 2)
        finally: p.disconnect(client)


if __name__ == '__main__': unittest.main()

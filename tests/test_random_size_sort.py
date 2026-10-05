"""랜덤 치수·영상 기반 지원 범위·가림 보류를 검증한다."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from multi_object_scene import LAYOUT, sample_spawns, build_scene, settle_scene
from rgbd_camera import capture, estimate_many

RANGE = (0.03, 0.064, 0.03, 0.07)


class RandomSizeTests(unittest.TestCase):
    def test_random_spawns_keep_counts_range_and_seed(self):
        for seed in range(6):
            spawns = sample_spawns(1, 1, seed, LAYOUT['region'], size_range_m=RANGE)
            self.assertEqual(spawns, sample_spawns(1, 1, seed, LAYOUT['region'], size_range_m=RANGE))
            self.assertEqual([s['shape'] for s in spawns], ['cuboid', 'cylinder'])
            for spawn in spawns:
                self.assertTrue(all(0.03 <= n <= 0.064 for n in spawn['size_m'][:-1]))
                self.assertTrue(0.03 <= spawn['size_m'][-1] <= 0.07)
                self.assertTrue(-math.pi <= spawn['yaw'] <= math.pi)
                for axis, key in enumerate(('x','y')):
                    self.assertGreaterEqual(spawn['xy'][axis]-spawn['half_xy'][axis], LAYOUT['region'][key][0])
                    self.assertLessEqual(spawn['xy'][axis]+spawn['half_xy'][axis], LAYOUT['region'][key][1])
        for invalid in [(0.0,0.064,0.03,0.07),(0.03,0.07,0.03,0.07),(0.04,0.03,0.03,0.07),(0.03,0.064,0.02,0.07),(0.03,0.064,0.03,float('nan'))]:
            with self.assertRaises(ValueError): sample_spawns(1,0,0,LAYOUT['region'],size_range_m=invalid)

    def test_camera_accepts_size_edges_without_truth_size_input(self):
        client = p.connect(p.DIRECT)
        try:
            for shape, size in [('cuboid',[0.03,0.064,0.03]),('cuboid',[0.064,0.064,0.07]),
                                ('cylinder',[0.03,0.03]),('cylinder',[0.064,0.07])]:
                scene = build_scene([{'shape':shape,'size_m':size,'xy':[0.5,0],'yaw':-0.9}],LAYOUT)
                self.assertTrue(settle_scene(scene)['settled'])
                # 개별 생성 치수가 아닌 공통 허용 범위만 영상 처리에 전달한다.
                results = estimate_many(capture(scene['camera']),scene['empty'],scene['camera'],shape,size_range_m=RANGE)
                self.assertEqual(len(results),1)
                self.assertTrue(results[0]['valid'], results[0])
                actual_xy = p.getBasePositionAndOrientation(scene['objects'][0]['body'])[0][:2]
                self.assertLess(math.dist(results[0]['center_xy_m'],actual_xy),0.002)
        finally:p.disconnect(client)

    def test_maximum_cylinder_at_pixel_edge_positions_keeps_gripper_limit(self):
        client = p.connect(p.DIRECT)
        try:
            for xy, yaw in [([0.4473142154,-0.0162841825],-2.0202928230),
                            ([0.4591001179,-0.0955955138],-2.7674220246),
                            ([0.4792,-0.152],-1.7)]:
                scene = build_scene([{'shape':'cylinder','size_m':[0.064,0.07],'xy':xy,'yaw':yaw}],LAYOUT)
                self.assertTrue(settle_scene(scene)['settled'])
                candidates = estimate_many(capture(scene['camera']),scene['empty'],scene['camera'],
                                           'cylinder',size_range_m=(0.064,0.064,0.07,0.07))
                self.assertTrue(candidates[0]['valid'],candidates[0])
                self.assertLessEqual(candidates[0]['width_m'],0.065)
        finally:p.disconnect(client)

    def test_random_policy_rejects_cut_surface_and_merged_neighbours(self):
        from rgbd_camera import world_points
        client = p.connect(p.DIRECT)
        try:
            scene = build_scene([{'shape':'cuboid','size_m':[0.05,0.04,0.05],
                                  'xy':[0.5,0],'yaw':0}],LAYOUT)
            self.assertTrue(settle_scene(scene)['settled'])
            observation = capture(scene['camera'])
            visible = estimate_many(observation,scene['empty'],scene['camera'],'cuboid',size_range_m=RANGE)
            self.assertTrue(visible[0]['valid'])
            cut = {**observation, 'depth':observation['depth'].copy()}
            points = world_points(observation)
            cut['depth'][visible[0]['mask'] & (points[...,1] < 0)] = scene['empty']['depth'][visible[0]['mask'] & (points[...,1] < 0)]
            rejected = estimate_many(cut,scene['empty'],scene['camera'],'cuboid',size_range_m=RANGE)
            self.assertFalse(any(g['valid'] for g in rejected))
            # 간격 없는 낮은 이웃 물체가 붙은 영역을 하나로 오인하지 않는지 확인한다.
            merged = build_scene([{'shape':'cuboid','size_m':[0.05,0.04,0.07],'xy':[0.5,0],'yaw':0},
                                  {'shape':'cuboid','size_m':[0.03,0.04,0.03],'xy':[0.54,0],'yaw':0}],LAYOUT)
            for _ in range(480):p.stepSimulation()
            rejected = estimate_many(capture(merged['camera']),merged['empty'],merged['camera'],'cuboid',size_range_m=RANGE)
            self.assertFalse(any(g['valid'] for g in rejected),rejected)
        finally:p.disconnect(client)

    def test_higher_occluder_is_deferred_then_reappears(self):
        client = p.connect(p.DIRECT)
        try:
            # 가까운 높은 물체를 제거한 뒤 재촬영하는 대조다. 초기 배치만 고정한다.
            spawns = [{'shape':'cuboid','size_m':[0.05,0.04,0.03],'xy':[0.50,0.03],'yaw':0},
                      {'shape':'cuboid','size_m':[0.05,0.04,0.07],'xy':[0.52,-0.05],'yaw':0}]
            scene = build_scene(spawns,LAYOUT)
            self.assertTrue(settle_scene(scene)['settled'])
            results = estimate_many(capture(scene['camera']),scene['empty'],scene['camera'],'cuboid',size_range_m=RANGE)
            self.assertTrue(any(not r['valid'] for r in results),results)
            self.assertTrue(any(r['valid'] for r in results),results)
            p.removeBody(scene['objects'][1]['body'])
            results = estimate_many(capture(scene['camera']),scene['empty'],scene['camera'],'cuboid',size_range_m=RANGE)
            self.assertEqual(sum(r['valid'] for r in results),1)
        finally:p.disconnect(client)

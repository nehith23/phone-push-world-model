import sys
import unittest
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_baseline import features, wrap, load_data
from extract_tracks import map_xy

class PipelineChecks(unittest.TestCase):
    def test_features_ignore_global_translation_and_rotation(self):
        state=np.array([10.,15.,.3]); front=np.array([5.,14.]); rear=np.array([1.,14.]); action=np.array([.4,.1])
        original=features(state,front,rear,action)
        angle=.7; c,s=np.cos(angle),np.sin(angle); R=np.array([[c,-s],[s,c]]); shift=np.array([3.,-8.])
        transformed=features(np.r_[R@state[:2]+shift,state[2]+angle],R@front+shift,R@rear+shift,R@action)
        np.testing.assert_allclose(original,transformed,atol=1e-12)

    def test_wrap_crossing_pi(self):
        self.assertAlmostEqual(wrap(-np.pi+.1-(np.pi-.1)),.2)

    def test_tracking_corner_mapping(self):
        for path in (ROOT/'results'/'tracking').glob('*_qa.json'):
            data=json.loads(path.read_text())
            result=map_xy(data['corner_pixels'],np.array(data['pixel_to_table_homography']))
            np.testing.assert_allclose(result,[[0,0],[41,0],[41,29],[0,29]],atol=1e-5)

    def test_whole_push_splits_and_no_reset_transitions(self):
        episodes,samples=load_data()
        self.assertEqual(len(episodes),25)
        membership={name:set() for name in ['train','validation','test']}
        for sample in samples:
            ep=episodes[sample['episode']]
            membership[sample['split']].add(sample['episode'])
            self.assertEqual(sample['split'],ep['split'])
            self.assertGreater(sample['dt_s'],.09); self.assertLess(sample['dt_s'],.11)
            self.assertGreaterEqual(sample['time_s'],ep['raw'][0,1])
            self.assertLessEqual(sample['time_s']+sample['dt_s'],ep['raw'][-1,1]+1e-9)
        self.assertFalse(membership['train']&membership['test'])
        self.assertFalse(membership['validation']&membership['test'])
        self.assertFalse(membership['train']&membership['validation'])
        self.assertEqual(len(membership['test']),4)

    def test_physical_replay_evidence(self):
        directions={'left':[1,0],'right':[-1,0],'down':[0,1],'top':[0,-1]}
        for name,d in directions.items():
            path=ROOT/'results'/'simulation'/f'{name}_01_replay.json'
            data=json.loads(path.read_text())
            self.assertGreater(data['physical_tool_contact_steps'],0)
            self.assertEqual(data['robot_body_contact_steps'],0)
            self.assertGreater(np.dot(data['object_displacement_m'],d),.02)
            self.assertLess(np.linalg.norm(data['approach_displacement_m']),.001)
            self.assertGreater(data['object_height_range_m'][0],.016)
            self.assertLess(data['object_height_range_m'][1],.019)

if __name__=='__main__': unittest.main()

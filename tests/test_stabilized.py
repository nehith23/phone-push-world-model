import sys
import csv
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from stabilize_dataset import observed_homography, geometry_reason
from extract_tracks import map_xy
from goal_control import candidates


class StabilizedChecks(unittest.TestCase):
    def test_camera_motion_removed_for_planar_observation(self):
        initial=np.array([[100,500],[900,500],[850,50],[150,50]],np.float32)
        moved=initial+np.array([13,-8],np.float32)
        original_point=np.array([[430.,280.]])
        def detect(points):
            return ([(points[i],200) for i in (0,2)],[(points[i],200) for i in (1,3)])
        y,p=detect(initial); H,_=observed_homography(y,p,initial,40,30)
        y,p=detect(moved); corrected,_=observed_homography(y,p,initial,40,30)
        np.testing.assert_allclose(map_xy(original_point,H),map_xy(original_point+[13,-8],corrected),atol=1e-6)
        self.assertGreater(np.linalg.norm(map_xy(original_point+[13,-8],H)-map_xy(original_point,H)),.5)

    def test_missing_or_ambiguous_corner_is_rejected(self):
        points=np.array([[100,500],[900,500],[850,50],[150,50]],np.float32)
        y=[(points[i],200) for i in (0,2)]; p=[(points[i],200) for i in (1,3)]
        self.assertIsNone(observed_homography(y,p[:1],points,40,30)[0])
        self.assertIsNone(observed_homography(y+[(points[0]+[2,2],150)],p,points,40,30)[0])

    def test_implausible_marker_geometry_rejected(self):
        self.assertEqual(geometry_reason(np.array([[0,0],[3,0],[-4,0],[-8,0]],float)),'accepted')
        self.assertEqual(geometry_reason(np.array([[0,0],[3,0],[-4,0],[-11,0]],float)),'tool_spacing')
        self.assertEqual(geometry_reason(np.array([[0,0],[3,0],[-4,0],[np.nan,0]],float)),'missing_marker')

    def test_no_rejected_frames_or_reset_gaps_in_training(self):
        data=np.load(ROOT/'results'/'stabilized'/'transitions.npz')
        frames={}
        for path in (ROOT/'results'/'stabilized').glob('*_frames.csv'):
            with path.open() as f:
                for row in csv.DictReader(f): frames[(row['episode'],int(row['frame']))]=row
        for i,episode in enumerate(data['episode']):
            start=int(data['start_frame'][i]); end=int(data['end_frame'][i])
            self.assertEqual(end-start,3)
            for frame in range(start,end+1):
                row=frames[(episode,frame)]
                self.assertEqual(row['status'],'accepted')
                self.assertEqual(row['split'],data['split'][i])
            if episode.startswith('v2_'): self.assertEqual(data['split'][i],'train')
        self.assertTrue(np.isfinite(data['features']).all())
        self.assertTrue(np.isfinite(data['targets']).all())

    def test_candidates_use_revised_dimensions(self):
        state=np.array([.5,0.,0.]); half=np.array([.0785,.037])/2
        for action in candidates(state,half):
            offset=action['start_xy']-state[:2]
            self.assertAlmostEqual(abs(offset[action['axis']]),half[action['axis']]+.009)


if __name__=='__main__': unittest.main()

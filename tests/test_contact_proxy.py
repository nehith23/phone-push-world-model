import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_baseline import contact_proxy, load_data
from goal_control import candidates, predict_candidate


class ContactProxyChecks(unittest.TestCase):
    def test_extrapolation_follows_observed_foreshortening(self):
        # A physical 4 cm spacing projects to 2 cm; the 1.1 cm offset
        # must project to 0.55 cm under the local affine assumption.
        front=np.array([3.,5.]); rear=np.array([3.,3.])
        np.testing.assert_allclose(contact_proxy(front,rear,11/40),[3.,5.55])

    def test_same_examples_and_targets_after_geometry_change(self):
        _,original=load_data(); _,corrected=load_data(11/40)
        self.assertEqual(len(original),511)
        self.assertEqual(len(original),len(corrected))
        for a,b in zip(original,corrected):
            self.assertEqual((a['episode'],a['split'],a['time_s']),(b['episode'],b['split'],b['time_s']))
            np.testing.assert_array_equal(a['target'],b['target'])
            np.testing.assert_allclose(b['front'],contact_proxy(a['front'],a['rear'],11/40))
            np.testing.assert_allclose(b['front']-b['rear'],a['front']-a['rear'])

    def test_simulated_contact_uses_leading_surface(self):
        class Recorder:
            tool_front_offset_cm=.8
            def __init__(self): self.points=[]
            def predict(self,state,front,rear,action):
                self.points.append(front.copy())
                return state
        state=np.array([.5,0.,0.])
        for action in candidates(state):
            model=Recorder()
            predict_candidate(state,action,'learned',model)
            expected=action['start_xy']*100+action['direction']*.8
            np.testing.assert_allclose(model.points[0],expected)
            self.assertAlmostEqual(abs((model.points[0]-state[:2]*100)[action['axis']]),[4.,1.75][action['axis']]+.1)


if __name__=='__main__': unittest.main()

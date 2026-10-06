import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from precision_experiment import actions, predict
from goal_control import candidates, predict_candidate, FrozenDynamics


class PrecisionChecks(unittest.TestCase):
    def test_short_push_is_not_rounded_to_zero(self):
        class CopyModel:
            tool_front_offset_cm=.8
            def __init__(self): self.calls=0
            def predict(self,state,front,rear,action):
                self.calls+=1
                return np.r_[state[:2]+action,state[2]]
        state=np.array([.5,0.,0.])
        for length in [.002,.005,.01,.02]:
            model=CopyModel(); a={**candidates(state)[0],'length_m':length}
            expected=state[:2]+a['direction']*length
            actual=predict(state,a,'learned',model)
            np.testing.assert_allclose(actual[:2],expected,atol=1e-12)
            self.assertGreaterEqual(model.calls,2)

    def test_fine_actions_only_added_near_goal(self):
        state=np.array([.5,0.,0.]); half=np.array([.0785,.037])/2
        self.assertEqual(len(actions(state,np.array([.56,0.]),half,'fine')),24)
        near=actions(state,np.array([.51,0.]),half,'fine')
        self.assertEqual(len(near),48)
        self.assertEqual(set(a['length_m'] for a in near),{.002,.005,.01,.02})

    def test_original_length_predictions_unchanged(self):
        model=FrozenDynamics(ROOT/'results'/'stabilized'/'plus_pilot'/'small_mlp.pt')
        state=np.array([.5,0.,.17])
        for action in candidates(state):
            np.testing.assert_allclose(predict(state,action,'learned',model),predict_candidate(state,action,'learned',model),atol=1e-12)


if __name__=='__main__': unittest.main()

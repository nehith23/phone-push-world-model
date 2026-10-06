import sys
import unittest
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from goal_control import candidates, FrozenDynamics, select_action, make_scenarios, predict_candidate
from train_baseline import wrap

class ControlChecks(unittest.TestCase):
    def test_adapter_preserves_known_displacement_units(self):
        class CopyDisplacement:
            def predict(self, state, front, rear, action):
                return np.r_[state[:2] + action, state[2]]
        state = np.array([.43, -.02, .71])
        for action in candidates(state):
            prediction = predict_candidate(state, action, 'learned', CopyDisplacement())
            expected = state[:2] + action['direction'] * action['length_m']
            np.testing.assert_allclose(prediction[:2], expected, atol=1e-12)

    def test_full_model_adapter_under_rigid_transform(self):
        model = FrozenDynamics()
        state = np.array([.5, 0., .2])
        angle = .73
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shift = np.array([.07, -.08])
        moved = np.r_[R @ state[:2] + shift, state[2] + angle]
        for a, b in zip(candidates(state), candidates(moved)):
            original = predict_candidate(state, a, 'learned', model)
            transformed = predict_candidate(moved, b, 'learned', model)
            np.testing.assert_allclose(transformed[:2], R @ original[:2] + shift, atol=1e-7)
            self.assertLess(abs(wrap(transformed[2] - original[2] - angle)), 1e-6)

    def test_candidate_contacts_are_outside_object(self):
        state=np.array([.5,0.,0.])
        actions=candidates(state)
        self.assertEqual(len(actions),24)
        for action in actions:
            r=action['start_xy']-state[:2]
            half=[.04,.0175][action['axis']]
            self.assertAlmostEqual(abs(r[action['axis']]),half+.009)
            self.assertLess(np.dot(r,action['direction']),0)

    def test_learned_selector_uses_frozen_model(self):
        model=FrozenDynamics()
        calls=[]
        original=model.predict
        def traced(*args):
            calls.append(1)
            return original(*args)
        model.predict=traced
        choice=select_action(np.array([.5,0.,0.]),np.array([.56,0.]),'learned',model)
        self.assertGreater(len(calls),0)
        self.assertTrue(np.isfinite(choice[3]).all())

    def test_evaluation_has_paired_identical_conditions(self):
        protocol=make_scenarios(); model=FrozenDynamics()
        for scenario in protocol['evaluation']:
            pair=[]
            for policy in ['geometric','learned']:
                record=json.loads((ROOT/'results'/'goal_control'/f'{scenario["id"]}_{policy}.json').read_text())
                pair.append(record)
                self.assertEqual(record['scenario'],scenario)
                self.assertEqual(record['checkpoint_sha256'],model.sha256)
                self.assertEqual(record['success'],record['final_error_m']<=protocol['success_distance_m'])
                self.assertLessEqual(record['pushes'],protocol['max_pushes'])
                self.assertEqual(record['robot_body_contact_steps'],0)
                self.assertGreater(record['tool_contact_steps'],0)
            self.assertEqual(pair[0]['observation_assumption'],pair[1]['observation_assumption'])

if __name__=='__main__': unittest.main()

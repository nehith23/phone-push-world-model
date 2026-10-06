"""Render the frozen controller unchanged: first case and every failed case.

Only scene rendering is enabled. Every replayed state and selected action must
match the saved evaluation before any demo is published.
"""
import hashlib
import json
from unittest.mock import patch
import cv2
import numpy as np
import pybullet as p
import precision_experiment as precision
from goal_control import ROOT, FrozenDynamics, observe
from replay_panda import PandaScene

OUT = ROOT / 'results' / 'precision' / 'demo'
FRESH = ROOT / 'results' / 'precision' / 'fresh'


def replay(scenario, policy, protocol, model):
    saved = json.loads((FRESH / f'{scenario["id"]}_{policy}.json').read_text())
    goal = np.array(scenario['goal_xy'])
    scenes = []

    class RenderScene(PandaScene):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, render=True, **kwargs)
            self.push_index = 0
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=.005, length=.001,
                                         rgbaColor=[.05, .65, .2, .8])
            p.createMultiBody(0, -1, visual, [*goal, .0005])
            scenes.append(self)

        def step_target(self, target, capture=True):
            old_count = len(self.frames)
            super().step_target(target, capture)
            if len(self.frames) > old_count and hasattr(self, 'box'):
                image = self.frames[-1]
                distance = np.linalg.norm(observe(self)[:2] - goal) * 1000
                cv2.rectangle(image, (0, 0), (640, 67), (245, 245, 245), -1)
                cv2.putText(image, f'{policy.upper()} | push {self.push_index}/12', (14, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, .64, (25, 25, 25), 2, cv2.LINE_AA)
                cv2.putText(image, f'Current distance {distance:.1f} mm | exact simulator pose', (14, 53),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (45, 45, 45), 1, cv2.LINE_AA)

    original_execute = precision.execute

    def execute(scene, action):
        scene.push_index += 1
        original_execute(scene, action)

    with patch.object(precision, 'PandaScene', RenderScene), patch.object(precision, 'execute', execute):
        actual = precision.run(scenario, policy, 'fine', protocol, model)
    np.testing.assert_allclose(actual['states'], saved['states'], atol=1e-12, rtol=0)
    assert [e['candidate_index'] for e in actual['events']] == [e['candidate_index'] for e in saved['events']]
    assert actual['success'] == saved['success']
    return scenes[0].frames, actual


def card(lines, frames=25):
    image = np.full((580, 1280, 3), 246, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (60, 160 + i * 55), cv2.FONT_HERSHEY_SIMPLEX,
                    .83 if i == 0 else .65, (25, 25, 25), 2 if i == 0 else 1, cv2.LINE_AA)
    return [image] * frames


def main():
    protocol = json.loads((ROOT / 'data' / 'precision_frozen_protocol.json').read_text())
    for name, digest in protocol['code_sha256'].items():
        assert hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() == digest
    model = FrozenDynamics(ROOT / 'results' / 'stabilized' / 'plus_pilot' / 'small_mlp.pt')
    assert model.sha256 == protocol['checkpoint_sha256']
    cases = [protocol['evaluation'][0]]
    for scenario in protocol['evaluation']:
        result = json.loads((FRESH / f'{scenario["id"]}_learned.json').read_text())
        if not result['success'] and scenario not in cases:
            cases.append(scenario)
    OUT.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT / 'precision_demo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 580))
    assert writer.isOpened()
    for image in card(['Phone-trained world model controls a simulated Panda',
                       'Frozen 5 mm evaluation: learned 23/24 | geometric 24/24',
                       'Shown: first case by index, followed by every learned failure.',
                       'Exact simulator poses; green disk is the 5 mm goal region.']):
        writer.write(image)
    checks = []
    for scenario in cases:
        panels = [replay(scenario, policy, protocol, model) for policy in ['geometric', 'learned']]
        for frame_index in range(max(len(frames) for frames, result in panels)):
            images = []
            for frames, result in panels:
                panel = np.full((580, 640, 3), 246, np.uint8)
                panel[:480] = frames[min(frame_index, len(frames) - 1)]
                for j, line in enumerate([f'{scenario["id"]} | 5 mm tolerance | 12-push budget',
                    f'Final: {result["final_error_m"]*1000:.2f} mm | {result["pushes"]} pushes | '+('REACHED' if result['success'] else 'MISSED'),
                    'Playback 1x; completed run holds its final frame.']):
                    cv2.putText(panel, line, (12, 506 + j * 27), cv2.FONT_HERSHEY_SIMPLEX, .48, (35, 35, 35), 1, cv2.LINE_AA)
                images.append(panel)
            combined = np.hstack(images)
            writer.write(combined)
        cv2.imwrite(str(OUT / f'{scenario["id"]}_final.jpg'), combined)
        for _ in range(15): writer.write(combined)
        checks.append({'case': scenario['id'], 'states_match_saved_atol': 1e-12,
                       'candidate_indices_match': True,
                       'outcomes': {r['policy']: {'error_mm': r['final_error_m']*1000, 'success': r['success']} for _, r in panels}})
        print(f'{scenario["id"]}: both replays match saved states and selected actions', flush=True)
    for image in card(['All 24 trials remain in the report',
                       'Learned: 5.60 mm mean final error, including the 56.37 mm failure.',
                       'Geometric: 2.40 mm mean final error and 24/24 reached.',
                       'One object and environment family; real-robot transfer is untested.']):
        writer.write(image)
    writer.release()
    cap = cv2.VideoCapture(str(OUT / 'precision_demo.mp4'))
    decoded = 0
    while cap.read()[0]: decoded += 1
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    assert decoded > 50
    (OUT / 'verification.json').write_text(json.dumps({'selection': 'First scenario by index and all learned failures; not representative random sampling.',
        'checks': checks, 'decoded_frames': decoded, 'fps': fps, 'duration_s': decoded/fps,
        'checkpoint_sha256': model.sha256, 'scope': 'Rendered replay of saved evaluation, not new trials.'}, indent=2))
    print(f'Demo verified: {decoded} frames, {decoded/fps:.1f} seconds')


if __name__ == '__main__': main()

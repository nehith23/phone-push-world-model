"""Post-evaluation audit; never fits weights or changes original benchmark files.

Training support distances are descriptive, not calibrated uncertainties.
Canonical interventions execute each candidate in a fresh identical scene.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from goal_control import FrozenDynamics, candidates, predict_candidate, observe, execute
from replay_panda import PandaScene
from train_baseline import load_data, body, world, features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'diagnosis'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = FrozenDynamics()
    _, samples = load_data()
    train = [s for s in samples if s['split'] == 'train']
    z = np.stack([(s['features'] - model.mean) / model.scale for s in train])
    reference = np.array([.5, 0., 0.])
    checks = {}
    # Verify the complete model adapter, including metre/cm conversion.
    angle = .73
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([.07, -.08])
    moved = np.r_[R @ reference[:2] + shift, angle]
    errors = []
    for a, b in zip(candidates(reference), candidates(moved)):
        pa = predict_candidate(reference, a, 'learned', model)
        pb = predict_candidate(moved, b, 'learned', model)
        errors.append(np.linalg.norm(pb[:2] - (R @ pa[:2] + shift)))
    checks['max_rigid_transform_position_error_m'] = float(max(errors))
    assert max(errors) < 1e-7
    # A recording's front marker and a simulated sphere centre are distinct
    # measurements. Quantify their relative coordinates without calling them
    # calibrated physical contact points.
    support = {}
    for direction in ['left', 'right', 'down', 'top']:
        moving = [s for s in train if s['episode'].startswith(direction)
                  and np.linalg.norm(s['target'][:2]) > .2]
        support[direction] = {'moving_train_samples': len(moving)}
        for label, arr in [
            ('relative_marker_cm', [body(s['front'] - s['state'][:2], s['state'][2]) for s in moving]),
            ('tool_action_cm', [body(s['action'], s['state'][2]) for s in moving]),
            ('object_delta_cm_rad', [s['target'] for s in moving]),
        ]:
            support[direction][label + '_p10_median_p90'] = np.percentile(arr, [10, 50, 90], axis=0).tolist()
    zero = []
    for s in train:
        prediction = model.predict(s['state'], s['front'], s['rear'], np.zeros(2))
        zero.append(np.linalg.norm(prediction[:2] - s['state'][:2]))
    checks['zero_action_predicted_translation_cm_p50_p95_max'] = np.percentile(zero, [50, 95, 100]).tolist()
    cases = []
    for i in range(1, 13):
        record = json.loads((ROOT / 'results' / 'goal_control' / f'eval_{i:02}_learned.json').read_text())
        goal = np.array(record['scenario']['goal_xy'])
        for event in record['events']:
            before = np.array(event['observed_before'])
            pred = np.array(event['predicted_state'])
            after = np.array(event['observed_after'])
            old = np.linalg.norm(before[:2] - goal)
            cases.append({'case': record['scenario']['id'], 'push': event['push'],
                          'predicted_progress_cm': float(100 * (old - np.linalg.norm(pred[:2] - goal))),
                          'actual_progress_cm': float(100 * (old - np.linalg.norm(after[:2] - goal))),
                          'prediction_error_cm': event['prediction_position_error_m'] * 100})
    # Controlled intervention: no target selection, all 24 candidates once.
    interventions = []
    for index in range(24):
        scene = PandaScene(reference[:2], reference[2], .4, False)
        before = observe(scene)
        action = candidates(before)[index]
        prediction = predict_candidate(before, action, 'learned', model)
        f = features(np.r_[before[:2] * 100, before[2]], action['start_xy'] * 100,
                     action['start_xy'] * 100 - action['direction'] * 4, action['direction'] * .5)
        distance = np.linalg.norm(z - (f - model.mean) / model.scale, axis=1)
        nearest = train[int(np.argmin(distance))]
        execute(scene, action)
        after = observe(scene)
        interventions.append({'candidate_index': index, 'axis': action['axis'], 'sign': action['sign'],
                              'offset_fraction': action['offset_fraction'], 'length_cm': action['length_m'] * 100,
                              'predicted_delta_cm': ((prediction[:2] - before[:2]) * 100).tolist(),
                              'actual_delta_cm': ((after[:2] - before[:2]) * 100).tolist(),
                              'position_error_cm': float(np.linalg.norm(prediction[:2] - after[:2]) * 100),
                              'nearest_training_feature_distance': float(min(distance)),
                              'nearest_training_episode': nearest['episode'],
                              'nearest_training_time_s': nearest['time_s'],
                              'tool_contact_steps': int(scene.contact_steps),
                              'robot_body_contact_steps': int(scene.finger_contact_steps)})
        scene.close()
        print(f'candidate {index:02}: error {interventions[-1]["position_error_cm"]:.2f} cm', flush=True)
    # Train leave-one-out distances are context only (temporally correlated samples).
    distances = np.linalg.norm(z[:, None] - z[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    checks['train_nearest_neighbor_distance_p50_p95'] = np.percentile(distances.min(1), [50, 95]).tolist()
    misleading = sum(c['predicted_progress_cm'] > 0 and c['actual_progress_cm'] < 0 for c in cases)
    data = {'status': 'Exploratory diagnosis after the original evaluation; not a new held-out benchmark.',
            'checkpoint_sha256': model.sha256, 'checks': checks, 'training_support': support,
            'evaluation_push_count': len(cases), 'predicted_improvement_actual_regression_count': misleading,
            'evaluation_events': cases, 'canonical_interventions': interventions}
    (OUT / 'audit.json').write_text(json.dumps(data, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    labels = ['left', 'right', 'down', 'top']
    colors = ['#2563eb', '#d97706', '#0f766e', '#9333ea']
    for direction, color in zip(labels, colors):
        points = np.array([body(s['front'] - s['state'][:2], s['state'][2]) for s in train
                           if s['episode'].startswith(direction) and np.linalg.norm(s['target'][:2]) > .2])
        axes[0].scatter(*points.T, s=10, alpha=.55, color=color, label=direction)
    starts = np.array([(a['start_xy'] - reference[:2]) * 100 for a in candidates(reference)])
    axes[0].scatter(*starts.T, marker='x', color='black', s=50, label='simulator starts')
    axes[0].set(xlabel='Object-relative x (projected cm / sim cm)', ylabel='Object-relative y', title='Recorded marker vs simulated tool centre')
    axes[0].legend(fontsize=7)
    for sign, color in [(-1, '#d97706'), (1, '#0f766e')]:
        for axis in [0, 1]:
            row = next(r for r in interventions if r['axis'] == axis and r['sign'] == sign
                       and r['offset_fraction'] == 0 and r['length_cm'] == 2)
            for label, style in [('predicted', '--'), ('actual', '-')]:
                x, y = row[label + '_delta_cm']
                axes[1].plot([0, x], [0, y], linestyle=style, color=color, marker='o',
                             label=label if axis == 0 and sign == -1 else None)
    axes[1].set(xlabel='x displacement (cm)', ylabel='y displacement (cm)', title='2 cm central pushes: prediction vs physics')
    axes[1].legend()
    axes[2].scatter([c['predicted_progress_cm'] for c in cases], [c['actual_progress_cm'] for c in cases], color='#0f766e', s=20)
    axes[2].axhline(0, color='gray'); axes[2].axvline(0, color='gray')
    axes[2].set(xlabel='Predicted improvement (cm)', ylabel='Actual improvement (cm)', title=f'{misleading}/{len(cases)} pushes: predicted gain, actual loss')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[1].set_aspect('equal', adjustable='datalim')
    fig.tight_layout(); fig.savefig(OUT / 'diagnosis.png', dpi=160); plt.close(fig)
    selected = next(r for r in interventions if r['candidate_index'] == 5)
    lines = ['# Why the frozen phone model misses simulation targets', '',
             'This audit was performed after viewing the original 12-case evaluation. It is diagnostic development work. The original model weights, control settings and benchmark results remain unchanged.', '',
             '## Findings', '',
             f'1. The full prediction adapter passes a rigid translation/rotation consistency check (maximum position discrepancy {max(errors):.2g} m). Existing tests also verify table corner mapping. These checks do not establish physical camera calibration.',
             f'2. Across the original evaluation, {misleading} of {len(cases)} executed pushes were predicted to reduce goal distance but actually increased it.',
             f'3. In a fresh canonical scene, candidate 5 (a 2 cm leftward push with lateral offset) predicts displacement {np.round(selected["predicted_delta_cm"], 2).tolist()} cm; physics gives {np.round(selected["actual_delta_cm"], 2).tolist()} cm. Incorrect lateral predictions can therefore cause bad action selection.',
             '4. The recorded front sticker and simulated tool sphere centre have different meanings. Roof markers are elevated, pusher height varies, and marker-to-contact offset is unknown. Directly using those proxy coordinates as physical contact geometry is an uncalibrated transfer assumption.',
             '5. Training contact configurations and velocities cover a narrow, direction-dependent range. The controller queries different positions and repeated 0.5 cm actions. The support plot shows this mismatch; nearest-neighbour distances in audit.json are descriptive, not calibrated confidence scores.',
             f'6. With zero commanded movement at training configurations, the network predicts a median {np.median(zero):.3f} cm translation. It has no enforced zero-action constraint. Because velocity is not in its state, this probe alone cannot distinguish a learned bias from residual motion.', '',
             '## Controlled simulation checks', '',
             'All 24 candidate pushes were executed independently from the same object pose and friction. This isolates individual action prediction from repeated controller decisions. Full outcomes, training-neighbour matches and contact counts are in audit.json.', '',
             f'Robot-body contact occurred in {sum(r["robot_body_contact_steps"] > 0 for r in interventions)}/24 interventions; the attached tool contacted the object in {sum(r["tool_contact_steps"] > 0 for r in interventions)}/24.', '',
             '![Diagnostic plots](diagnosis.png)', '',
             '## What this supports', '',
             'The observed failure is inaccurate action-conditioned prediction under the simulation inputs. A simple global axis reversal is not supported by the coordinate checks. Marker/contact geometry mismatch and sparse configuration coverage are plausible contributors, but this audit does not isolate their individual causal effects.', '',
             'The next change should make the observation/action interface consistent and constrain unsupported predictions. A measured marker-to-contact offset and a flat or calibrated pusher would support better data collection. Any contact-centred representation, geometric fallback, or additional training must be reported explicitly. Preserve the original benchmark and use a fresh final evaluation after development.', '',
             'Reproduce with `python scripts/diagnose_transfer.py`. This script does not train a model or overwrite the original evaluation.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'checks': checks, 'misleading_pushes': misleading, 'pushes': len(cases)}, indent=2))


if __name__ == '__main__':
    main()

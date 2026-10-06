"""Describe the saved failed trial without changing or tuning the controller."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'results'/'precision'/'fresh'
    learned=json.loads((folder/'precision_06_learned.json').read_text())
    geometric=json.loads((folder/'precision_06_geometric.json').read_text())
    goal=np.array(learned['scenario']['goal_xy'])
    rows=[]; bad=0; prediction_errors=[]
    for event in learned['events']:
        before=np.array(event['observed_before'])[:2]
        after=np.array(event['observed_after'])[:2]
        predicted=np.array(event['predicted_state'])[:2]
        d0=np.linalg.norm(before-goal)*1000
        dp=np.linalg.norm(predicted-goal)*1000
        da=np.linalg.norm(after-goal)*1000
        pe=np.linalg.norm(predicted-after)*1000
        bad+=int(dp<d0 and da>d0)
        prediction_errors.append(pe)
        rows.append(f'| {event["push"]} | {event["length_m"]*1000:.0f} | {d0:.2f} | {dp:.2f} | {da:.2f} | {pe:.2f} |')
    states=np.array(learned['states'])
    geo_states=np.array(geometric['states'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].plot(*(states[:,:2]*1000).T,'o-',color='#0f766e',label='Learned observations')
    axes[0].plot(*(geo_states[:,:2]*1000).T,'o-',color='#64748b',label='Geometric observations')
    axes[0].scatter(*(goal*1000),marker='x',color='black',label='Goal')
    axes[0].set(xlabel='Simulator x (mm)',ylabel='Simulator y (mm)',title='Failed case precision_06')
    axes[0].axis('equal'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    axes[1].plot(range(13),np.linalg.norm(states[:,:2]-goal,axis=1)*1000,'o-',color='#0f766e',label='Observed')
    axes[1].plot(range(1,13),[np.linalg.norm(np.array(e['predicted_state'])[:2]-goal)*1000 for e in learned['events']],':o',color='#b45309',label='Predicted after chosen push')
    axes[1].axhline(5,color='black',linestyle='--',label='5 mm success threshold')
    axes[1].set(xlabel='Push number',ylabel='Distance to goal (mm)',title='Predicted improvement does not reliably occur')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'failure_analysis.png',dpi=150); plt.close(fig)
    report=f'''# Why the remaining trial failed

This is a post-evaluation analysis of the saved `precision_06` trial. It did not change the network, controller or evaluation outcomes.

## What the trace shows

The learned controller starts {np.linalg.norm(states[0,:2]-goal)*1000:.2f} mm from the goal and finishes {learned['final_error_m']*1000:.2f} mm away after exhausting all 12 pushes. All selected strokes are 20 mm; it never enters the 20 mm fine-action region. The minimum observed goal distance is {np.linalg.norm(states[:,:2]-goal,axis=1).min()*1000:.2f} mm. The geometric controller reaches {geometric['final_error_m']*1000:.2f} mm in {geometric['pushes']} pushes under the same conditions.

All 12 selected actions are predicted to reduce goal distance, but {bad} actually increase it. Predicted-versus-observed endpoint position differences range from {min(prediction_errors):.2f} to {max(prediction_errors):.2f} mm. The controller repeatedly chooses long pushes with inaccurate lateral motion predictions, spends its budget, and misses the target. This is not a failure of the short 2 mm stroke or a case that just falls outside the 5 mm threshold.

![Observed paths and predicted progress](../results/precision/fresh/failure_analysis.png)

## Evidence for every push

All distances below are simulator measurements in millimetres. Predicted error here means distance of the predicted endpoint from the goal; model mismatch means distance between predicted and observed endpoints.

| Push | Stroke | Before | Predicted goal error | Observed goal error | Model mismatch |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## What is still unknown

The trace establishes prediction mismatch and unsuccessful action selection. It does not isolate whether the mismatch comes mainly from sparse directional/contact coverage, projection bias, motion outside the training distribution, or differences between real and simulated contact. Those are hypotheses, not proven causes. An exact-state geometric controller succeeds here, so inability of this simulator configuration to reach the target is not the explanation.

A future experiment could log all candidate predictions and independently execute matched candidates, then compare errors by contact face, offset and speed. New data could target the poorly covered cases. Any fallback to geometry or uncertainty-based rejection would be a new hybrid controller and would need a newly frozen evaluation; it must not replace this failed outcome retrospectively.
'''
    (ROOT/'docs'/'FAILURE_ANALYSIS.md').write_text(report,encoding='utf-8')
    print(f'Failure analysis: {bad}/12 predicted improvements worsened actual distance; no fine actions used.')


if __name__=='__main__': main()

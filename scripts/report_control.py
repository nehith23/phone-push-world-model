"""Report every fixed evaluation scenario, including failures."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'goal_control'

def main():
    results=json.loads((OUT/'evaluation_summary.json').read_text())
    protocol=results['protocol']; labels=['geometric','learned']; colors=['#64748b','#0f766e']
    fig,axes=plt.subplots(3,4,figsize=(14,10))
    table=['| Scenario | Geometric error (cm) | Learned error (cm) | Learned reached goal? |','|---|---:|---:|---|']
    for ax,scenario in zip(axes.flat,protocol['evaluation']):
        goal=np.array(scenario['goal_xy'])*100; initial=np.array(scenario['initial_xy'])*100
        ax.add_patch(plt.Circle(goal,protocol['success_distance_m']*100,color='#22c55e',alpha=.2))
        ax.scatter(*goal,marker='x',color='#15803d',s=65,label='goal')
        ax.scatter(*initial,marker='s',color='#111827',s=20,label='start')
        episodes={}
        for label,color in zip(labels,colors):
            episode=json.loads((OUT/f'{scenario["id"]}_{label}.json').read_text()); episodes[label]=episode
            path=np.array(episode['states'])[:,:2]*100
            ax.plot(path[:,0],path[:,1],'-o',color=color,markersize=3,label=label)
        ax.set_title(scenario['id']+' | '+('reached' if episodes['learned']['success'] else 'missed'))
        ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.2); ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
        table.append(f'| {scenario["id"]} | {episodes["geometric"]["final_error_m"]*100:.2f} | {episodes["learned"]["final_error_m"]*100:.2f} | {"Yes" if episodes["learned"]["success"] else "No"} |')
    axes[0,0].legend(fontsize=8,loc='best')
    fig.suptitle('Goal reaching: 12 fixed scenarios, identical actions and execution\nGreen disk: 1.5 cm position tolerance; learned model frozen from phone data')
    fig.tight_layout(); fig.savefig(OUT/'all_scenarios.png',dpi=140); plt.close(fig)
    geometric=results['summary']['geometric']; learned=results['summary']['learned']
    text=f'''# Frozen phone-trained model in closed-loop simulation

The learned model now selects actions. At each decision it predicts the outcome of 24 candidate pushes and chooses the one whose predicted endpoint is closest to the goal. Panda executes the push through contact physics, receives a new object pose, and replans. The geometric controller uses the same candidates and execution, predicting simple straight translation.

## Fixed protocol

- 12 scenarios generated with seed 20261005 before evaluation.
- Initial object positions, yaw, target directions and friction vary.
- Success: object centre within 1.5 cm of the target within eight pushes.
- Both controllers receive exact simulator object position and yaw. This is not a vision-controlled robot benchmark.
- The neural checkpoint is unchanged from the phone-video experiment. No simulation transitions train it and evaluation outcomes did not tune its parameters or controller settings.
- The environment is a simplified cuboid on a flat plane. Its physical properties are assumptions.

| Controller | Targets reached | Mean final error | Mean pushes used |
|---|---:|---:|---:|
| Geometric | {geometric['successes']}/12 | {geometric['mean_final_error_cm']:.2f} cm | {geometric['mean_pushes']:.2f} |
| Learned model | {learned['successes']}/12 | {learned['mean_final_error_cm']:.2f} cm | {learned['mean_pushes']:.2f} |

The learned model reaches some goals, but the geometric controller is more reliable on this position-only task. The offline one-step prediction improvement therefore does **not** establish a control advantage. Likely contributors include projection bias, sparse coverage of tool/object configurations, compounded prediction errors and the difference between a hollow cover and the simulated solid box. These explanations are hypotheses, not isolated experimental findings.

## All outcomes

'''+ '\n'.join(table)+'''

![Every evaluation trajectory](all_scenarios.png)

`eval_01` is the first evaluation scenario, chosen for the demo by index. It is not presented as representative of all results. All scenario JSON files include selected candidates, predictions, measured outcomes and the model-checkpoint hash.

## Reproduce

From the repository root:

```powershell
python scripts/goal_control.py --split development
python scripts/goal_control.py --split evaluation
python scripts/goal_control.py --split evaluation --case eval_01 --render
python scripts/report_control.py
```

The source evaluation protocol is `data/control_scenarios.json`. Preserve it when comparing changes. If controller/model choices are informed by these outcomes, describe later results as development results and generate a fresh final evaluation set.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    print('Saved control report and all-scenario plot.')

if __name__=='__main__': main()

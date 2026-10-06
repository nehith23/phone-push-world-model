"""Complete frozen-scenario reporting and optional first-case demonstration."""
import argparse
import json
import hashlib
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import goal_control as control
from goal_control import ROOT, FrozenDynamics

OUT=ROOT/'results'/'frozen_evaluation'


def demo(protocol,model):
    scenario=protocol['evaluation'][0]
    control.OUT=OUT/'demo'; control.OUT.mkdir(parents=True,exist_ok=True)
    policies=['geometric','learned']; records=[]; caps=[]
    for policy in policies:
        original=json.loads((OUT/policy/f'{scenario["id"]}_{policy}.json').read_text())
        result=control.run(scenario,policy,protocol,model,True)
        np.testing.assert_allclose(original['states'],result['states'],atol=1e-12)
        records.append(result)
        cap=cv2.VideoCapture(str(control.OUT/f'{scenario["id"]}_{policy}.mp4'))
        if not cap.isOpened(): raise ValueError('Rendered video is unavailable')
        caps.append(cap)
    count=int(max(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps))
    writer=cv2.VideoWriter(str(OUT/'comparison_demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,520))
    if not writer.isOpened(): raise ValueError('Cannot write demo')
    last=[None,None]; gif=[]
    for index in range(count):
        panels=[]
        for k,(cap,record) in enumerate(zip(caps,records)):
            ok,frame=cap.read()
            if ok: last[k]=frame
            if last[k] is None: raise ValueError('Empty video')
            panel=np.full((520,640,3),245,np.uint8); panel[:480]=last[k]
            label=f'Fresh case 1/24 | final: {"reached" if record["success"] else "missed"}, {record["final_error_m"]*100:.2f} cm'
            cv2.putText(panel,label,(12,506),cv2.FONT_HERSHEY_SIMPLEX,.48,(45,45,45),1,cv2.LINE_AA)
            panels.append(panel)
        combined=np.hstack(panels); writer.write(combined)
        gif.append(Image.fromarray(cv2.cvtColor(cv2.resize(combined,(960,390)),cv2.COLOR_BGR2RGB)))
    writer.release()
    for cap in caps: cap.release()
    gif[0].save(OUT/'comparison_demo.gif',save_all=True,append_images=gif[1:],duration=100,loop=0,optimize=True)
    cv2.imwrite(str(OUT/'comparison_final.jpg'),combined)
    print(f'Demo: {count} frames; rendering replay matches recorded physics states.')


def main(render):
    data=json.loads((OUT/'summary.json').read_text()); protocol=data['protocol']; summary=data['summary']
    model=FrozenDynamics(ROOT/'results'/'stabilized'/'plus_pilot'/'small_mlp.pt')
    assert model.sha256==protocol['checkpoint_sha256']
    for name,digest in protocol['code_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
    fig,axes=plt.subplots(4,6,figsize=(16,11)); rows=[]
    for ax,scenario in zip(axes.flat,protocol['evaluation']):
        goal=np.array(scenario['goal_xy'])*100; pair=[]
        ax.add_patch(plt.Circle(goal,1.5,color='#22c55e',alpha=.2)); ax.scatter(*goal,marker='x',color='#15803d',s=25)
        for policy,color in [('geometric','#64748b'),('learned','#0f766e')]:
            r=json.loads((OUT/policy/f'{scenario["id"]}_{policy}.json').read_text())
            assert r['scenario']==scenario and r['checkpoint_sha256']==model.sha256
            assert r['robot_body_contact_steps']==0 and r['tool_contact_steps']>0
            pair.append(r)
            path=np.array(r['states'])[:,:2]*100
            ax.plot(*path.T,'-o',markersize=2,color=color,label=policy)
        ax.set_title(scenario['id']+(' reached' if pair[1]['success'] else ' missed'),fontsize=9)
        ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.2); ax.tick_params(labelsize=7)
        rows.append(f'| {scenario["id"]} | {pair[0]["final_error_m"]*100:.2f} | {pair[1]["final_error_m"]*100:.2f} | {pair[1]["pushes"]} | {"yes" if pair[1]["success"] else "no"} |')
    axes[0,0].legend(fontsize=7); fig.suptitle('Frozen model: all 24 fresh simulation scenarios\nExact simulator pose; green disk = 1.5 cm goal tolerance')
    fig.tight_layout(); fig.savefig(OUT/'all_scenarios.png',dpi=150); plt.close(fig)
    geo=summary['geometric']; learned=summary['learned']
    report=f'''# Frozen-model evaluation on 24 new simulation scenarios

The cleaned-data-plus-pilot model was frozen after development and then evaluated once on 24 newly generated initial states, goals and friction assignments. The protocol and checkpoint/code hashes were saved before either controller ran. No outcomes from these cases selected the model or tuned thresholds.

| Controller | Targets reached | Mean final error | Mean pushes |
|---|---:|---:|---:|
| Geometric | {geo['successes']}/24 | {geo['mean_final_error_cm']:.2f} cm | {geo['mean_pushes']:.2f} |
| Learned world model | {learned['successes']}/24 | {learned['mean_final_error_cm']:.2f} cm | {learned['mean_pushes']:.2f} |

The frozen learned controller completes {learned['successes']} of these 24 tasks. This establishes its observed performance on this small simulation sample; it does not establish superiority to the geometric controller or a population-wide success rate. Both receive exact simulator object pose. No simulator transitions were used to train the world model.

## Conditions

- Seed 824173; 24 fresh cases generated before execution.
- Object start within 1.5 cm of (0.5, 0) m; yaw within +/-0.25 radians.
- Random goal angle and distance 5.5-7.5 cm; friction 0.25/0.4/0.6, eight cases each.
- Revised object size 78.5 x 37 x 33 mm; assumed mass 0.1 kg; same solid-box approximation.
- Same 24 candidates, execution and eight-push budget; success within 1.5 cm of goal.
- All 48 runs have tool/object contact and zero robot-body/object contact.

These are fresh **simulation scenarios in the same environment family**, not fresh physical recordings, new objects, new surfaces, or vision-based robot control. The physical-video geometry remains approximate. One seed and 24 cases provide limited evidence; do not tune on them and continue calling them unseen.

## Every outcome

| Scenario | Geometric error (cm) | Learned error (cm) | Learned pushes | Learned reached? |
|---|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

![All fresh scenario trajectories](all_scenarios.png)

## Demonstration

The optional side-by-side demo shows `fresh_01`, selected by index rather than outcome. Rendered replay must reproduce the saved physics states exactly within numerical tolerance.

![First fresh case, both controllers](comparison_demo.gif)

## Reproduce

```powershell
python scripts/evaluate_frozen.py
python scripts/report_frozen.py --render-demo
```

The evaluator reuses the saved protocol and refuses changes to the frozen checkpoint or relevant code. Subsequent reruns reproduce this evaluation; they are not additional independent samples. The original and all intermediate development experiments remain in separate result folders.
'''
    (OUT/'REPORT.md').write_text(report)
    if render: demo(protocol,model)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--render-demo',action='store_true')
    main(parser.parse_args().render_demo)

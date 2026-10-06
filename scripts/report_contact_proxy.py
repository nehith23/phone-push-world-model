"""Compare prescribed contact-proxy experiment with preserved original results."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'contact_proxy'


def main():
    original=json.loads((ROOT/'results'/'baseline'/'metrics.json').read_text())
    corrected=json.loads((OUT/'metrics.json').read_text())
    before=json.loads((ROOT/'results'/'goal_control'/'evaluation_summary.json').read_text())
    after=json.loads((OUT/'control'/'evaluation_summary.json').read_text())
    oldhash=hashlib.sha256((ROOT/'results'/'baseline'/'small_mlp.pt').read_bytes()).hexdigest()
    newhash=hashlib.sha256((OUT/'small_mlp.pt').read_bytes()).hexdigest()
    assert original['dataset']==corrected['dataset']
    rows=[]; olderr=[]; newerr=[]; geoerr=[]; allrecords=[]
    for scenario in before['protocol']['evaluation']:
        sid=scenario['id']
        old=json.loads((ROOT/'results'/'goal_control'/f'{sid}_learned.json').read_text())
        new=json.loads((OUT/'control'/f'{sid}_learned.json').read_text())
        oldgeo=json.loads((ROOT/'results'/'goal_control'/f'{sid}_geometric.json').read_text())
        newgeo=json.loads((OUT/'control'/f'{sid}_geometric.json').read_text())
        assert old['checkpoint_sha256']==oldhash
        assert new['checkpoint_sha256']==newhash
        assert old['scenario']==new['scenario']==oldgeo['scenario']==newgeo['scenario']
        np.testing.assert_allclose(oldgeo['states'],newgeo['states'],atol=1e-12)
        for r in [new,newgeo]:
            assert r['robot_body_contact_steps']==0 and r['tool_contact_steps']>0
        allrecords.extend([new,newgeo])
        olderr.append(old['final_error_m']*100); newerr.append(new['final_error_m']*100); geoerr.append(newgeo['final_error_m']*100)
        rows.append(f'| {sid} | {olderr[-1]:.2f} | {newerr[-1]:.2f} | {"reached" if new["success"] else "missed"} |')
    a=original['one_step']['test']['small_mlp']; b=corrected['one_step']['test']['small_mlp']
    old=before['summary']['learned']; new=after['summary']['learned']; geo=after['summary']['geometric']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),gridspec_kw={'width_ratios':[1,2]})
    colors=['#64748b','#0f766e','#d97706']
    axes[0].bar(['Geometric','Original MLP','Contact proxy'],[geo['successes'],old['successes'],new['successes']],color=colors)
    axes[0].set(ylim=(0,13),ylabel='Targets reached out of 12',title='Same scenarios reused for development')
    x=np.arange(12); width=.25
    for i,(label,values,color) in enumerate(zip(['Geometric','Original MLP','Contact proxy'],[geoerr,olderr,newerr],colors)):
        axes[1].bar(x+(i-1)*width,values,width,label=label,color=color)
    axes[1].axhline(1.5,color='#222222',linestyle='--',label='Success tolerance')
    axes[1].set(xticks=x,xticklabels=[f'{i:02}' for i in range(1,13)],xlabel='Scenario',ylabel='Final goal distance (cm)',title='Improvements and regressions across all cases')
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=150); plt.close(fig)
    summary={'status':'Post-evaluation development comparison; no new held-out data.',
             'original_checkpoint_sha256':oldhash,'corrected_checkpoint_sha256':newhash,
             'same_dataset_and_splits':True,'geometric_trajectories_identical':True,
             'control':{'original':old,'contact_proxy':new,'geometric':geo},
             'position_prediction_mae_projected_cm':{'original':a['position_mae_projected_cm'],'contact_proxy':b['position_mae_projected_cm']},
             'rotation_prediction_mae_deg':{'original':a['heading_mae_deg'],'contact_proxy':b['heading_mae_deg']}}
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2))
    report=f'''# The 11 mm contact-offset experiment

**Result: mixed.** Target success rose from {old['successes']}/12 to {new['successes']}/12, but mean final error increased from {old['mean_final_error_cm']:.2f} to {new['mean_final_error_cm']:.2f} cm. This is not evidence of a reliable overall control improvement. Simple geometry still reaches all 12 targets.

## What changed

The user measured 11 mm from the front pink sticker centre to the contacting edge and confirmed that the same edge was used in all four recordings. The marker separation is 40 mm. We compute an estimated contact point:

`contact = front + (11 / 40) * (front - rear)`

This extends the observed marker vector by 27.5%, approximately following its foreshortening. It does not add a fixed 1.1 projected cm. The action is the change in this estimated contact point over each transition, including changes caused by shaft rotation. Tool direction is preserved.

This is **local affine extrapolation**, not exact projective or 3-D calibration. Roof height, caliper height/tilt, the correspondence between the roof-marker midpoint and box centre, and different physical dynamics remain unresolved. In simulation, the corresponding input point is the leading surface of the 8 mm-radius sphere, rather than its centre.

## What stayed fixed

The same 511 examples and targets, train/validation/test memberships, seed, architecture, optimizer, early stopping rule, candidates, robot execution and scenario set were used. Training normalization was recomputed only from the corrected training inputs. No simulation transitions trained the network. Validation selected epoch {corrected['training']['mlp_best_epoch']} for the corrected model. The experiment was specified in `data/contact_proxy_protocol.json` before running it; no offset sweep was performed.

The original checkpoint and results remain intact. The geometric controller's rerun trajectories match its original trajectories to numerical tolerance. Both corrected-run controllers have zero robot-body/object contacts and positive attached-tool contacts in every scenario.

## Results

| Metric | Original model | Contact-proxy model |
|---|---:|---:|
| Position MAE, projected cm | {a['position_mae_projected_cm']:.3f} | {b['position_mae_projected_cm']:.3f} |
| Heading MAE, degrees | {a['heading_mae_deg']:.3f} | {b['heading_mae_deg']:.3f} |
| Moving-only position MAE, projected cm | {a['moving_position_mae_projected_cm']:.3f} | {b['moving_position_mae_projected_cm']:.3f} |
| Targets reached | {old['successes']}/12 | {new['successes']}/12 |
| Mean final goal error, cm | {old['mean_final_error_cm']:.2f} | {new['mean_final_error_cm']:.2f} |
| Mean pushes used | {old['mean_pushes']:.2f} | {new['mean_pushes']:.2f} |

Position prediction improves on the reused recorded test clips, while heading and moving-only translation worsen. Across long recorded-action rollouts, substantial drift persists. Physical position accuracy remains unvalidated.

These recordings and scenarios were already inspected during development. Their labels still say `test` and `evaluation` for reproducibility, but the new results are **development comparisons**, not fresh held-out evidence. One training seed and 12 scenarios are insufficient to establish a robust advantage.

![All scenario comparisons](comparison.png)

| Scenario | Original error (cm) | Contact-proxy error (cm) | Contact-proxy outcome |
|---|---:|---:|---|
'''+ '\n'.join(rows)+'''

## Next step

Retain both experiments. Do not tune the 11 mm measurement to make this benchmark look better. First record a short pilot with a more stable camera/tool setup and varied contact positions. Inspect its projected geometry and tracking before requesting a larger dataset. Reserve a separate recording session for final evaluation after the design is frozen.

## Reproduce

```powershell
python scripts/train_baseline.py --representation contact_proxy
python scripts/goal_control.py --experiment contact_proxy --split evaluation
python scripts/report_contact_proxy.py
python -m unittest discover -s tests -v
```
'''
    (OUT/'REPORT.md').write_text(report)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()

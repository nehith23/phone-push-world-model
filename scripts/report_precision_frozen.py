"""Report the entire fresh precision sample without changing any settings."""
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from goal_control import ROOT

OUT=ROOT/'results'/'precision'/'fresh'


def main():
    data=json.loads((OUT/'summary.json').read_text()); protocol=data['protocol']; summary=data['summary']
    for name,digest in protocol['code_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/'results'/'stabilized'/'plus_pilot'/'small_mlp.pt').read_bytes()).hexdigest()==protocol['checkpoint_sha256']
    rows=[]; errors={'geometric':[],'learned':[]}
    for scenario in protocol['evaluation']:
        pair=[]
        for policy in errors:
            r=json.loads((OUT/f'{scenario["id"]}_{policy}.json').read_text())
            assert r['scenario']==scenario and r['mode']=='fine'
            assert r['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert r['success']==(r['final_error_m']<=.005)
            assert r['robot_body_contact_steps']==0 and r['tool_contact_steps']>0
            errors[policy].append(r['final_error_m']*1000); pair.append(r)
        rows.append(f'| {scenario["id"]} | {pair[0]["final_error_m"]*1000:.2f} | {pair[1]["final_error_m"]*1000:.2f} | {pair[1]["pushes"]} | {"yes" if pair[1]["success"] else "no"} |')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].bar(['Geometric','Learned'],[summary[p]['successes'] for p in errors],color=['#64748b','#0f766e'])
    axes[0].set(ylim=(0,25),ylabel='Targets reached within 5 mm / 24',title='Fixed precision controller: fresh scenarios')
    for policy,color in [('geometric','#64748b'),('learned','#0f766e')]:
        axes[1].plot(range(1,25),errors[policy],'-o',label=policy,color=color,markersize=3)
    axes[1].axhline(5,color='black',linestyle='--',label='5 mm tolerance')
    axes[1].set(xlabel='Fresh precision scenario',ylabel='Final error (mm)',title='Every result, including failures'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)
    geo=summary['geometric']; learned=summary['learned']
    report=f'''# Fresh precision evaluation: 5 mm target tolerance

After the prescribed 12-case development comparison, the fine-action controller settings were frozen. Twenty-four new scenarios were generated and saved before either controller ran (seed 637251). No model retraining, threshold tuning, or action-length sweep followed these outcomes.

| Controller | Within 5 mm | Mean final error (mm) | Median error (mm) | Mean pushes |
|---|---:|---:|---:|---:|
| Geometric + fine actions | {geo['successes']}/24 | {geo['mean_final_error_mm']:.2f} | {geo['median_final_error_mm']:.2f} | {geo['mean_pushes']:.2f} |
| Learned + fine actions | {learned['successes']}/24 | {learned['mean_final_error_mm']:.2f} | {learned['median_final_error_mm']:.2f} | {learned['mean_pushes']:.2f} |

Both use the same 5 mm tolerance, 12-push budget, exact simulator pose and revised box. Both retain 10/20 mm strokes and can choose 2/5 mm strokes within 20 mm of the target. The world-model weights are identical to the earlier 22/24, 1.5 cm-tolerance experiment. All failures remain in the averages.

![All precision results](comparison.png)

## Every outcome

| Scenario | Geometric error (mm) | Learned error (mm) | Learned pushes | Learned reached? |
|---|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

## What this means

These are simulator errors with exact observations, not real-world placement accuracy. The phone-video measurement geometry remains approximate. The cases are new but drawn from the same environment family, one object and the same parameter ranges. One small sample does not establish a general superiority claim.

The earlier 1.5 cm result uses a different stopping rule, push budget and scenario set. Do not present the difference in average errors as an isolated effect of shorter actions. The paired coarse/fine comparison in `../REPORT.md` is the relevant development ablation.

All 48 executions have attached-tool contact and zero robot-body/object contacts. Protocol, code and checkpoint hashes are verified before reporting. The original 1.5 cm evaluation is preserved. Future tuning on these results would make them development evidence for the modified controller.

## Reproduce

```powershell
python scripts/evaluate_precision_frozen.py
python scripts/report_precision_frozen.py
```

Use the saved `data/precision_frozen_protocol.json`; rerunning it reproduces the same cases, not another independent sample.
'''
    (OUT/'REPORT.md').write_text(report)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()

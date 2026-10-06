"""Precision experiment results, preserving the scope of the original claims."""
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from goal_control import ROOT

OUT=ROOT/'results'/'precision'


def main():
    data=json.loads((OUT/'summary.json').read_text()); protocol=data['protocol']; summary=data['summary']
    checkpoint=ROOT/'results'/'stabilized'/'plus_pilot'/'small_mlp.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==protocol['checkpoint_sha256']
    for name,digest in protocol['code_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
    # Confirm that relevant code/weights from the earlier frozen run remain intact.
    frozen=json.loads((ROOT/'data'/'frozen_evaluation_protocol.json').read_text())
    assert protocol['checkpoint_sha256']==frozen['checkpoint_sha256']
    for name,digest in frozen['code_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
    conditions=['geometric_coarse','geometric_fine','learned_coarse','learned_fine']
    names=['Geometry: coarse','Geometry: fine','Learned: coarse','Learned: fine']
    colors=['#94a3b8','#475569','#5eead4','#0f766e']
    rows=[]; bycase=[]
    for scenario in protocol['evaluation']:
        values=[]
        for condition in conditions:
            policy,mode=condition.split('_')
            r=json.loads((OUT/f'{scenario["id"]}_{policy}_{mode}.json').read_text())
            assert r['scenario']==scenario and r['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert r['success']==(r['final_error_m']<=.005)
            assert r['robot_body_contact_steps']==0 and r['tool_contact_steps']>0
            values.append(r['final_error_m']*1000)
        bycase.append(values)
        rows.append('| '+scenario['id']+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    table=['| Controller | Reached within 5 mm | Mean final error (mm) | Median error (mm) | Mean pushes | Fine pushes used |','|---|---:|---:|---:|---:|---:|']
    for condition,name in zip(conditions,names):
        r=summary[condition]
        table.append(f'| {name} | {r["successes"]}/12 | {r["mean_final_error_mm"]:.2f} | {r["median_final_error_mm"]:.2f} | {r["mean_pushes"]:.2f} | {r["total_fine_pushes"]} |')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].bar(range(4),[summary[k]['successes'] for k in conditions],color=colors)
    axes[0].set(xticks=range(4),xticklabels=names,ylim=(0,13),ylabel='Reached within 5 mm / 12',title='Identical 12-push budget')
    axes[0].tick_params(axis='x',rotation=20)
    x=np.arange(12); values=np.array(bycase)
    for i,(name,color) in enumerate(zip(names,colors)):
        axes[1].plot(x,values[:,i],'-o',label=name,color=color,markersize=3)
    axes[1].axhline(5,color='black',linestyle='--',label='5 mm tolerance')
    axes[1].set(xticks=x,xticklabels=[f'{i:02}' for i in range(1,13)],xlabel='Reused development scenario',ylabel='Final error (mm)',title='All outcomes, including failures')
    axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)
    geom=summary['geometric_fine']; learned=summary['learned_fine']
    report='''# Precision follow-up: 5 mm targets and shorter pushes

The original 1.5 cm goal tolerance intentionally allowed approximate placement. To investigate the user's precision concern, we prescribed one bounded experiment using the 12 previously inspected development scenarios. This experiment does not replace the earlier 24-case frozen evaluation.

## Fair comparison

All four conditions share a **5 mm target tolerance and 12-push budget**, the revised box dimensions, exact simulator pose, environment and execution code. The coarse set contains 10 and 20 mm strokes. The fine condition retains those and adds 2 and 5 mm strokes when within 20 mm of the target. Both geometric and learned controllers get the same candidate sets.

The trained model remains frozen. Short strokes use at least two nominal 0.1 s prediction steps, matching the existing execution's minimum 0.2 s push duration. A regression test confirms original-length predictions are unchanged and a 2 mm action is not rounded to zero prediction steps. The physical execution still includes the existing approach and settling phases; its behavior is not assumed to be an exact displacement actuator.

Thresholds and action lengths were fixed before execution; no parameter sweep followed the results. Protocol and code/model hashes are recorded in `data/precision_protocol.json`.

## Results

'''+ '\n'.join(table)+f'''

The fine-action geometric controller reaches {geom['successes']}/12 targets; the fine-action learned controller reaches {learned['successes']}/12. Both mean and median errors are reported because a few failures can dominate the mean. Adding finer actions alone does not guarantee accurate predictions or successful choices by the learned model.

![Precision comparison](comparison.png)

| Scenario | Geometry coarse (mm) | Geometry fine (mm) | Learned coarse (mm) | Learned fine (mm) |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## How to interpret this

Compare coarse versus fine within this experiment to assess the added finishing actions. The earlier 1.5 cm experiment had a different budget and stopping rule, so its average errors are not directly comparable as a single-factor improvement.

All 48 runs used physical attached-tool contact and zero robot-body/object contacts. These are **simulator millimetres** with exact object observations. The phone measurements still have unresolved height/projection bias; this does not establish real-world millimetre accuracy.

The 12 cases are reused development scenarios. The original frozen 22/24 result remains valid as a record of its original protocol; this precision follow-up must not be advertised as an unseen test. Preserve the working baseline and report this limitation instead of repeatedly tuning the same cases until they pass.

## Reproduce

```powershell
python scripts/precision_experiment.py
python scripts/report_precision.py
python -m unittest discover -s tests -p test_precision.py -v
```

The evaluator checks the prescribed code/checkpoint hashes. The report also verifies that the earlier frozen evaluation's relevant code and model remain unchanged.
'''
    (OUT/'REPORT.md').write_text(report)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()

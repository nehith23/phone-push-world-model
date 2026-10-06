"""Report paired controls and filtering attrition, including negative outcomes."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'stabilized'


def main():
    extraction=json.loads((OUT/'extraction.json').read_text())
    variants=['geometric','original','offset_11mm','old_only','plus_pilot']
    names=['Geometric','Original MLP','11 mm MLP','Cleaned old data','Cleaned + pilot']
    colors=['#64748b','#a1a1aa','#d97706','#2563eb','#0f766e']
    results={v:json.loads((OUT/'control'/v/'summary.json').read_text()) for v in variants}
    training={v:json.loads((OUT/v/'metrics.json').read_text()) for v in ['old_only','plus_pilot']}
    protocol=results['geometric']['protocol']; episodes=[]; contacts=[]
    checkpoint_paths={'geometric':ROOT/'results'/'baseline'/'small_mlp.pt','original':ROOT/'results'/'baseline'/'small_mlp.pt',
                      'offset_11mm':ROOT/'results'/'contact_proxy'/'small_mlp.pt','old_only':OUT/'old_only'/'small_mlp.pt','plus_pilot':OUT/'plus_pilot'/'small_mlp.pt'}
    for v in variants:
        assert results[v]['protocol']==protocol
        digest=hashlib.sha256(checkpoint_paths[v].read_bytes()).hexdigest()
        assert results[v]['summary']['checkpoint_sha256']==digest
        policy='geometric' if v=='geometric' else 'learned'
        records=[]
        for scenario in protocol['evaluation']:
            r=json.loads((OUT/'control'/v/f'{scenario["id"]}_{policy}.json').read_text())
            assert r['scenario']==scenario and r['checkpoint_sha256']==digest
            assert r['robot_body_contact_steps']==0 and r['tool_contact_steps']>0
            assert r['success']==(r['final_error_m']<=protocol['success_distance_m'])
            records.append(r); contacts.append(r)
        episodes.append(records)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    x=np.arange(5)
    axes[0].bar(x,[results[v]['summary']['successes'] for v in variants],color=colors)
    axes[0].set(xticks=x,xticklabels=names,ylim=(0,13),ylabel='Targets reached / 12',title='Paired development scenarios, revised box dimensions')
    axes[1].bar(x,[results[v]['summary']['mean_final_error_cm'] for v in variants],color=colors)
    axes[1].set(xticks=x,xticklabels=names,ylabel='Mean final error (cm)',title='All failures retained in the average')
    for ax in axes: ax.tick_params(axis='x',rotation=25)
    fig.tight_layout(); fig.savefig(OUT/'control_comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(14,10))
    for i,(ax,scenario) in enumerate(zip(axes.flat,protocol['evaluation'])):
        goal=np.array(scenario['goal_xy'])*100
        ax.add_patch(plt.Circle(goal,1.5,color='#22c55e',alpha=.2)); ax.scatter(*goal,marker='x',color='#15803d')
        for k in [0,3,4]:
            path=np.array(episodes[k][i]['states'])[:,:2]*100
            ax.plot(*path.T,'-o',markersize=2,color=colors[k],label=names[k])
        ax.set_title(scenario['id']); ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7); fig.suptitle('Every reused scenario: identical goals, revised dimensions and execution')
    fig.tight_layout(); fig.savefig(OUT/'all_scenarios.png',dpi=140); plt.close(fig)
    control_table=['| Controller | Targets reached | Mean final error (cm) | Mean pushes |','|---|---:|---:|---:|']
    for v,name in zip(variants,names):
        r=results[v]['summary']; control_table.append(f'| {name} | {r["successes"]}/12 | {r["mean_final_error_cm"]:.2f} | {r["mean_pushes"]:.2f} |')
    counts=[]
    for source in extraction['sources']:
        e=source['episodes']; counts.append(f'| {source["filename"]} | {sum(a["frames"] for a in e)} | {sum(a["frame_status"].get("accepted",0) for a in e)} | {sum(a["transitions"] for a in e)} |')
    old=training['old_only']['metrics']; new=training['plus_pilot']['metrics']
    probe=extraction['sources'][-1]['stationary_probe']
    test_table=['| Predictor | Position MAE, projected cm | Heading MAE, degrees | Moving-only position MAE, projected cm |','|---|---:|---:|---:|']
    for name,m in [('No movement',new['reused_test']['no_motion']),('Copy contact',new['reused_test']['copy_contact']),('Ridge, cleaned + pilot',new['reused_test']['ridge']),('MLP, cleaned old data',old['reused_test']['small_mlp']),('MLP, cleaned + pilot',new['reused_test']['small_mlp'])]:
        test_table.append(f'| {name} | {m["position_mae_projected_cm"]:.3f} | {m["heading_mae_deg"]:.3f} | {m["moving_position_mae_projected_cm"]:.3f} |')
    text='''# Camera compensation, filtering and pilot-data experiment

This is development work on previously inspected recordings and reused simulation scenarios. It is not a fresh final evaluation. All historical result folders are preserved.

## Processing changes

Each frame requires four observed, uniquely associated table-corner markers. A new homography maps them to the remeasured 40 x 30 cm workspace. Missing or ambiguous corners are rejected; no coordinates or homographies are interpolated. Magenta-only tool detection avoids the cover-colour confusion found in pilot review.

Fixed, broad spacing gates retain apparent caliper spacing of 3-5 cm and roof spacing of 2.25-3.75 cm. The contact proxy extends the shaft vector by 9/40. All four frames of each training transition must pass the gates and existing jump checks. These gates reject gross inconsistency; they do not make accepted positions physically calibrated.

Applying the remeasurements to historical videos assumes marker geometry was comparable before the stickers were secured. Roof/tool heights, tool tilt and roof-marker midpoint alignment remain unresolved. This experiment does not identify each change's causal contribution separately.

## Retained observations

| Source | Reviewed frames | Accepted frames | Retained transitions |
|---|---:|---:|---:|
'''+ '\n'.join(counts)+f'''

The cleaned original recordings supply **264 training, 61 validation and 58 reused-test transitions**. The pilot adds **160 training transitions**, for 424 total training examples. The pilot is not used as a held-out test. Original bottom-to-top recording coverage collapses to 20 transitions across all splits, including just one validation transition. Reported mean errors therefore underrepresent that difficult direction.

![Every episode's retained data](retained_data.png)

The first five seconds of the pilot show a stationary cover. Its x/y position standard deviations are {np.round(probe['fixed_mapping_position_std_cm'],4).tolist()} projected cm with the fixed mapping and {np.round(probe['dynamic_mapping_position_std_cm'],4).tolist()} with framewise mapping. Framewise mapping adds a little detector jitter there; we do not claim it universally reduces noise. A synthetic translated-camera test verifies compensation mathematically. Four-corner fit residuals would be circular evidence of physical calibration and are not used as such.

## Training and offline comparison

Two MLPs use identical architecture, seed and optimization rules: cleaned original data alone, and the same data plus the pilot. Only training data determine normalization; the original validation episodes select stopping. Both models are evaluated on the same 58 retained original test transitions. Those clips were already inspected, so these are development checks. Scores from earlier experiments used different target coordinates and sample subsets and must not be compared as if only the model changed.

'''+ '\n'.join(test_table)+'''

Pilot prediction scores for the combined model are in-sample and cannot demonstrate generalization. The added pilot includes 37 bottom-to-top transitions, helping fill the coverage gap without fixing the measurement uncertainty.

## Paired robot comparison

All five controllers were rerun on the same 12 goals with the **78.5 x 37 x 33 mm** box. The candidate generator uses these revised dimensions too. Every controller has the same observations (exact simulator pose), candidate set, IK/motor execution, friction scenarios and eight-push budget. Success remains within 1.5 cm of the target centre. Original and 11 mm checkpoint weights are unchanged.

'''+ '\n'.join(control_table)+'''

The revised box dimensions can change outcomes even for unchanged checkpoints. These results form a new paired comparison and do not replace the earlier 80 x 35 x 35 mm experiment. All 60 episodes have attached-tool contact and zero robot-body/object contacts.

![Controller comparison](control_comparison.png)

![All trajectories](all_scenarios.png)

## Limits and next decision

The experiment combines revised geometry, camera compensation and filtering. The old-only versus plus-pilot comparison isolates adding the pilot within that processing pipeline; it does not isolate every other change. This is one seed and a small reused scenario set. Exact simulator observations, assumed mass/friction, and a solid cuboid remain simplifications.

Judge the model by both control success and error, rather than just offline prediction or the best demonstration. Any final generalization claim needs a fresh evaluation with settings frozen beforehand. More indiscriminate filming is not the immediate remedy for unmeasured marker height or uncertain contact geometry.

## Reproduce

```powershell
python scripts/stabilize_dataset.py --video-dir "C:\\path\\to\\recordings"
python scripts/train_stabilized.py
python scripts/control_stabilized.py
python scripts/report_stabilized.py
python -m unittest discover -s tests -v
```

The prescribed experiment is `data/stabilized_protocol.json`. Per-frame CSVs include rejection reasons; corner CSVs preserve observations; model files record the training-data hash; controller logs record checkpoint hashes. No physical mass/friction estimates or missing marker measurements were invented.
'''
    (OUT/'REPORT.md').write_text(text)
    summary={'control':{v:results[v]['summary'] for v in variants},'paired_conditions_verified':True,'all_tool_contacts_positive':True,'all_robot_body_contacts_zero':True,
             'offline':{v:training[v]['metrics']['reused_test']['small_mlp'] for v in training},'status':'Reused development data and scenarios.'}
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()

"""Paired development comparisons, all using the revised box dimensions."""
import argparse
import json
import numpy as np
import goal_control as control
from goal_control import ROOT, FrozenDynamics


def main(only=None,case=None,render=False):
    measurement=json.loads((ROOT/'data'/'remeasured_setup.json').read_text())
    protocol=json.loads((ROOT/'data'/'control_scenarios.json').read_text())
    protocol={**protocol,'object_dimensions_m':[measurement['object'][k]/1000 for k in ['length','width','roof_height']],
              'split_note':'Development comparison on reused scenarios with revised dimensions. No fresh generalization claim.'}
    specs={'geometric':('geometric',ROOT/'results'/'baseline'/'small_mlp.pt'),
           'original':('learned',ROOT/'results'/'baseline'/'small_mlp.pt'),
           'offset_11mm':('learned',ROOT/'results'/'contact_proxy'/'small_mlp.pt'),
           'old_only':('learned',ROOT/'results'/'stabilized'/'old_only'/'small_mlp.pt'),
           'plus_pilot':('learned',ROOT/'results'/'stabilized'/'plus_pilot'/'small_mlp.pt')}
    for variant,(policy,path) in specs.items():
        if only and variant!=only: continue
        control.OUT=ROOT/'results'/'stabilized'/'control'/variant
        control.OUT.mkdir(parents=True,exist_ok=True)
        model=FrozenDynamics(path); results=[]
        for scenario in protocol['evaluation']:
            if case and scenario['id']!=case: continue
            r=control.run(scenario,policy,protocol,model,render)
            results.append(r)
            print(json.dumps({'variant':variant,'case':scenario['id'],'success':r['success'],'error_cm':r['final_error_m']*100,'body_contacts':r['robot_body_contact_steps']}),flush=True)
        if not case:
            summary={'episodes':len(results),'successes':sum(r['success'] for r in results),'mean_final_error_cm':float(np.mean([r['final_error_m']*100 for r in results])),
                     'mean_pushes':float(np.mean([r['pushes'] for r in results])),'checkpoint_sha256':model.sha256}
            (control.OUT/'summary.json').write_text(json.dumps({'protocol':protocol,'variant':variant,'summary':summary},indent=2))
            print(json.dumps({'variant':variant,**summary}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--variant',choices=['geometric','original','offset_11mm','old_only','plus_pilot']); parser.add_argument('--case'); parser.add_argument('--render',action='store_true')
    args=parser.parse_args(); main(args.variant,args.case,args.render)

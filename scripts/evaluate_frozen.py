"""One frozen, fresh simulation evaluation after the development experiment.

Protocol is written before any case runs. Subsequent replays require identical
checkpoint/code hashes and reuse exactly the saved scenarios, never a new seed.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import goal_control as control
from goal_control import ROOT, FrozenDynamics

OUT=ROOT/'results'/'frozen_evaluation'
PROTOCOL=ROOT/'data'/'frozen_evaluation_protocol.json'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checkpoint=ROOT/'results'/'stabilized'/'plus_pilot'/'small_mlp.pt'
    code_hashes={name:digest(ROOT/'scripts'/name) for name in ['goal_control.py','replay_panda.py','train_baseline.py','evaluate_frozen.py']}
    if PROTOCOL.exists():
        protocol=json.loads(PROTOCOL.read_text())
        if protocol['checkpoint_sha256']!=digest(checkpoint) or protocol['code_sha256']!=code_hashes:
            raise ValueError('Frozen evaluation code/checkpoint changed. Preserve the prior run; do not silently reuse as fresh evidence.')
    else:
        rng=np.random.default_rng(824173)
        frictions=np.array([.25,.4,.6]*8); rng.shuffle(frictions)
        scenarios=[]
        for i in range(24):
            xy=np.array([.5,0.])+rng.uniform(-.015,.015,2)
            angle=rng.uniform(-np.pi,np.pi); radius=rng.uniform(.055,.075)
            scenarios.append({'id':f'fresh_{i+1:02}','initial_xy':xy.tolist(),'initial_yaw':float(rng.uniform(-.25,.25)),
                              'goal_xy':(xy+radius*np.array([np.cos(angle),np.sin(angle)])).tolist(),'friction':float(frictions[i])})
        measurement=json.loads((ROOT/'data'/'remeasured_setup.json').read_text())
        protocol={'seed':824173,'checkpoint_sha256':digest(checkpoint),'code_sha256':code_hashes,
                  'object_dimensions_m':[measurement['object'][k]/1000 for k in ['length','width','roof_height']],
                  'success_distance_m':.015,'max_pushes':8,'observations':'Exact simulator object position and yaw for both controllers.',
                  'selection_note':'Combined model frozen after cleaned-data development comparison; fresh scenarios generated before running either controller. No parameter or threshold tuning on these outcomes.',
                  'scope':'New simulation initial states and goals in the same simplified environment and parameter ranges. Not fresh real-world data, unseen objects, or vision-based robot control.',
                  'evaluation':scenarios}
        PROTOCOL.write_text(json.dumps(protocol,indent=2))
    model=FrozenDynamics(checkpoint); results=[]
    for policy in ['geometric','learned']:
        control.OUT=OUT/policy; control.OUT.mkdir(parents=True,exist_ok=True)
        records=[]
        for scenario in protocol['evaluation']:
            r=control.run(scenario,policy,protocol,model)
            records.append(r); results.append(r)
            print(json.dumps({'policy':policy,'case':scenario['id'],'success':r['success'],'error_cm':r['final_error_m']*100}),flush=True)
        summary={'episodes':len(records),'successes':sum(r['success'] for r in records),'mean_final_error_cm':float(np.mean([r['final_error_m']*100 for r in records])),
                 'mean_pushes':float(np.mean([r['pushes'] for r in records]))}
        (control.OUT/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({'policy':policy,**summary}),flush=True)
    # Validate actual execution and exact scenario pairing before reporting.
    for r in results:
        assert r['checkpoint_sha256']==protocol['checkpoint_sha256']
        assert r['robot_body_contact_steps']==0 and r['tool_contact_steps']>0
        assert r['success']==(r['final_error_m']<=protocol['success_distance_m'])
    summary={p:json.loads((OUT/p/'summary.json').read_text()) for p in ['geometric','learned']}
    (OUT/'summary.json').write_text(json.dumps({'protocol':protocol,'summary':summary,'all_attached_tool_contacts_positive':True,'all_robot_body_contacts_zero':True},indent=2))


if __name__=='__main__': main()

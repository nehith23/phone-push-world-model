"""Single new simulation sample for the fixed fine-action precision controller."""
import hashlib
import json
import numpy as np
from precision_experiment import ROOT, FrozenDynamics, run

OUT=ROOT/'results'/'precision'/'fresh'
PROTOCOL=ROOT/'data'/'precision_frozen_protocol.json'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    model=FrozenDynamics(ROOT/'results'/'stabilized'/'plus_pilot'/'small_mlp.pt')
    base=json.loads((ROOT/'data'/'precision_protocol.json').read_text())
    assert base['checkpoint_sha256']==model.sha256
    for name,digest in base['code_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
    hashes={**base['code_sha256'],'evaluate_precision_frozen.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if PROTOCOL.exists():
        protocol=json.loads(PROTOCOL.read_text())
        if protocol['code_sha256']!=hashes or protocol['checkpoint_sha256']!=model.sha256:
            raise ValueError('Precision evaluation code/model changed; preserve the saved run.')
    else:
        rng=np.random.default_rng(637251)
        frictions=np.array([.25,.4,.6]*8); rng.shuffle(frictions); scenarios=[]
        for i in range(24):
            xy=np.array([.5,0.])+rng.uniform(-.015,.015,2)
            angle=rng.uniform(-np.pi,np.pi); radius=rng.uniform(.055,.075)
            scenarios.append({'id':f'precision_{i+1:02}','initial_xy':xy.tolist(),'initial_yaw':float(rng.uniform(-.25,.25)),
                              'goal_xy':(xy+radius*np.array([np.cos(angle),np.sin(angle)])).tolist(),'friction':float(frictions[i])})
        protocol={**base,'seed':637251,'code_sha256':hashes,'evaluation':scenarios,
                  'status':'New simulation scenarios saved before either fine-action controller runs. Parameters fixed after the 12-case precision development comparison.',
                  'comparison':'Geometric fine versus learned fine, 5 mm tolerance, 12 pushes. Same frozen network and observed simulator pose.',
                  'scope':'One fresh precision simulation sample; no new real-world recording or physical accuracy claim. Do not tune on these outcomes and continue calling them unseen.'}
        PROTOCOL.write_text(json.dumps(protocol,indent=2))
    summaries={}
    for policy in ['geometric','learned']:
        records=[]
        for scenario in protocol['evaluation']:
            record=run(scenario,policy,'fine',protocol,model); records.append(record)
            assert record['robot_body_contact_steps']==0 and record['tool_contact_steps']>0
            (OUT/f'{scenario["id"]}_{policy}.json').write_text(json.dumps(record,indent=2))
            print(json.dumps({'policy':policy,'case':scenario['id'],'success':record['success'],'error_mm':record['final_error_m']*1000}),flush=True)
        summaries[policy]={'episodes':len(records),'successes':sum(r['success'] for r in records),
                           'mean_final_error_mm':float(np.mean([r['final_error_m']*1000 for r in records])),
                           'median_final_error_mm':float(np.median([r['final_error_m']*1000 for r in records])),
                           'mean_pushes':float(np.mean([r['pushes'] for r in records]))}
        print(json.dumps({policy:summaries[policy]}),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'protocol':protocol,'summary':summaries},indent=2))


if __name__=='__main__':
    from pathlib import Path
    main()

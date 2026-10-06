"""Fixed precision development experiment; original frozen evaluation is intact."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from goal_control import ROOT, FrozenDynamics, candidates, observe, execute
from replay_panda import PandaScene

OUT=ROOT/'results'/'precision'
PROTOCOL=ROOT/'data'/'precision_protocol.json'


def actions(state,goal,half,mode):
    base=candidates(state,half)
    if mode=='coarse' or np.linalg.norm(state[:2]-goal)>.02: return base
    # Keep old candidates first so exact ties preserve the old choice order.
    fine=[]
    for a in base[::2]:
        for length in [.002,.005]: fine.append({**a,'length_m':length})
    return base+fine


def predict(state,action,policy,model):
    if policy=='geometric': return np.r_[state[:2]+action['direction']*action['length_m'],state[2]]
    prediction=np.r_[state[:2]*100,state[2]]
    front=action['start_xy']*100+action['direction']*model.tool_front_offset_cm
    rear=front-action['direction']*4
    # Execution uses at least 0.2 s. Match it with >=2 nominal 0.1 s
    # predictions, rather than rounding a 2 mm action down to zero steps.
    count=max(2,int(round(action['length_m']/.005)))
    step=action['direction']*action['length_m']*100/count
    for _ in range(count):
        prediction=model.predict(prediction,front,rear,step)
        front=front+step; rear=rear+step
    return np.r_[prediction[:2]/100,prediction[2]]


def run(scenario,policy,mode,protocol,model):
    scene=PandaScene(scenario['initial_xy'],scenario['initial_yaw'],scenario['friction'],object_dimensions=protocol['object_dimensions_m'])
    half=np.array(protocol['object_dimensions_m'][:2])/2; goal=np.array(scenario['goal_xy'])
    states=[observe(scene).tolist()]; events=[]
    try:
        for attempt in range(protocol['max_pushes']):
            before=observe(scene)
            if np.linalg.norm(before[:2]-goal)<=protocol['success_distance_m']: break
            choices=[]
            for index,action in enumerate(actions(before,goal,half,mode)):
                predicted=predict(before,action,policy,model)
                cost=float(np.linalg.norm(predicted[:2]-goal)**2)
                choices.append((cost,index,action,predicted))
            _,index,action,predicted=min(choices,key=lambda item:(item[0],item[1]))
            execute(scene,action); after=observe(scene); states.append(after.tolist())
            events.append({'push':attempt+1,'candidate_index':index,'length_m':action['length_m'],
                           'axis':action['axis'],'sign':action['sign'],'offset_fraction':action['offset_fraction'],
                           'predicted_state':predicted.tolist(),'observed_before':before.tolist(),'observed_after':after.tolist(),
                           'distance_to_goal_m':float(np.linalg.norm(after[:2]-goal))})
            if not(.25<after[0]<.75 and -.25<after[1]<.25): break
        error=float(np.linalg.norm(observe(scene)[:2]-goal))
        return {'scenario':scenario,'policy':policy,'mode':mode,'success':error<=protocol['success_distance_m'],
                'final_error_m':error,'pushes':len(events),'events':events,'states':states,
                'fine_pushes':sum(e['length_m']<.01 for e in events),'tool_contact_steps':scene.contact_steps,
                'robot_body_contact_steps':scene.finger_contact_steps,'checkpoint_sha256':model.sha256}
    finally:
        scene.close()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'results'/'stabilized'/'plus_pilot'/'small_mlp.pt'
    model=FrozenDynamics(checkpoint)
    hashes={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ['precision_experiment.py','goal_control.py','replay_panda.py','train_baseline.py']}
    if PROTOCOL.exists():
        protocol=json.loads(PROTOCOL.read_text())
        if protocol['checkpoint_sha256']!=model.sha256 or protocol['code_sha256']!=hashes:
            raise ValueError('Preserve the prescribed precision run; code or checkpoint has changed.')
    else:
        old=json.loads((ROOT/'data'/'control_scenarios.json').read_text())
        frozen=json.loads((ROOT/'data'/'frozen_evaluation_protocol.json').read_text())
        protocol={'status':'Prescribed precision development experiment, using the original 12 already-inspected scenarios, not the 24 fresh frozen-evaluation cases.',
                  'checkpoint_sha256':model.sha256,'code_sha256':hashes,'object_dimensions_m':frozen['object_dimensions_m'],
                  'success_distance_m':.005,'max_pushes':12,'fine_action_activation_distance_m':.02,
                  'coarse_lengths_m':[.01,.02],'added_fine_lengths_m':[.002,.005],
                  'comparison':'Coarse versus fine action sets, crossed with geometric versus learned controller. Identical 5 mm tolerance and 12-push budget for all four conditions.',
                  'prediction':'Short strokes use >=2 nominal 0.1 s model steps matching the minimum 0.2 s physical push execution. No retraining; low-speed predictions may be inaccurate.',
                  'observations':'Exact simulated object position and yaw; this does not establish physical millimetre accuracy.',
                  'scope':'One bounded experiment; no sweeps or post-outcome threshold changes. Original frozen evaluation remains unchanged.',
                  'evaluation':old['evaluation']}
        PROTOCOL.write_text(json.dumps(protocol,indent=2))
    summaries={}
    for mode in ['coarse','fine']:
        for policy in ['geometric','learned']:
            records=[]
            for scenario in protocol['evaluation']:
                record=run(scenario,policy,mode,protocol,model); records.append(record)
                assert record['robot_body_contact_steps']==0 and record['tool_contact_steps']>0
                (OUT/f'{scenario["id"]}_{policy}_{mode}.json').write_text(json.dumps(record,indent=2))
                print(json.dumps({'mode':mode,'policy':policy,'case':scenario['id'],'success':record['success'],'error_mm':record['final_error_m']*1000,'fine_pushes':record['fine_pushes']}),flush=True)
            summaries[f'{policy}_{mode}']={'episodes':len(records),'successes':sum(r['success'] for r in records),
                                         'mean_final_error_mm':float(np.mean([r['final_error_m']*1000 for r in records])),
                                         'median_final_error_mm':float(np.median([r['final_error_m']*1000 for r in records])),
                                         'mean_pushes':float(np.mean([r['pushes'] for r in records])),
                                         'total_fine_pushes':sum(r['fine_pushes'] for r in records)}
            print(json.dumps({f'{policy}_{mode}':summaries[f'{policy}_{mode}']}),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'protocol':protocol,'summary':summaries},indent=2))


if __name__=='__main__': main()

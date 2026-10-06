"""Compare learned-dynamics action selection with a geometric controller.

Both controllers get exact simulated object pose, the same candidate pushes,
the same IK/motor execution, and an identical episode budget. The neural model
is frozen from the phone-data experiment. Direct transfer identifies its biased
projected coordinates with simulator coordinates; no transfer accuracy is assumed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import cv2
import numpy as np
import torch
from torch import nn
import pybullet as p
from replay_panda import PandaScene, rotation
from train_baseline import features, world, wrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'goal_control'
OUT.mkdir(parents=True,exist_ok=True)

class FrozenDynamics:
    def __init__(self,checkpoint_path=None):
        torch.set_num_threads(2)
        path=Path(checkpoint_path) if checkpoint_path else ROOT/'results'/'baseline'/'small_mlp.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        self.model=nn.Sequential(nn.Linear(checkpoint['input_dim'],32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,3))
        self.model.load_state_dict(checkpoint['state_dict']); self.model.eval()
        self.mean=np.array(checkpoint['x_mean']); self.scale=np.array(checkpoint['x_scale']); self.ys=np.array(checkpoint['y_scale'])
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        self.representation=checkpoint.get('representation','marker')
        # Panda tool has 0.8 cm radius: its leading surface is ahead of its
        # centre. Use this point only for the contact-proxy representation.
        self.tool_front_offset_cm=.8 if self.representation=='contact_proxy' else 0.

    def predict(self,state,front,rear,action):
        value=(features(state,front,rear,action)-self.mean)/self.scale
        with torch.no_grad(): delta=self.model(torch.tensor(value[None,:],dtype=torch.float32)).numpy()[0]*self.ys
        return np.r_[state[:2]+world(delta[:2],state[2]),wrap(state[2]+delta[2])]

def candidates(state,object_half_xy=None):
    xy=state[:2]; theta=state[2]; R=rotation(theta)
    half=np.array(object_half_xy if object_half_xy is not None else [.04,.0175]); result=[]
    for axis in [0,1]:
        for sign in [-1,1]:
            d=np.zeros(2); d[axis]=sign; direction=R@d
            for offset in [0.,-.5,.5]:
                start=np.zeros(2); start[axis]=-sign*(half[axis]+.008+.001)
                start[1-axis]=offset*half[1-axis]
                for length in [.01,.02]:
                    result.append({'axis':axis,'sign':sign,'offset_fraction':offset,'length_m':length,'start_xy':xy+R@start,'direction':direction})
    return result

def predict_candidate(state,action,policy,model):
    if policy=='geometric': return np.r_[state[:2]+action['direction']*action['length_m'],state[2]]
    prediction=np.r_[state[:2]*100,state[2]]
    front=action['start_xy']*100
    front=front+action['direction']*getattr(model,'tool_front_offset_cm',0.)
    rear=front-action['direction']*4
    # Approx. 0.1-s learned steps, each commanding 0.5 cm of tool travel.
    count=int(round(action['length_m']/.005))
    step=action['direction']*.5
    for _ in range(count):
        prediction=model.predict(prediction,front,rear,step)
        front=front+step; rear=rear+step
    return np.r_[prediction[:2]/100,prediction[2]]

def select_action(state,goal,policy,model,object_half_xy=None):
    choices=[]
    for index,action in enumerate(candidates(state,object_half_xy)):
        prediction=predict_candidate(state,action,policy,model)
        # Goal task is position only. Tie break prefers central contacts by order.
        cost=float(np.linalg.norm(prediction[:2]-goal)**2)
        choices.append((cost,index,action,prediction))
    return min(choices,key=lambda x:(x[0],x[1]))

def observe(scene):
    xyz,q=p.getBasePositionAndOrientation(scene.box)
    return np.r_[xyz[:2],p.getEulerFromQuaternion(q)[2]]

def execute(scene,action):
    # Lift before traversing so repositioning does not push through the object.
    current=np.array(p.getBasePositionAndOrientation(scene.tool)[0])
    scene.move([*current[:2],.14],60)
    direction=action['direction']
    tool_yaw=np.arctan2(direction[1],direction[0])
    # A parallel gripper is symmetric under a half turn. Avoid unnecessary wrist wrap.
    tool_yaw=(tool_yaw+np.pi/2)%np.pi-np.pi/2
    scene.orientation=p.getQuaternionFromEuler([np.pi,0,tool_yaw])
    scene.move([*action['start_xy'],.14],96)
    scene.move([*action['start_xy'],.022],96)
    end=action['start_xy']+direction*(action['length_m']+.001)
    scene.move([*end,.022],max(48,int(240*action['length_m']/.05)))
    for _ in range(48): scene.step_target([*end,.022])

def make_scenarios():
    path=ROOT/'data'/'control_scenarios.json'
    if path.exists(): return json.loads(path.read_text())
    rng=np.random.default_rng(20261005)
    development=[]
    for i,angle in enumerate([0,np.pi/2,np.pi,3*np.pi/2]):
        development.append({'id':f'dev_{i+1:02}','initial_xy':[.5,0.],'initial_yaw':0.,'goal_xy':[float(.5+.06*np.cos(angle)),float(.06*np.sin(angle))],'friction':.4})
    evaluation=[]
    for i in range(12):
        xy=np.array([.5,0.])+rng.uniform(-.015,.015,2)
        angle=2*np.pi*i/12; distance=float(rng.uniform(.055,.075))
        goal=xy+distance*np.array([np.cos(angle),np.sin(angle)])
        evaluation.append({'id':f'eval_{i+1:02}','initial_xy':xy.tolist(),'initial_yaw':float(rng.uniform(-.25,.25)),'goal_xy':goal.tolist(),'friction':[.25,.4,.6][i%3]})
    protocol={'seed':20261005,'success_distance_m':.015,'max_pushes':8,'observations':'Exact simulated object position and yaw, identical for both controllers.','candidates':'Four object-relative inward directions, three lateral offsets, two stroke lengths.','split_note':'Development cases debug execution. Evaluation scenarios generated before running either controller; do not tune on evaluation outcomes.','development':development,'evaluation':evaluation}
    path.write_text(json.dumps(protocol,indent=2)); return protocol

def run(scenario,policy,protocol,model,render=False):
    dimensions=protocol.get('object_dimensions_m')
    half_xy=np.array(dimensions[:2])/2 if dimensions is not None else None
    scene=PandaScene(scenario['initial_xy'],scenario['initial_yaw'],scenario['friction'],render,object_dimensions=dimensions)
    goal=np.array(scenario['goal_xy'])
    # A visible goal has no collision shape and cannot influence physics.
    visual=p.createVisualShape(p.GEOM_CYLINDER,radius=protocol['success_distance_m'],length=.001,rgbaColor=[.05,.65,.2,.65])
    p.createMultiBody(0,-1,visual,[*goal,.0005])
    events=[]; states=[observe(scene)]; final_error=float(np.linalg.norm(states[-1][:2]-goal))
    for attempt in range(protocol['max_pushes']):
        before=observe(scene)
        if np.linalg.norm(before[:2]-goal)<=protocol['success_distance_m']: break
        cost,index,action,prediction=select_action(before,goal,policy,model,half_xy)
        execute(scene,action); after=observe(scene); states.append(after)
        events.append({'push':attempt+1,'candidate_index':index,'axis':action['axis'],'sign':action['sign'],'offset_fraction':action['offset_fraction'],'length_m':action['length_m'],'predicted_state':prediction.tolist(),'observed_before':before.tolist(),'observed_after':after.tolist(),'prediction_position_error_m':float(np.linalg.norm(prediction[:2]-after[:2])),'distance_to_goal_m':float(np.linalg.norm(after[:2]-goal))})
        # Same workspace bounds for both controllers; a departure is a failure.
        if not(.25<after[0]<.75 and -.25<after[1]<.25): break
    final=observe(scene); final_error=float(np.linalg.norm(final[:2]-goal))
    report={'scenario':scenario,'policy':policy,'success':final_error<=protocol['success_distance_m'],'final_error_m':final_error,'pushes':len(events),'events':events,'states':np.array(states).tolist(),'tool_contact_steps':int(scene.contact_steps),'robot_body_contact_steps':int(scene.finger_contact_steps),'checkpoint_sha256':model.sha256,'observation_assumption':protocol['observations']}
    name=f'{scenario["id"]}_{policy}'
    if render and scene.frames:
        writer=cv2.VideoWriter(str(OUT/f'{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480))
        if not writer.isOpened(): raise RuntimeError('Could not open video writer')
        for frame in scene.frames:
            cv2.rectangle(frame,(0,0),(640,68),(245,245,245),-1)
            cv2.putText(frame,f'{policy}: choose a push, observe, replan',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.6,(25,25,25),2,cv2.LINE_AA)
            cv2.putText(frame,'Green disk = target | exact simulator state supplied',(12,53),cv2.FONT_HERSHEY_SIMPLEX,.48,(65,65,65),1,cv2.LINE_AA)
            writer.write(frame)
        writer.release(); cv2.imwrite(str(OUT/f'{name}_final.jpg'),scene.frames[-1])
    scene.close(); (OUT/f'{name}.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--split',choices=['development','evaluation'],default='development'); parser.add_argument('--policy',choices=['geometric','learned','both'],default='both'); parser.add_argument('--case'); parser.add_argument('--render',action='store_true')
    parser.add_argument('--experiment',choices=['original','contact_proxy'],default='original')
    args=parser.parse_args(); protocol=make_scenarios(); results=[]
    checkpoint_path=None
    if args.experiment=='contact_proxy':
        OUT=ROOT/'results'/'contact_proxy'/'control'; OUT.mkdir(parents=True,exist_ok=True)
        checkpoint_path=ROOT/'results'/'contact_proxy'/'small_mlp.pt'
        protocol={**protocol,'split_note':'Reused scenarios after original evaluation inspection. Development comparison only; not fresh held-out evidence.'}
    model=FrozenDynamics(checkpoint_path)
    for scenario in protocol[args.split]:
        if args.case and scenario['id']!=args.case: continue
        for policy in ['geometric','learned'] if args.policy=='both' else [args.policy]:
            result=run(scenario,policy,protocol,model,args.render); results.append(result)
            print(json.dumps({'case':scenario['id'],'policy':policy,'success':result['success'],'error_cm':result['final_error_m']*100,'pushes':result['pushes'],'body_contacts':result['robot_body_contact_steps']}),flush=True)
    if not args.case and args.policy=='both':
        summary={policy:{'episodes':sum(r['policy']==policy for r in results),'successes':sum(r['policy']==policy and r['success'] for r in results),'mean_final_error_cm':float(np.mean([r['final_error_m']*100 for r in results if r['policy']==policy])),'mean_pushes':float(np.mean([r['pushes'] for r in results if r['policy']==policy]))} for policy in ['geometric','learned']}
        (OUT/f'{args.split}_summary.json').write_text(json.dumps({'protocol':protocol,'summary':summary},indent=2)); print(json.dumps(summary,indent=2))

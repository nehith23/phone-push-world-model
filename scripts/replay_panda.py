"""Replay user-recorded tool motion through Panda IK and physical contact.

This is demonstration retargeting, NOT a learned policy or validated sim-to-real
transfer. A cuboid approximates the user's hollow robot cover. All object motion
after initialization comes from PyBullet contact dynamics, not pose teleportation.
"""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import pybullet as p
import pybullet_data

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'simulation'
OUT.mkdir(parents=True,exist_ok=True)

def rotation(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s],[s,c]])

def recorded_path(clip_id,episode):
    raw=np.genfromtxt(ROOT/'results'/'tracking'/f'{clip_id}_tracks.csv',delimiter=',',skip_header=1)
    block=raw[raw[:,2]==episode]
    valid=np.isfinite(block[:,11:18]).all(axis=1)
    edges=np.diff(np.r_[False,valid,False].astype(int))
    runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    if not runs: raise ValueError('No fully observed run')
    start,end=max(runs,key=lambda r:r[1]-r[0]); segment=block[start:end]
    delta=np.linalg.norm(np.diff(segment[:,11:13],axis=0),axis=1)
    moving=np.flatnonzero(delta>.04)
    if len(moving)<3: raise ValueError('Too little measured object movement')
    lo=max(0,moving[0]-2); hi=min(len(segment),moving[-1]+4)
    segment=segment[lo:hi]
    return segment,{'source_frames':[int(segment[0,0]),int(segment[-1,0])],'source_time_s':[float(segment[0,1]),float(segment[-1,1])],'review_frames':len(block),'used_frames':len(segment),'selection':'Longest continuously observed run; trimmed to observed object movement with 2-frame margins. No missing-frame interpolation.'}

class PandaScene:
    def __init__(self,object_xy,theta,friction,render=False,tool_yaw=0.,object_dimensions=None):
        self.client=p.connect(p.DIRECT); self.frames=[]; self.steps=0; self.render=render; self.tool_length=.06
        p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0,0,-9.81); p.setTimeStep(1/240)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        self.plane=p.loadURDF('plane.urdf')
        self.robot=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True)
        for j,v in enumerate([0,-.5,0,-2.3,0,1.8,.8]): p.resetJointState(self.robot,j,v)
        for j in [9,10]: p.resetJointState(self.robot,j,.04)
        self.orientation=p.getQuaternionFromEuler([np.pi,0,tool_yaw])
        self.view=p.computeViewMatrix([.93,-.78,.65],[.48,0,.12],[0,0,1])
        self.projection=p.computeProjectionMatrixFOV(42,4/3,.01,3)
        self.move([.42,0,.25],240,capture=False)
        link=p.getLinkState(self.robot,11)
        start=np.asarray(link[4])+np.asarray(p.getMatrixFromQuaternion(link[5])).reshape(3,3)@np.array([0,0,self.tool_length])
        shape=p.createCollisionShape(p.GEOM_SPHERE,radius=.008)
        visual=p.createVisualShapeArray(shapeTypes=[p.GEOM_SPHERE,p.GEOM_CYLINDER],radii=[.008,.003],lengths=[0,self.tool_length],rgbaColors=[[.85,.1,.5,1],[.3,.3,.35,1]],visualFramePositions=[[0,0,0],[0,0,-self.tool_length/2]])
        self.tool=p.createMultiBody(.03,shape,visual,start)
        constraint=p.createConstraint(self.robot,11,self.tool,-1,p.JOINT_FIXED,[0,0,0],[0,0,self.tool_length],[0,0,0])
        p.changeConstraint(constraint,maxForce=200)
        for j in range(-1,p.getNumJoints(self.robot)): p.setCollisionFilterPair(self.robot,self.tool,j,-1,0)
        half=np.array(object_dimensions if object_dimensions is not None else [.08,.035,.035])/2
        shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=half.tolist())
        visual=p.createVisualShape(p.GEOM_BOX,halfExtents=half.tolist(),rgbaColor=[1,.4,.05,1])
        self.box=p.createMultiBody(.1,shape,visual,[*object_xy,float(half[2])],p.getQuaternionFromEuler([0,0,theta]))
        p.changeDynamics(self.box,-1,lateralFriction=friction,rollingFriction=0,restitution=0)
        self.contact_steps=0; self.finger_contact_steps=0; self.object_history=[]; self.tool_history=[]

    def step_target(self,target,capture=True):
        ee_target=np.asarray(target)+[0,0,self.tool_length]
        q=p.calculateInverseKinematics(self.robot,11,ee_target,self.orientation,maxNumIterations=80,residualThreshold=1e-5)
        p.setJointMotorControlArray(self.robot,list(range(7)),p.POSITION_CONTROL,targetPositions=q[:7],forces=[87,87,87,87,12,12,12])
        p.setJointMotorControlArray(self.robot,[9,10],p.POSITION_CONTROL,targetPositions=[.04,.04],forces=[20,20])
        p.stepSimulation(); self.steps+=1
        if hasattr(self,'box'):
            self.object_history.append(p.getBasePositionAndOrientation(self.box)[0])
            self.tool_history.append(p.getBasePositionAndOrientation(self.tool)[0])
            self.contact_steps+=bool(p.getContactPoints(self.tool,self.box))
            self.finger_contact_steps+=bool(p.getContactPoints(self.robot,self.box))
        if capture and self.render and self.steps%24==0:
            image=p.getCameraImage(640,480,self.view,self.projection,renderer=p.ER_TINY_RENDERER)
            rgba=np.reshape(image[2],(480,640,4))
            self.frames.append(cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR))

    def move(self,target,steps,capture=True):
        link=p.getLinkState(self.robot,11)
        start=np.asarray(link[4])+np.asarray(p.getMatrixFromQuaternion(link[5])).reshape(3,3)@np.array([0,0,self.tool_length])
        for k in range(steps): self.step_target(start+(np.asarray(target)-start)*(k+1)/steps,capture)

    def close(self): p.disconnect(self.client)

def replay(clip,episode,friction=.4,render=False):
    segment,selection=recorded_path(clip['id'],episode)
    action_path=(segment[:,14:16]-segment[0,14:16])/100
    theta=float(segment[0,13]); R=rotation(theta)
    object_xy=np.array([.5,0.]); direction=np.array([clip['screen_push_direction'][0],-clip['screen_push_direction'][1]],float)
    # Select the appropriate contact face and preserve a bounded lateral offset.
    local_direction=R.T@direction
    face_axis=int(np.argmax(np.abs(local_direction)))
    half=np.array([.04,.0175]); local_offset=np.zeros(2)
    local_offset[face_axis]=-np.sign(local_direction[face_axis])*(half[face_axis]+.008+.003)
    source_offset=R.T@((segment[0,14:16]-segment[0,11:13])/100)
    tangent=1-face_axis
    local_offset[tangent]=np.clip(source_offset[tangent],-.6*half[tangent],.6*half[tangent])
    start_xy=object_xy+R@local_offset
    scene=PandaScene(object_xy,theta,friction,render,tool_yaw=np.pi/2 if abs(direction[1])>.5 else 0.)
    scene.move([*start_xy,.18],240); scene.move([*start_xy,.022],480)
    actual_start=np.array(p.getBasePositionAndOrientation(scene.box)[0])
    # A fixed 3-s execution time permits stable tracking; trajectory shape is data-derived.
    simulation_times=np.linspace(0,1,720)
    source_times=(segment[:,1]-segment[0,1])/(segment[-1,1]-segment[0,1])
    xy=np.column_stack([np.interp(simulation_times,source_times,action_path[:,k]) for k in range(2)])
    for displacement in xy: scene.step_target([*(start_xy+displacement),.022])
    endpoint=start_xy+xy[-1]
    for _ in range(120): scene.step_target([*endpoint,.022])
    final_pos,final_q=p.getBasePositionAndOrientation(scene.box)
    final_theta=p.getEulerFromQuaternion(final_q)[2]
    history=np.array(scene.object_history)
    report={'clip':clip['id'],'episode':episode,'source_selection':selection,'friction_assumed':friction,'mass_kg_assumed':.1,'replay_duration_s':3.0,'initial_object_xy_m':actual_start[:2].tolist(),'approach_displacement_m':(actual_start[:2]-object_xy).tolist(),'final_object_xy_m':list(final_pos[:2]),'object_displacement_m':(np.array(final_pos[:2])-actual_start[:2]).tolist(),'final_yaw_rad':float(final_theta),'physical_tool_contact_steps':int(scene.contact_steps),'robot_body_contact_steps':int(scene.finger_contact_steps),'source_tool_displacement_projected_m':action_path[-1].tolist(),'object_height_range_m':[float(history[:,2].min()),float(history[:,2].max())],'scope':'Physical-contact demonstration replay, not a learned policy or target-success benchmark. Trajectories derive from tabletop-projected video markers; cover approximated by solid box.'}
    name=f'{clip["id"]}_{episode:02}'
    if render and scene.frames:
        writer=cv2.VideoWriter(str(OUT/f'{name}_replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480))
        if not writer.isOpened(): raise RuntimeError('MP4 writer could not open')
        for frame in scene.frames:
            cv2.rectangle(frame,(0,0),(640,68),(245,245,245),-1)
            cv2.putText(frame,'Panda replay driven by your phone recording',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.57,(25,25,25),2,cv2.LINE_AA)
            cv2.putText(frame,'Physical contact | approximate retargeting | not a learned policy',(12,53),cv2.FONT_HERSHEY_SIMPLEX,.44,(65,65,65),1,cv2.LINE_AA)
            writer.write(frame)
        writer.release(); cv2.imwrite(str(OUT/f'{name}_final.jpg'),scene.frames[-1])
    np.savez_compressed(OUT/f'{name}_trajectory.npz',object_xyz=history,tool_xyz=np.array(scene.tool_history),source_action_xy=action_path)
    (OUT/f'{name}_replay.json').write_text(json.dumps(report,indent=2)); scene.close()
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--clip',choices=['left','right','down','top','all'],default='all'); parser.add_argument('--episode',type=int,default=1); parser.add_argument('--render',action='store_true')
    args=parser.parse_args(); manifest=json.loads((ROOT/'data'/'manifest.json').read_text())
    for clip in manifest['videos']:
        if args.clip!='all' and args.clip!=clip['id']: continue
        print(json.dumps(replay(clip,args.episode,render=args.render)),flush=True)

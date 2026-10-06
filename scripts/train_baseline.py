"""Small action-conditioned state predictor and honest, episode-held-out baselines.

Inputs use current observed state and the executed tool displacement over 3 frames.
Targets are observed object translation and rotation over those same frames.
No video/frame random split: pushes 1-4 (+7) train, 5 validate, 6 test, per direction.
Positions are tabletop projections of elevated markers, NOT accurate metric poses.
"""
from pathlib import Path
import json
import copy
import argparse
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'baseline'
OUT.mkdir(parents=True,exist_ok=True)
STRIDE=3

def wrap(theta): return (theta+np.pi)%(2*np.pi)-np.pi

def body(v,theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1]])

def world(v,theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([c*v[0]-s*v[1],s*v[0]+c*v[1]])

def features(state,front,rear,action):
    r=body(front-state[:2],state[2])/8.0
    a=body(action,state[2])
    u=body(front-rear,state[2]); u=u/max(np.linalg.norm(u),1e-8)
    return np.r_[a,r,u,r[0]*a[1]-r[1]*a[0],np.dot(r,a),r[0]*a,r[1]*a]

def contact_proxy(front, rear, ratio):
    """Local affine extrapolation along the shaft, not full 3-D calibration.

    A 1.1 cm offset along a 4 cm marker separation is 0.275 of the
    observed vector. This follows foreshortening approximately instead
    of adding a fixed 1.1 projected cm in every view.
    """
    return front + ratio * (front - rear)


def load_data(contact_offset_ratio=0.0):
    manifest=json.loads((ROOT/'data'/'manifest.json').read_text())
    episodes={}; samples=[]
    for clip in manifest['videos']:
        raw=np.genfromtxt(ROOT/'results'/'tracking'/f'{clip["id"]}_tracks.csv',delimiter=',',skip_header=1)
        observed_raw=raw.copy()
        if contact_offset_ratio:
            front=raw[:,14:16].copy(); rear=raw[:,16:18].copy()
            tip=contact_proxy(front,rear,contact_offset_ratio)
            raw[:,14:16]=tip
            # Virtual rear point preserves the observed shaft direction.
            raw[:,16:18]=tip-(front-rear)
        for j in range(1,len(clip['windows'])+1):
            ep=f'{clip["id"]}_{j:02}'; split='validation' if j==5 else ('test' if j==6 else 'train')
            block=raw[raw[:,2]==j]; episodes[ep]={'split':split,'raw':block,'direction':clip['id']}
            observed_block=observed_raw[observed_raw[:,2]==j]
            for start in range(0,len(block)-STRIDE,STRIDE):
                seq=block[start:start+STRIDE+1]
                measured=observed_block[start:start+STRIDE+1]
                if not np.isfinite(measured[:,11:18]).all(): continue
                # Prevent transitions across missing/corrupted frames or tool swaps.
                if np.max(np.linalg.norm(np.diff(measured[:,11:13],axis=0),axis=1))>2.0: continue
                if np.max(np.linalg.norm(np.diff(measured[:,14:16],axis=0),axis=1))>3.0: continue
                first,last=seq[0],seq[-1]; state=first[11:14]; nxt=last[11:14]
                front,rear=first[14:16],first[16:18]; action=last[14:16]-front
                target=np.r_[body(nxt[:2]-state[:2],state[2]),wrap(nxt[2]-state[2])]
                samples.append({'episode':ep,'split':split,'state':state,'next':nxt,'front':front,'rear':rear,'action':action,'target':target,'features':features(state,front,rear,action),'time_s':float(first[1]),'dt_s':float(last[1]-first[1])})
    return episodes,samples

def scores(pred,target):
    pos=np.linalg.norm(pred[:,:2]-target[:,:2],axis=1)
    yaw=np.abs(wrap(pred[:,2]-target[:,2]))*180/np.pi
    moving=np.linalg.norm(target[:,:2],axis=1)>.2
    return {'samples':len(target),'position_mae_projected_cm':float(np.mean(pos)),'position_rmse_projected_cm':float(np.sqrt(np.mean(pos**2))),'heading_mae_deg':float(np.mean(yaw)),'moving_samples':int(moving.sum()),'moving_position_mae_projected_cm':float(np.mean(pos[moving])) if moving.any() else None}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--representation',choices=['marker','contact_proxy'],default='marker')
    args=parser.parse_args()
    ratio=0.0
    if args.representation=='contact_proxy':
        manifest=json.loads((ROOT/'data'/'manifest.json').read_text())
        if manifest['pusher'].get('same_contact_edge_all_clips_confirmed') is not True:
            raise ValueError('Contact-edge consistency must be confirmed first')
        ratio=manifest['pusher']['marker_to_contact_offset']/manifest['pusher']['marker_spacing']
        OUT=ROOT/'results'/'contact_proxy'
        OUT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(7); np.random.seed(7); torch.set_num_threads(2)
    episodes,samples=load_data(ratio)
    X=np.stack([s['features'] for s in samples]); Y=np.stack([s['target'] for s in samples])
    masks={split:np.array([s['split']==split for s in samples]) for split in ['train','validation','test']}
    xm=X[masks['train']].mean(0); xs=np.maximum(X[masks['train']].std(0),.03)
    ys=np.maximum(Y[masks['train']].std(0),[.1,.1,.03])
    Z=(X-xm)/xs; T=Y/ys
    ridge_candidates=[]
    for alpha in [.1,1.0,10.0,100.0]:
        model=Ridge(alpha=alpha).fit(Z[masks['train']],T[masks['train']])
        val=float(np.mean((model.predict(Z[masks['validation']])-T[masks['validation']])**2))
        ridge_candidates.append((val,alpha,model))
    ridge_val,alpha,ridge=min(ridge_candidates,key=lambda r:r[0])
    net=nn.Sequential(nn.Linear(X.shape[1],32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,3))
    optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001)
    xt=torch.tensor(Z[masks['train']],dtype=torch.float32); yt=torch.tensor(T[masks['train']],dtype=torch.float32)
    xv=torch.tensor(Z[masks['validation']],dtype=torch.float32); yv=torch.tensor(T[masks['validation']],dtype=torch.float32)
    best=float('inf'); best_epoch=0; best_weights=None; curve=[]
    for epoch in range(1,1501):
        net.train(); optimizer.zero_grad(); loss=((net(xt)-yt)**2).mean(); loss.backward(); optimizer.step()
        net.eval()
        with torch.no_grad(): vl=float(((net(xv)-yv)**2).mean())
        curve.append([epoch,float(loss.detach()),vl])
        if vl<best-1e-6: best=vl; best_epoch=epoch; best_weights=copy.deepcopy(net.state_dict())
        if epoch-best_epoch>=180: break
    net.load_state_dict(best_weights); net.eval()
    with torch.no_grad(): neural=net(torch.tensor(Z,dtype=torch.float32)).numpy()*ys
    predictions={'no_motion':np.zeros_like(Y),'copy_pusher':np.stack([np.r_[body(s['action'],s['state'][2]),0.] for s in samples]),'ridge':ridge.predict(Z)*ys,'small_mlp':neural}
    results={split:{name:scores(pred[mask],Y[mask]) for name,pred in predictions.items()} for split,mask in masks.items()}
    # Freeze selection on validation only; the test set is never used to choose it.
    selected='small_mlp' if best<ridge_val else 'ridge'
    np.savez(OUT/'ridge_model.npz',coef=ridge.coef_,intercept=ridge.intercept_,x_mean=xm,x_scale=xs,y_scale=ys)
    torch.save({'state_dict':best_weights,'x_mean':xm.tolist(),'x_scale':xs.tolist(),'y_scale':ys.tolist(),'input_dim':X.shape[1], 'representation':args.representation,'contact_offset_ratio':ratio},OUT/'small_mlp.pt')
    by_episode={}
    for ep in episodes:
        m=np.array([s['episode']==ep for s in samples])
        if m.any(): by_episode[ep]={name:scores(pred[m],Y[m]) for name,pred in predictions.items()}
    # Open-loop rollouts use recorded tool positions/actions; no future object states
    # are fed back after initialization. Missing markers split, rather than bridge, runs.
    rollout_results=[]; rollout_paths=[]
    for ep,record in episodes.items():
        if record['split']!='test': continue
        block=record['raw'][::STRIDE]
        valid=np.isfinite(block[:,11:18]).all(axis=1)
        edges=np.diff(np.r_[False,valid,False].astype(int))
        runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
        if not runs: continue
        start,end=max(runs,key=lambda r:r[1]-r[0]); block=block[start:end]
        if len(block)<3: continue
        obs=block[:,11:14]; p=block[:,14:16]; rear=block[:,16:18]
        paths={}
        for name in predictions:
            states=[obs[0].copy()]
            for k in range(len(obs)-1):
                st=states[-1]; action=p[k+1]-p[k]
                f=(features(st,p[k],rear[k],action)-xm)/xs
                if name=='no_motion': delta=np.zeros(3)
                elif name=='copy_pusher': delta=np.r_[body(action,st[2]),0.]
                elif name=='ridge': delta=ridge.predict(f[None,:])[0]*ys
                else:
                    with torch.no_grad(): delta=net(torch.tensor(f[None,:],dtype=torch.float32)).numpy()[0]*ys
                states.append(np.r_[st[:2]+world(delta[:2],st[2]),wrap(st[2]+delta[2])])
            path=np.array(states); paths[name]=path
            err=path[-1]-obs[-1]
            rollout_results.append({'episode':ep,'model':name,'duration_s':float(block[-1,1]-block[0,1]),'states':len(obs),'endpoint_error_projected_cm':float(np.linalg.norm(err[:2])),'endpoint_heading_error_deg':float(abs(wrap(err[2]))*180/np.pi)})
        rollout_paths.append((ep,obs,paths))
    report={'dataset':{'episodes':len(episodes),'transitions':len(samples),'split_episodes':{split:[ep for ep,e in episodes.items() if e['split']==split] for split in masks},'split_transitions':{split:int(mask.sum()) for split,mask in masks.items()},'transition_stride_frames':STRIDE},'training':{'ridge_alpha':alpha,'mlp_best_epoch':best_epoch,'mlp_parameters':sum(p.numel() for p in net.parameters()),'selection_normalized_validation_mse':{'ridge':ridge_val,'small_mlp':best},'validation_selected_model':selected},'one_step':results,'by_episode':by_episode,'open_loop_recorded_action_rollouts':rollout_results,'limitations':['Tiny four-push test set from the same recording sessions; no cross-session generalization claim.','Positions/headings derived from tabletop projection of raised roof markers; physical pose accuracy unvalidated.','Pusher movement is measured executed action, not force or exact contact-point motion.','Model selection used validation pushes; no test-driven tuning.','These are offline state predictions, not robot-control success rates.']}
    report['representation']={'name':args.representation,'contact_offset_ratio':ratio,'note':'Contact extrapolation is a local affine approximation. Original camera/roof-height bias remains. Changes after initial test inspection are development experiments.'}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2))
    np.savetxt(OUT/'learning_curve.csv',curve,delimiter=',',header='epoch,train_normalized_mse,validation_normalized_mse',comments='')
    np.savez_compressed(OUT/'transitions.npz',features=X,targets=Y,episode=np.array([s['episode'] for s in samples]),split=np.array([s['split'] for s in samples]))
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    labels=list(predictions); test=results['test']; colors=['#9ca3af','#64748b','#d97706','#0f766e']
    axes[0].bar(labels,[test[n]['position_mae_projected_cm'] for n in labels],color=colors)
    axes[0].set_ylabel('Mean position error (projected cm)'); axes[0].set_title('One-step translation error (~0.1 s)')
    axes[1].bar(labels,[test[n]['heading_mae_deg'] for n in labels],color=colors)
    axes[1].set_ylabel('Mean heading error (degrees)'); axes[1].set_title('One-step orientation error')
    for ax in axes: ax.tick_params(axis='x',rotation=20); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Preliminary results: 4 held-out pushes, same recording sessions')
    fig.tight_layout(); fig.savefig(OUT/'one_step_comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for ax,(ep,obs,paths) in zip(axes.flat,rollout_paths):
        ax.plot(obs[:,0],obs[:,1],'k-o',label='observed',markersize=3)
        for name,color in zip(labels,colors): ax.plot(paths[name][:,0],paths[name][:,1],label=name,color=color,linestyle='--')
        ax.set_title(ep); ax.set_aspect('equal',adjustable='datalim'); ax.set_xlabel('Projected x (cm)'); ax.set_ylabel('Projected y (cm)'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); fig.suptitle('Open-loop prediction with recorded tool actions\nObject predictions feed back into the model; no object-state correction')
    fig.tight_layout(); fig.savefig(OUT/'rollouts.png',dpi=150); plt.close(fig)
    print(json.dumps({'dataset':report['dataset'],'training':report['training'],'test':results['test'],'rollouts':rollout_results},indent=2))

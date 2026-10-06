"""A stronger geometric predictor: copy tool motion only near inward contact.

The rule uses current state and executed tool action, never future object motion.
Its tolerance is selected on validation pushes only. This is an exploratory
additional comparison after the original test set was first inspected.
"""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_baseline import load_data, body, scores
from goal_control import FrozenDynamics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'contact_baseline'
OUT.mkdir(parents=True,exist_ok=True)

def contact_prediction(sample,margin_cm):
    theta=sample['state'][2]
    relative=body(sample['front']-sample['state'][:2],theta)
    action=body(sample['action'],theta)
    distance=np.linalg.norm(np.maximum(np.abs(relative)-np.array([4.,1.75]),0.))
    inward=np.dot(relative,action)<0
    active=distance<=margin_cm and inward
    return np.r_[action if active else np.zeros(2),0.]

def main():
    episodes,samples=load_data(); target=np.stack([s['target'] for s in samples])
    masks={split:np.array([s['split']==split for s in samples]) for split in ['train','validation','test']}
    options=[]
    for margin in [.5,1.,1.5,2.,2.5,3.]:
        pred=np.stack([contact_prediction(s,margin) for s in samples])
        metric=scores(pred[masks['validation']],target[masks['validation']])
        options.append((metric['position_mae_projected_cm'],margin,pred))
    _,margin,pred=min(options,key=lambda item:item[0])
    model=FrozenDynamics()
    neural=[]
    for sample in samples:
        nxt=model.predict(sample['state'],sample['front'],sample['rear'],sample['action'])
        neural.append(np.r_[body(nxt[:2]-sample['state'][:2],sample['state'][2]),(nxt[2]-sample['state'][2]+np.pi)%(2*np.pi)-np.pi])
    predictions={'no_motion':np.zeros_like(target),'copy_pusher':np.stack([np.r_[body(s['action'],s['state'][2]),0.] for s in samples]),'contact_aware':pred,'small_mlp':np.array(neural)}
    metrics={split:{label:scores(values[mask],target[mask]) for label,values in predictions.items()} for split,mask in masks.items()}
    macro={}
    for split in masks:
        macro[split]={}
        for label,values in predictions.items():
            episode_scores=[]
            for ep,record in episodes.items():
                if record['split']!=split: continue
                mask=np.array([s['episode']==ep for s in samples])
                episode_scores.append(scores(values[mask],target[mask]))
            macro[split][label]={'episodes':len(episode_scores),'mean_episode_position_mae_projected_cm':float(np.mean([s['position_mae_projected_cm'] for s in episode_scores])),'mean_episode_heading_mae_deg':float(np.mean([s['heading_mae_deg'] for s in episode_scores]))}
    result={'rule':'If current pusher marker is within a tolerance of the oriented box and action points inward, copy tool translation; otherwise predict no movement. Heading stays constant.','chosen_margin_cm':margin,'margin_selection':'Lowest validation position MAE; candidates fixed at 0.5, 1, 1.5, 2, 2.5, 3 projected cm. No future object state is used by predictor.','validation_grid':[{'margin_cm':m,'position_mae_projected_cm':v} for v,m,_ in options],'metrics':metrics,'macro_episode_metrics':macro,'checkpoint_sha256':model.sha256,'evaluation_note':'Exploratory added baseline after initial held-out results were seen. Frozen neural weights; tolerance chosen on validation only. A fresh recording session is needed for a stronger final comparison.'}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    names=list(predictions); colors=['#9ca3af','#64748b','#d97706','#0f766e']; test=metrics['test']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,key,title in [(axes[0],'position_mae_projected_cm','All held-out transitions'),(axes[1],'moving_position_mae_projected_cm','Moving transitions only')]:
        ax.bar(names,[test[n][key] for n in names],color=colors); ax.set_title(title); ax.set_ylabel('Position MAE (projected cm)'); ax.tick_params(axis='x',rotation=20); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Contact-aware baseline: current geometry + known tool action\n4 held-out pushes; exploratory comparison, not physical pose accuracy')
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)
    print(json.dumps({'margin_cm':margin,'test':test,'macro_test':macro['test']},indent=2))

if __name__=='__main__': main()

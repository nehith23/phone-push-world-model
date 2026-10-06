"""Fixed old-only/plus-pilot comparison on stabilized development data."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from train_baseline import scores

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'stabilized'


def fit(data,include_pilot):
    name='plus_pilot' if include_pilot else 'old_only'
    path=OUT/name; path.mkdir(parents=True,exist_ok=True)
    X=data['features']; Y=data['targets']; split=data['split']
    pilot=np.array([e.startswith('v2_') for e in data['episode']])
    train=(split=='train') & (True if include_pilot else ~pilot)
    val=split=='validation'; test=split=='test'
    if min(train.sum(),val.sum(),test.sum())<1: raise ValueError('A split is empty after filtering')
    torch.manual_seed(7); np.random.seed(7); torch.set_num_threads(2)
    xm=X[train].mean(0); xs=np.maximum(X[train].std(0),.03)
    ys=np.maximum(Y[train].std(0),[.1,.1,.03]); Z=(X-xm)/xs; T=Y/ys
    choices=[]
    for alpha in [.1,1.,10.,100.]:
        ridge=Ridge(alpha=alpha).fit(Z[train],T[train])
        choices.append((float(np.mean((ridge.predict(Z[val])-T[val])**2)),alpha,ridge))
    ridge_loss,alpha,ridge=min(choices,key=lambda x:x[0])
    net=nn.Sequential(nn.Linear(12,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,3))
    optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001)
    x=torch.tensor(Z[train],dtype=torch.float32); y=torch.tensor(T[train],dtype=torch.float32)
    xv=torch.tensor(Z[val],dtype=torch.float32); yv=torch.tensor(T[val],dtype=torch.float32)
    best=float('inf'); best_epoch=0; weights=None; curve=[]
    for epoch in range(1,1501):
        optimizer.zero_grad(); loss=((net(x)-y)**2).mean(); loss.backward(); optimizer.step()
        with torch.no_grad(): vl=float(((net(xv)-yv)**2).mean())
        curve.append([epoch,float(loss.detach()),vl])
        if vl<best-1e-6: best=vl; best_epoch=epoch; weights=copy.deepcopy(net.state_dict())
        if epoch-best_epoch>=180: break
    net.load_state_dict(weights); net.eval()
    with torch.no_grad(): prediction=net(torch.tensor(Z,dtype=torch.float32)).numpy()*ys
    predictions={'no_motion':np.zeros_like(Y),'copy_contact':data['copy_targets'],'ridge':ridge.predict(Z)*ys,'small_mlp':prediction}
    metrics={label:{key:scores(p[m],Y[m]) for key,p in predictions.items()} for label,m in [('train',train),('validation',val),('reused_test',test),('pilot_inspected',pilot)]}
    metadata={'representation':'contact_proxy','contact_offset_ratio':9/40,'input_dim':12,
              'data_sha256':hashlib.sha256((OUT/'transitions.npz').read_bytes()).hexdigest(),'variant':name,
              'observation_note':'Per-frame table homography; no height correction. Reused development data.'}
    torch.save({'state_dict':weights,'x_mean':xm.tolist(),'x_scale':xs.tolist(),'y_scale':ys.tolist(),**metadata},path/'small_mlp.pt')
    np.savez(path/'ridge_model.npz',coef=ridge.coef_,intercept=ridge.intercept_,x_mean=xm,x_scale=xs,y_scale=ys)
    np.savetxt(path/'learning_curve.csv',curve,delimiter=',',header='epoch,train_normalized_mse,validation_normalized_mse',comments='')
    report={'status':'Development comparison; old test clips and pilot have been inspected. Pilot scores are in-sample for plus_pilot and inspected cross-session transfer for old_only.',
            'variant':name,'train_samples':int(train.sum()),'validation_samples':int(val.sum()),'reused_test_samples':int(test.sum()),'pilot_samples':int(pilot.sum()),
            'train_episodes':sorted(set(data['episode'][train].tolist())),
            'training':{'seed':7,'best_epoch':best_epoch,'parameters':1571,'ridge_alpha':alpha,
                        'normalized_validation_mse':{'ridge':ridge_loss,'small_mlp':best},'validation_selected_predictor':'small_mlp' if best<ridge_loss else 'ridge'},
            'metrics':metrics,'by_episode':{e:{key:scores(p[data['episode']==e],Y[data['episode']==e]) for key,p in predictions.items()} for e in sorted(set(data['episode']))},**metadata}
    (path/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['variant','train_samples','validation_samples','reused_test_samples','pilot_samples','training','metrics']},indent=2))


if __name__=='__main__':
    data=np.load(OUT/'transitions.npz')
    fit(data,False)
    fit(data,True)

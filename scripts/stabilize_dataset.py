"""Observed-corner camera compensation and explicit rejection; no interpolation."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract_tracks import colours, blobs, find_corners, map_xy, object_pair, pusher_pair
from train_baseline import body, contact_proxy, features, wrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'stabilized'


def observed_homography(yellow,pink,initial,width,depth,max_distance=40):
    found=[]
    for k,ref in enumerate(initial):
        pool=yellow if k in (0,2) else pink
        near=[point for point,area in pool if np.linalg.norm(point-ref)<max_distance]
        # Multiple contenders are ambiguity, not permission to guess a corner.
        if len(near)!=1: return None,None
        found.append(near[0])
    current=np.array(found,np.float32)
    if not cv2.isContourConvex(current): return None,None
    area=cv2.contourArea(current)/cv2.contourArea(initial)
    if not .75<=area<=1.25: return None,None
    target=np.array([[0,0],[width,0],[width,depth],[0,depth]],np.float32)
    return cv2.getPerspectiveTransform(current,target),current


def geometry_reason(mapped,tool_spacing=4.,roof_spacing=3.,tolerance=.25):
    if not np.isfinite(mapped).all(): return 'missing_marker'
    roof=np.linalg.norm(mapped[1]-mapped[0]); tool=np.linalg.norm(mapped[2]-mapped[3])
    if not roof_spacing*(1-tolerance)<=roof<=roof_spacing*(1+tolerance): return 'roof_spacing'
    if not tool_spacing*(1-tolerance)<=tool<=tool_spacing*(1+tolerance): return 'tool_spacing'
    return 'accepted'


def extract(source,episodes,measurement):
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise ValueError(f'Cannot read {source}')
    fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok,first=cap.read()
    if not ok: raise ValueError('Empty video')
    hsv,_,yellow,_=colours(first); pink=blobs(hsv,[140,80,130],[179,255,255])
    initial=find_corners(first,yellow,pink)
    w=measurement['workspace']['width_left_to_right']/10
    d=measurement['workspace']['depth_near_to_far']/10
    static=cv2.getPerspectiveTransform(initial,np.array([[0,0],[w,0],[w,d],[0,d]],np.float32))
    ratio=measurement['pusher']['front_marker_center_to_contact_edge']/measurement['pusher']['marker_center_spacing']
    blocks={e['id']:[] for e in episodes}; rows=[]; snapshots=[]; calibration=[]; stationary=[]
    midframes={round(np.mean(e['window_s'])*fps):e['id'] for e in episodes}
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    for index in range(count):
        ok,frame=cap.read()
        if not ok: break
        t=index/fps
        episode=next((e for e in episodes if e['window_s'][0]<=t<=e['window_s'][1]),None)
        # The pilot's first 5 seconds visibly contain a stationary cover.
        static_probe=source.name=='calibration_pilot_v2.mp4' and t<5
        if episode is None and not static_probe: continue
        hsv,blue,yellow,_=colours(frame); pink=blobs(hsv,[140,80,130],[179,255,255])
        H,current=observed_homography(yellow,pink,initial,w,d)
        b,y=object_pair(hsv,blue,yellow)
        if static_probe and H is not None and np.isfinite([b,y]).all():
            stationary.append([t,*map_xy([b,y],static).mean(0),*map_xy([b,y],H).mean(0)])
        if episode is None: continue
        pair=pusher_pair(pink,current if current is not None else initial,(b+y)/2,np.array(episode['screen_direction']))
        pixels=np.array([b,y,*pair])
        mapped=map_xy(pixels,H) if H is not None else np.full((4,2),np.nan)
        reason=geometry_reason(mapped) if H is not None else 'missing_or_ambiguous_corners'
        centre=mapped[:2].mean(0); angle=np.arctan2(*(mapped[1]-mapped[0])[::-1])
        tool_front=contact_proxy(mapped[2],mapped[3],ratio)
        rear=tool_front-(mapped[2]-mapped[3])
        value={'frame':index,'time_s':t,'episode':episode['id'],'split':episode['split'],
               'reason':reason,'state':np.r_[centre,angle], 'front':tool_front,'rear':rear}
        blocks[episode['id']].append(value)
        rows.append([index,t,episode['id'],episode['split'],reason,*centre,angle,*tool_front,*rear,
                     np.linalg.norm(mapped[1]-mapped[0]),np.linalg.norm(mapped[2]-mapped[3])])
        calibration.append([index,*(current.flatten() if current is not None else np.full(8,np.nan))])
        if index in midframes:
            for p,label,color in [(b,'A',(255,255,0)),(y,'B',(0,255,255)),(pair[0],'front',(255,0,255)),(pair[1],'rear',(255,0,255))]:
                if np.isfinite(p).all():
                    xy=tuple(p.astype(int)); cv2.circle(frame,xy,7,color,2)
                    cv2.putText(frame,label,(xy[0]+7,xy[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            if current is not None:
                for p in current: cv2.circle(frame,tuple(p.astype(int)),10,(0,255,0),2)
            cv2.putText(frame,f'{episode["id"]}: {reason}',(10,70),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2)
            snapshots.append(cv2.resize(frame,(512,288)))
    cap.release()
    transitions=[]; summary=[]
    for ep in episodes:
        block=blocks[ep['id']]; count_good=0
        for start in range(0,len(block)-3,3):
            seq=block[start:start+4]
            if any(r['reason']!='accepted' for r in seq): continue
            if any(seq[j+1]['frame']-seq[j]['frame']!=1 for j in range(3)): continue
            if np.max(np.linalg.norm(np.diff([r['state'][:2] for r in seq],axis=0),axis=1))>2: continue
            if np.max(np.linalg.norm(np.diff([r['front'] for r in seq],axis=0),axis=1))>3: continue
            a,z=seq[0],seq[-1]; state=a['state']; action=z['front']-a['front']
            target=np.r_[body(z['state'][:2]-state[:2],state[2]),wrap(z['state'][2]-state[2])]
            transitions.append({'episode':ep['id'],'split':ep['split'],'time_s':a['time_s'],
                                'start_frame':a['frame'],'end_frame':z['frame'],
                                'features':features(state,a['front'],a['rear'],action),'target':target,
                                'copy_target':np.r_[body(action,state[2]),0.]})
            count_good+=1
        reasons={r:sum(v['reason']==r for v in block) for r in sorted(set(v['reason'] for v in block))}
        summary.append({'episode':ep['id'],'split':ep['split'],'frames':len(block),'frame_status':reasons,'transitions':count_good})
    stem=source.stem.replace(' ','_')
    with (OUT/f'{stem}_frames.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['frame','time_s','episode','split','status','object_x_proxy_cm','object_y_proxy_cm','heading_rad','contact_x_proxy_cm','contact_y_proxy_cm','virtual_rear_x_cm','virtual_rear_y_cm','roof_spacing_proxy_cm','tool_spacing_proxy_cm']); writer.writerows(rows)
    np.savetxt(OUT/f'{stem}_corners.csv',calibration,delimiter=',',header='frame,near_left_x,near_left_y,near_right_x,near_right_y,far_right_x,far_right_y,far_left_x,far_left_y',comments='')
    if snapshots:
        if len(snapshots)%2: snapshots.append(np.zeros_like(snapshots[0]))
        cv2.imwrite(str(OUT/f'{stem}_preview.jpg'),np.vstack([np.hstack(snapshots[i:i+2]) for i in range(0,len(snapshots),2)]))
    probe=None
    if stationary:
        ar=np.array(stationary)
        probe={'frames':len(ar),'fixed_mapping_position_std_cm':np.std(ar[:,1:3],axis=0).tolist(),'dynamic_mapping_position_std_cm':np.std(ar[:,3:5],axis=0).tolist(),'note':'Stationary cover in first 5 seconds. This checks jitter, not metric accuracy or later motion compensation.'}
    return transitions,{'filename':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'fps':fps,'episodes':summary,'stationary_probe':probe}


def main(video_dir):
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'data'/'manifest.json').read_text()); pilot=json.loads((ROOT/'data'/'pilot_v2_manifest.json').read_text())
    measurement=json.loads((ROOT/'data'/'remeasured_setup.json').read_text())
    sources=[]
    for clip in old['videos']:
        episodes=[{'id':f'{clip["id"]}_{i:02}','split':'validation' if i==5 else ('test' if i==6 else 'train'),
                   'window_s':window,'screen_direction':clip['screen_push_direction']} for i,window in enumerate(clip['windows'],1)]
        sources.append((video_dir/clip['filename'],episodes))
    sources.append((video_dir/pilot['filename'],[{**e,'split':'train'} for e in pilot['episodes']]))
    samples=[]; reports=[]
    for source,episodes in sources:
        values,report=extract(source,episodes,measurement); samples.extend(values); reports.append(report)
        print(json.dumps({'source':source.name,'transitions':len(values),'episodes':report['episodes']}),flush=True)
    np.savez_compressed(OUT/'transitions.npz',features=np.stack([s['features'] for s in samples]),targets=np.stack([s['target'] for s in samples]),copy_targets=np.stack([s['copy_target'] for s in samples]),
                        episode=np.array([s['episode'] for s in samples]),split=np.array([s['split'] for s in samples]),time_s=np.array([s['time_s'] for s in samples]),start_frame=np.array([s['start_frame'] for s in samples]),end_frame=np.array([s['end_frame'] for s in samples]))
    counts={split:sum(s['split']==split for s in samples) for split in ['train','validation','test']}
    pilot_count=sum(s['episode'].startswith('v2_') for s in samples)
    report={'status':'Development data with per-frame observed table homographies; not calibrated 3-D poses.','measurements_snapshot':measurement,'sources':reports,'split_transitions':counts,'pilot_train_transitions':pilot_count,
            'limitations':['No held-out fresh final test session.','Table homography does not correct roof/tool height, tilt, object-centre alignment or different dynamics.','Broad spacing gates may discard directions unequally; inspect per-episode attrition.','Measured 9 mm offset applied retrospectively to original clips as an explicit assumption.','Four corner reprojection residuals are not independent validation of calibration.']}
    (OUT/'extraction.json').write_text(json.dumps(report,indent=2))
    episodes=[ep for r in reports for ep in r['episodes']]
    fig,ax=plt.subplots(figsize=(14,5))
    ax.bar(range(len(episodes)),[e['transitions'] for e in episodes],color=['#d97706' if e['episode'].startswith('v2_') else '#0f766e' for e in episodes])
    ax.set(xticks=range(len(episodes)),xticklabels=[e['episode'] for e in episodes],ylabel='Retained transitions',title='All episodes shown, including losses from camera and spacing filters')
    ax.tick_params(axis='x',rotation=75); fig.tight_layout(); fig.savefig(OUT/'retained_data.png',dpi=150); plt.close(fig)
    print(json.dumps({'split_transitions':counts,'pilot_train_transitions':pilot_count}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--video-dir',type=Path,required=True)
    main(parser.parse_args().video_dir)

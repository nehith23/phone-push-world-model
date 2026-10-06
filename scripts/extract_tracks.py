"""Extract observed marker centres, preserving misses and separating manual resets.

The metric columns are tabletop-projected proxies: roof height is NOT corrected.
They are useful for initial prediction QA, not ground-truth object/contact poses.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def blobs(hsv,low,high,pink=False):
    mask=cv2.inRange(hsv,np.array(low,np.uint8),np.array(high,np.uint8))
    if pink:
        mask|=cv2.inRange(hsv,np.array([0,85,130],np.uint8),np.array([3,255,255],np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for contour in contours:
        area=cv2.contourArea(contour); x,y,w,h=cv2.boundingRect(contour)
        if not(65<area<2000) or max(w,h)>3.5*min(w,h) or area/(w*h)<.4: continue
        m=cv2.moments(contour)
        out.append((np.array([m['m10']/m['m00'],m['m01']/m['m00']]),area))
    return out

def colours(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    return hsv,blobs(hsv,[85,85,75],[110,255,255]),blobs(hsv,[24,80,100],[43,255,255]),blobs(hsv,[140,80,130],[179,255,255],True)

def find_corners(frame,yellow,pink):
    h,w=frame.shape[:2]
    # Clockwise from near-left; world x right, world y away from camera.
    regions=[(0,.65,.3,1,yellow),(.7,.65,1,1,pink),(.6,0,1,.25,yellow),(0,0,.35,.25,pink)]
    pts=[]
    for x0,y0,x1,y1,candidates in regions:
        found=[(p,a) for p,a in candidates if x0*w<p[0]<x1*w and y0*h<p[1]<y1*h]
        if not found: raise ValueError('Corner marker not found in initial frame')
        pts.append(max(found,key=lambda item:item[1])[0])
    return np.array(pts,np.float32)

def map_xy(points,H):
    points=np.asarray(points,np.float64)
    q=np.column_stack([points,np.ones(len(points))])@H.T
    return q[:,:2]/q[:,2,None]

def object_pair(hsv,blue,yellow):
    orange=cv2.inRange(hsv,np.array([4,140,80],np.uint8),np.array([25,255,255],np.uint8))
    pairs=[]; h,w=hsv.shape[:2]
    for b,ba in blue:
        if not(.15*w<b[0]<.9*w and .09*h<b[1]<.87*h): continue
        for y,ya in yellow:
            if not(20<np.linalg.norm(y-b)<115 and .3<ba/ya<3): continue
            center=(b+y)/2
            x0,y0=np.maximum((center-[100,75]).astype(int),0)
            x1,y1=np.minimum((center+[100,75]).astype(int),[w,h])
            support=np.mean(orange[y0:y1,x0:x1]>0)
            if support>.11: pairs.append((support,b,y))
    if pairs:
        _,b,y=max(pairs,key=lambda item:item[0]); return b,y
    return np.full(2,np.nan),np.full(2,np.nan)

def pusher_pair(pink,corners,center,direction):
    if not np.isfinite(center).all(): return np.full((2,2),np.nan)
    candidates=[p for p,a in pink if np.min(np.linalg.norm(corners-p,axis=1))>32 and np.linalg.norm(p-center)<400 and np.dot(p-center,direction)<50]
    pairs=[]
    for p in candidates:
        for q in candidates:
            delta=p-q; length=np.linalg.norm(delta)
            if 22<length<115 and np.dot(delta,direction)/length>.70:
                pairs.append((np.linalg.norm(p-center),p,q))
    if pairs:
        _,front,rear=min(pairs,key=lambda item:item[0]); return np.array([front,rear])
    return np.full((2,2),np.nan)

def review(source,clip,workspace,out):
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise ValueError(f'Cannot open {source}')
    fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok,first=cap.read()
    if not ok: raise ValueError('Empty video')
    hsv,blue,yellow,pink=colours(first)
    corners=find_corners(first,yellow,pink)
    world=np.array([[0,0],[workspace['width'],0],[workspace['width'],workspace['depth']],[0,workspace['depth']]],np.float32)
    H=cv2.getPerspectiveTransform(corners,world)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    sample_ids={round((lo+hi)/2*fps):j for j,(lo,hi) in enumerate(clip['windows'],1)}
    rows=[]; previews=[]; corner_positions=[]
    for i in range(n):
        ok,frame=cap.read()
        if not ok: break
        t=i/fps
        episode=next((j for j,(lo,hi) in enumerate(clip['windows'],1) if lo<=t<=hi),0)
        hsv,blue,yellow,pink=colours(frame)
        b,y=object_pair(hsv,blue,yellow)
        center=(b+y)/2
        tool=pusher_pair(pink,corners,center,np.array(clip['screen_push_direction']))
        mapped=map_xy([b,y,*tool],H)
        mc=(mapped[0]+mapped[1])/2
        heading=np.arctan2(*(mapped[1]-mapped[0])[::-1])
        rows.append([i,t,episode,*b,*y,*tool[0],*tool[1],*mc,heading,*mapped[2],*mapped[3]])
        observed=[]
        for k,reference in enumerate(corners):
            pool=yellow if k in (0,2) else pink
            candidates=[p for p,a in pool if np.linalg.norm(p-reference)<15]
            observed.append(min(candidates,key=lambda p:np.linalg.norm(p-reference)) if candidates else np.full(2,np.nan))
        corner_positions.append(observed)
        if i in sample_ids:
            for point,label,color in [(b,'A',(255,255,0)),(y,'B',(0,255,255)),(tool[0],'tip',(255,0,255)),(tool[1],'rear',(255,0,255))]:
                if np.isfinite(point).all():
                    p=tuple(point.astype(int)); cv2.circle(frame,p,8,color,2)
                    cv2.putText(frame,label,(p[0]+10,p[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2,cv2.LINE_AA)
            cv2.putText(frame,f'{clip["id"]} / push {sample_ids[i]} / {t:.2f}s',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
            previews.append(cv2.resize(frame,(640,360)))
    cap.release()
    a=np.array(rows); summaries=[]
    for j,window in enumerate(clip['windows'],1):
        s=a[a[:,2]==j]
        roof=np.isfinite(s[:,3:7]).all(axis=1); tool=np.isfinite(s[:,7:11]).all(axis=1)
        good=s[roof]
        summaries.append({'episode':f'{clip["id"]}_{j:02}','window_s':window,'frames':len(s),'roof_pair_detected':int(roof.sum()),'tool_pair_detected':int(tool.sum()),'both_pairs_detected':int((roof&tool).sum()),'projected_displacement_cm':(good[-1,11:13]-good[0,11:13]).tolist() if len(good)>1 else None})
    header=['frame','time_s','episode','roof_a_px_x','roof_a_px_y','roof_b_px_x','roof_b_px_y','tool_front_px_x','tool_front_px_y','tool_rear_px_x','tool_rear_px_y','object_proxy_x_cm','object_proxy_y_cm','proxy_heading_rad','tool_front_proxy_x_cm','tool_front_proxy_y_cm','tool_rear_proxy_x_cm','tool_rear_proxy_y_cm']
    with (out/f'{clip["id"]}_tracks.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)
    # Pad the odd seven-push preview without inventing another observation.
    if len(previews)%2: previews.append(np.zeros_like(previews[0]))
    cv2.imwrite(str(out/f'{clip["id"]}_preview.jpg'),np.vstack([np.hstack(previews[k:k+2]) for k in range(0,len(previews),2)]))
    drift=np.linalg.norm(np.asarray(corner_positions)-corners,axis=2)
    summary={'id':clip['id'],'filename':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frames':len(rows),'fps':fps,'duration_s':len(rows)/fps,'resolution':[first.shape[1],first.shape[0]],'corner_pixels':corners.tolist(),'pixel_to_table_homography':H.tolist(),'corner_drift_px_p95':np.nanpercentile(drift,95,axis=0).tolist(),'episodes':summaries,'coordinates':'Tabletop-projected proxies. Roof height uncorrected. Counts are coverage, not labelled accuracy. Tool direction prior is used only for marker association.'}
    (out/f'{clip["id"]}_qa.json').write_text(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--video-dir',type=Path,required=True); parser.add_argument('--only',choices=['left','right','down','top'])
    args=parser.parse_args(); manifest=json.loads((ROOT/'data'/'manifest.json').read_text()); out=ROOT/'results'/'tracking'; out.mkdir(parents=True,exist_ok=True)
    for clip in manifest['videos']:
        if args.only and args.only!=clip['id']: continue
        result=review(args.video_dir/clip['filename'],clip,manifest['workspace'],out)
        print(json.dumps({'id':result['id'],'duration_s':result['duration_s'],'episodes':result['episodes']}))

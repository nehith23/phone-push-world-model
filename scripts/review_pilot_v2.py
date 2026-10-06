"""Inspect the new mixed-direction pilot without training or changing old results."""
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
from extract_tracks import colours, find_corners, map_xy, object_pair, pusher_pair, blobs
from train_baseline import contact_proxy, wrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'pilot_v2'


def percentiles(values):
    a=np.asarray(values)
    return np.nanpercentile(a,[10,50,90]).tolist() if np.isfinite(a).any() else None


def main(source):
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data'/'pilot_v2_manifest.json').read_text())
    measurement=json.loads((ROOT/'data'/'remeasured_setup.json').read_text())
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise ValueError(f'Cannot open {source}')
    fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok,frame=cap.read()
    if not ok: raise ValueError('Empty video')
    resolution=[frame.shape[1],frame.shape[0]]
    hsv,_,yellow,_=colours(frame)
    pink=blobs(hsv,[140,80,130],[179,255,255])
    corners=find_corners(frame,yellow,pink)
    w=measurement['workspace']['width_left_to_right']/10
    d=measurement['workspace']['depth_near_to_far']/10
    H=cv2.getPerspectiveTransform(corners,np.array([[0,0],[w,0],[w,d],[0,d]],np.float32))
    ratio=measurement['pusher']['front_marker_center_to_contact_edge']/measurement['pusher']['marker_center_spacing']
    rows=[]; drifts=[]; snapshots={}; sampleframes={}
    for i,e in enumerate(manifest['episodes'],1):
        sampleframes[round(np.mean(e['window_s'])*fps)]=i
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    for index in range(count):
        ok,frame=cap.read()
        if not ok: break
        t=index/fps
        hsv,blue,yellow,_=colours(frame)
        # On this taped pilot the original red-wrap fallback also detects
        # highlights on the orange cover. Magenta-only avoids that false pair.
        # This is an explicitly pilot-specific detector adjustment.
        pink=blobs(hsv,[140,80,130],[179,255,255])
        drift=[]
        for k,ref in enumerate(corners):
            pool=yellow if k in (0,2) else pink
            near=[point for point,area in pool if np.linalg.norm(point-ref)<20]
            drift.append(min(np.linalg.norm(point-ref) for point in near) if near else np.nan)
        drifts.append(drift)
        which=next((i for i,e in enumerate(manifest['episodes'],1) if e['window_s'][0]<=t<=e['window_s'][1]),0)
        if not which: continue
        episode=manifest['episodes'][which-1]
        b,y=object_pair(hsv,blue,yellow)
        pair=pusher_pair(pink,corners,(b+y)/2,np.array(episode['screen_direction']))
        mapped=map_xy([b,y,*pair],H)
        centre=(mapped[0]+mapped[1])/2
        heading=np.arctan2(*(mapped[1]-mapped[0])[::-1])
        contact=contact_proxy(mapped[2],mapped[3],ratio)
        rows.append([index,t,which,*b,*y,*pair[0],*pair[1],*centre,heading,*mapped[2],*mapped[3],*contact,np.linalg.norm(mapped[1]-mapped[0]),np.linalg.norm(mapped[2]-mapped[3])])
        if index in sampleframes:
            for point,label,color in [(b,'A',(255,255,0)),(y,'B',(0,255,255)),(pair[0],'front marker',(255,0,255)),(pair[1],'rear',(255,0,255))]:
                if np.isfinite(point).all():
                    xy=tuple(point.astype(int)); cv2.circle(frame,xy,7,color,2)
                    cv2.putText(frame,label,(xy[0]+7,xy[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
            for k,ref in enumerate(corners):
                cv2.circle(frame,tuple(ref.astype(int)),10,(0,255,0),2)
            cv2.putText(frame,f'{episode["id"]} | {t:.2f}s',(10,70),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            snapshots[which]=cv2.resize(frame,(640,360))
    cap.release()
    a=np.array(rows); drift=np.array(drifts); summaries=[]
    for i,e in enumerate(manifest['episodes'],1):
        block=a[a[:,2]==i]
        roof=np.isfinite(block[:,11:14]).all(1)
        tool=np.isfinite(block[:,14:18]).all(1)
        observed=block[roof]
        summaries.append({**e,'frames':len(block),'roof_pair_detected':int(roof.sum()),'tool_pair_detected':int(tool.sum()),'both_pairs_detected':int((roof&tool).sum()),
                          'roof_spacing_projected_cm_p10_median_p90':percentiles(block[:,20]),
                          'pusher_spacing_projected_cm_p10_median_p90':percentiles(block[:,21]),
                          'object_displacement_projected_cm':(observed[-1,11:13]-observed[0,11:13]).tolist() if len(observed)>1 else None,
                          'heading_change_deg':float(wrap(observed[-1,13]-observed[0,13])*180/np.pi) if len(observed)>1 else None})
    header=['frame','time_s','episode','roof_a_px_x','roof_a_px_y','roof_b_px_x','roof_b_px_y','tool_front_px_x','tool_front_px_y','tool_rear_px_x','tool_rear_px_y','object_proxy_x_cm','object_proxy_y_cm','proxy_heading_rad','tool_front_proxy_x_cm','tool_front_proxy_y_cm','tool_rear_proxy_x_cm','tool_rear_proxy_y_cm','contact_proxy_x_cm','contact_proxy_y_cm','roof_spacing_projected_cm','pusher_spacing_projected_cm']
    with (OUT/'tracks.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)
    complete=np.isfinite(a[:,11:18]).all(1)
    # Exactly the same three-frame stride and observation/jump criteria used
    # in the original extraction, applied independently within each episode.
    transitions=0
    for i in range(1,9):
        block=a[a[:,2]==i]
        for start in range(0,len(block)-3,3):
            seq=block[start:start+4]
            if not np.isfinite(seq[:,11:18]).all(): continue
            if np.max(np.linalg.norm(np.diff(seq[:,11:13],axis=0),axis=1))>2: continue
            if np.max(np.linalg.norm(np.diff(seq[:,14:16],axis=0),axis=1))>3: continue
            transitions+=1
    report={'source_filename':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'resolution':resolution,'fps':fps,'frames':len(drifts),'duration_s':len(drifts)/fps,
            'role':manifest['role'],'detector_revision':'Pilot-specific magenta hue 140-179; original red-wrap fallback disabled after visually finding a false front-marker association on the orange cover.','measurements_snapshot':measurement,'initial_corner_pixels':corners.tolist(),'pixel_to_table_homography':H.tolist(),
            'corner_detection_fraction':np.isfinite(drift).mean(0).tolist(),'corner_drift_px_p95':np.nanpercentile(drift,95,axis=0).tolist(),
            'review_frames':len(a),'both_pairs_detected':int(complete.sum()),'usable_three_frame_transitions':transitions,
            'episodes':summaries,'limitations':['Detector coverage is not hand-labelled accuracy.','All metric positions are tabletop-projected proxies; roof height and pusher height remain uncorrected.','Marker centres use colour-blob centroids, not the manually drawn centre crosses.','Manual windows include approach and withdrawal; no contact labels are inferred from future object motion.','No model has been trained on this pilot by this review script.']}
    (OUT/'qa.json').write_text(json.dumps(report,indent=2))
    tiles=[snapshots[i] for i in range(1,9)]
    cv2.imwrite(str(OUT/'tracking_preview.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)]))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for i,e in enumerate(manifest['episodes'],1):
        block=a[a[:,2]==i]
        axes[0].plot(block[:,1],block[:,21],'.',markersize=2,label=e['id'])
        axes[1].plot(block[:,1],block[:,20],'.',markersize=2)
    axes[0].axhline(4,color='black',linestyle='--',label='Measured: 4 cm')
    axes[1].axhline(3,color='black',linestyle='--',label='Measured: ~3 cm')
    axes[0].set_title('Caliper marker spacing'); axes[1].set_title('Roof marker spacing')
    for ax in axes:
        ax.set(xlabel='Video time (s)',ylabel='Tabletop-projected spacing (cm)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,ncol=2); axes[1].legend(fontsize=8)
    fig.suptitle('Pilot geometry checks: changing projected spacing is not calibrated motion')
    fig.tight_layout(); fig.savefig(OUT/'spacing_checks.png',dpi=150); plt.close(fig)
    table=['| Push | Both pairs / frames | Caliper spacing, median cm | Roof spacing, median cm | Heading change, degrees |','|---|---:|---:|---:|---:|']
    for e in summaries:
        table.append(f'| {e["id"]} | {e["both_pairs_detected"]}/{e["frames"]} | {e["pusher_spacing_projected_cm_p10_median_p90"][1]:.2f} | {e["roof_spacing_projected_cm_p10_median_p90"][1]:.2f} | {e["heading_change_deg"]:.1f} |')
    text=f'''# Calibration pilot v2: recording review

The uploaded clip contains eight pushes in the observed order left, left, top, top, right, right, down, down. It is {report['duration_s']:.2f} seconds at {fps:.2f} fps, {resolution[0]} x {resolution[1]} pixels. This review uses the remeasured 40 x 30 cm workspace and the 9/40 contact-offset ratio.

Both marker pairs were detected in **{int(complete.sum())}/{len(a)}** reviewed frames. The existing three-frame observation/jump filters retain **{transitions} transitions**. These are coverage counts, not proof of tracking accuracy or precise physical calibration. Resets are excluded by the saved manual windows.

Corner detection fractions: {np.round(report['corner_detection_fraction'],3).tolist()}. The 95th-percentile detected corner displacements from the first frame are {np.round(report['corner_drift_px_p95'],2).tolist()} pixels, ordered near-left, near-right, far-right, far-left. The corner search is local (20 pixel radius); coverage must be read alongside drift.

'''+ '\n'.join(table)+'''

![Tracked examples from all eight pushes](tracking_preview.jpg)

![Known marker-spacing checks](spacing_checks.png)

## Interpretation and limits

The camera sees the entire workspace and the recording includes useful object rotation. Some pushes turn the cover substantially; that is valid manipulation data, not automatically a bad attempt. Approximate table-plane scaling remains distinct from actual 3-D geometry. The roof markers are elevated 33 mm, pusher height is not yet provided, and the roof-marker midpoint has not been confirmed as the physical cover centre. This review does not correct those effects.

The detector follows the coloured sticker centroids. The new drawn centre marks help physical measurement and visual inspection but are not independently detected. The 9 mm offset is an affine estimate along the observed caliper-marker vector, not an exact reconstruction of contact.

The pilot-specific pink detector excludes the original red-hue fallback, which incorrectly selected a reddish cover highlight as a tool marker during visual review. The initial uncorrected QA snapshots are preserved in `initial_detector/`. Detector tuning on this pilot is another reason it is development data.

This pilot is development data. No updated model performance claim follows from this review, and it must not later be called an unseen final test set. Preserve the original experiments; inspect these observations before integrating more data.

## Reproduce

`python scripts/review_pilot_v2.py --source "C:\\path\\to\\calibration_pilot_v2.mp4"`

Windows and directions are in `data/pilot_v2_manifest.json`. Full observed tracks, source hash, geometry snapshot and coverage statistics are saved beside this report.
'''
    (OUT/'REPORT.md').write_text(text)
    print(json.dumps({k:report[k] for k in ['duration_s','resolution','review_frames','both_pairs_detected','usable_three_frame_transitions','corner_detection_fraction','corner_drift_px_p95','episodes']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    main(parser.parse_args().source)

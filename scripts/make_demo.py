"""Show original phone data alongside the physical replay, with explicit phase labels."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main(video_dir):
    out=ROOT/'results'/'demo'; out.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(out/'phone_to_panda.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,480))
    if not writer.isOpened(): raise RuntimeError('Could not open video writer')
    gif=[]; checkpoints=[]
    for name in ['left','right','down','top']:
        info=json.loads((ROOT/'results'/'simulation'/f'{name}_01_replay.json').read_text())
        lo,hi=info['source_selection']['source_time_s']
        original=cv2.VideoCapture(str(video_dir/f'{name} push.mp4'))
        simulation=cv2.VideoCapture(str(ROOT/'results'/'simulation'/f'{name}_01_replay.mp4'))
        if not original.isOpened() or not simulation.isOpened(): raise ValueError(f'Missing input for {name}')
        i=0
        while True:
            ok,right=simulation.read()
            if not ok: break
            t=(i+1)/10
            fraction=np.clip((t-3)/3,0,1)
            original.set(cv2.CAP_PROP_POS_MSEC,float((lo+(hi-lo)*fraction)*1000))
            ok,left=original.read()
            if not ok: raise ValueError(f'Cannot decode original at {lo+(hi-lo)*fraction}')
            panel=np.full((480,640,3),245,np.uint8)
            panel[78:438]=cv2.resize(left,(640,360))
            cv2.putText(panel,f'Your recording: {name} push',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),2,cv2.LINE_AA)
            phase='Robot positioning; source frame held' if t<3 else ('Recorded path replay (time stretched)' if t<=6 else 'End of push; source frame held')
            cv2.putText(panel,phase,(12,55),cv2.FONT_HERSHEY_SIMPLEX,.48,(65,65,65),1,cv2.LINE_AA)
            cv2.putText(panel,'Recorded marker motion supplies the robot tool trajectory.',(12,465),cv2.FONT_HERSHEY_SIMPLEX,.48,(65,65,65),1,cv2.LINE_AA)
            combined=np.hstack([panel,right]); writer.write(combined)
            if name=='left':
                gif.append(Image.fromarray(cv2.cvtColor(cv2.resize(combined,(960,360)),cv2.COLOR_BGR2RGB)))
            if i==44:
                checkpoints.append(combined)
                if name=='left': cv2.imwrite(str(out/'preview.jpg'),combined)
            i+=1
        original.release(); simulation.release()
    writer.release()
    gif[0].save(out/'left_push_demo.gif',save_all=True,append_images=gif[1:],duration=100,loop=0,optimize=True)
    cv2.imwrite(str(out/'all_directions_preview.jpg'),np.vstack(checkpoints))
    print(f'Saved demo to {out}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--video-dir',type=Path,required=True)
    main(parser.parse_args().video_dir)

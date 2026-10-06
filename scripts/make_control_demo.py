"""Side-by-side display of the first fixed evaluation case; all outcomes reported separately."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'goal_control'

def main():
    policies=['geometric','learned']; caps=[cv2.VideoCapture(str(OUT/f'eval_01_{policy}.mp4')) for policy in policies]
    records=[json.loads((OUT/f'eval_01_{policy}.json').read_text()) for policy in policies]
    if not all(cap.isOpened() for cap in caps): raise ValueError('Render both eval_01 controllers first')
    total=int(max(cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps))
    writer=cv2.VideoWriter(str(OUT/'comparison_demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,520))
    if not writer.isOpened(): raise RuntimeError('Cannot create demo')
    last=[None,None]; gif=[]
    for i in range(total):
        panels=[]
        for k,(cap,policy,record) in enumerate(zip(caps,policies,records)):
            ok,frame=cap.read()
            if ok: last[k]=frame
            elif last[k] is None: raise ValueError('Empty controller video')
            frame=last[k].copy()
            panel=np.full((520,640,3),245,np.uint8); panel[:480]=frame
            outcome='reached' if record['success'] else 'missed'
            label=f'Case 1 of 12 | final: {outcome}, {record["final_error_m"]*100:.2f} cm error'
            cv2.putText(panel,label,(12,506),cv2.FONT_HERSHEY_SIMPLEX,.48,(45,45,45),1,cv2.LINE_AA)
            panels.append(panel)
        combined=np.hstack(panels); writer.write(combined)
        gif.append(Image.fromarray(cv2.cvtColor(cv2.resize(combined,(960,390)),cv2.COLOR_BGR2RGB)))
    writer.release()
    for cap in caps: cap.release()
    gif[0].save(OUT/'comparison_demo.gif',save_all=True,append_images=gif[1:],duration=100,loop=0,optimize=True)
    cv2.imwrite(str(OUT/'comparison_final.jpg'),combined)
    print(f'Saved {total} frames of comparison video and GIF.')

if __name__=='__main__': main()

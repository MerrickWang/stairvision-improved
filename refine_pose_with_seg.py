import argparse,json,math
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
from transformers import AutoImageProcessor
from depth_model import EncodedDepthModel
from structure_postprocess import clean

def snap(point, contours, limit=50):
    best=None; bd=1e9
    for c in contours:
        for p in c.reshape(-1,2):
            d=math.hypot(float(p[0])-point[0],float(p[1])-point[1])
            if d<bd: bd=d; best=[float(p[0]),float(p[1]),1.0]
    return best if best is not None and bd<=limit else [point[0],point[1],point[2] if len(point)>2 else 1.0]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--images',required=True); p.add_argument('--pose',required=True); p.add_argument('--seg',required=True); p.add_argument('--depth',required=True); p.add_argument('--output',required=True); p.add_argument('--limit',type=int,default=20); p.add_argument('--depth-model',default='models/depth-anything-v2-small'); a=p.parse_args()
    dev='cuda' if torch.cuda.is_available() else 'cpu'; pose=YOLO(a.pose); seg=YOLO(a.seg)
    dm=EncodedDepthModel(a.depth_model).to(dev); dm.load_state_dict(torch.load(a.depth,map_location='cpu',weights_only=False)['model']); dm.eval()
    proc=AutoImageProcessor.from_pretrained(a.depth_model,local_files_only=True); rows=[]
    for path in sorted(Path(a.images).glob('*.jpg'))[:a.limit]:
        im=cv2.imread(str(path)); h,w=im.shape[:2]; pr=pose.predict(str(path),conf=.25,device=dev,verbose=False)[0]; sr=seg.predict(str(path),conf=.25,device=dev,verbose=False)[0]
        contours=[]
        if sr.masks is not None:
            for mask in sr.masks.data.cpu().numpy():
                contours.extend(cv2.findContours(cv2.resize((mask>.5).astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0])
        lines=[]
        if pr.keypoints is not None:
            for cls,conf,pts in zip(pr.boxes.cls.tolist(),pr.boxes.conf.tolist(),pr.keypoints.data.tolist()):
                lines.append({'class':int(cls),'confidence':float(conf),'endpoints':[snap(pts[0],contours),snap(pts[1],contours)]})
        structured=clean({'lines':lines},.25)
        pix=proc(images=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),return_tensors='pt')['pixel_values'].to(dev)
        with torch.inference_mode(): dep=dm(pix); dep=torch.nn.functional.interpolate(dep[:,None],(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        tread=[]; riser=[]
        if sr.masks is not None:
            for mask,cls in zip(sr.masks.data.cpu().numpy(),sr.boxes.cls.cpu().numpy().astype(int)):
                mask=cv2.resize((mask>.5).astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool); val=float(np.median(dep[mask])) if mask.any() else None
                (tread if cls==0 else riser).append(val)
        tv=[x for x in tread if x is not None]; rv=[x for x in riser if x is not None]
        rows.append({'image':str(path),'structured':structured,'tread_depth':float(np.median(tv)) if tv else None,'riser_depth':float(np.median(rv)) if rv else None,'depth_consistent':bool(tv and rv)})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,indent=2))
    print(json.dumps({'images':len(rows),'raw_lines':sum(len(x['structured']['lines']) for x in rows),'convex_total':sum(x['structured']['structured_convex_count'] for x in rows),'depth_consistent':sum(x['depth_consistent'] for x in rows),'output':str(out)}))
if __name__=='__main__': main()

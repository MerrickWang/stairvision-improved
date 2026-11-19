import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
from transformers import AutoImageProcessor
from depth_model import EncodedDepthModel

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True); p.add_argument('--pose',required=True); p.add_argument('--seg',required=True); p.add_argument('--depth',required=True); p.add_argument('--output',required=True)
    p.add_argument('--depth-model',default='models/depth-anything-v2-small')
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    pose=YOLO(a.pose); seg=YOLO(a.seg)
    depth=EncodedDepthModel(a.depth_model).to(dev); depth.load_state_dict(torch.load(a.depth,map_location='cpu',weights_only=False)['model']); depth.eval()
    proc=AutoImageProcessor.from_pretrained(a.depth_model,local_files_only=True)
    path=Path(a.source); image=cv2.imread(str(path)); h,w=image.shape[:2]
    pr=pose.predict(str(path),conf=.25,device=dev,verbose=False)[0]
    sr=seg.predict(str(path),conf=.25,device=dev,verbose=False)[0]
    canvas=image.copy(); tread=0; riser=0
    if sr.masks is not None:
        masks=sr.masks.data.cpu().numpy(); classes=sr.boxes.cls.cpu().numpy().astype(int); confs=sr.boxes.conf.cpu().numpy()
        for mask,cls,conf in zip(masks,classes,confs):
            mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)>0.5
            color=np.array([255,170,0],np.uint8) if cls==0 else np.array([0,0,255],np.uint8)
            canvas[mask]=(canvas[mask]*.35+color*.65).astype(np.uint8)
            if cls==0: tread+=1
            else: riser+=1
    lines=[]
    if pr.keypoints is not None:
        for cls,conf,pts in zip(pr.boxes.cls.tolist(),pr.boxes.conf.tolist(),pr.keypoints.data.tolist()):
            lines.append({'class':int(cls),'confidence':float(conf),'endpoints':pts})
            cv2.line(canvas,tuple(map(int,pts[0][:2])),tuple(map(int,pts[1][:2])),(255,0,0),2)
    pix=proc(images=cv2.cvtColor(image,cv2.COLOR_BGR2RGB),return_tensors='pt')['pixel_values'].to(dev)
    with torch.inference_mode(): d=depth(pix); d=torch.nn.functional.interpolate(d[:,None],(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    lo,hi=np.percentile(d,[2,98]); heat=np.uint8(np.clip((d-lo)/max(hi-lo,1e-6),0,1)*255)
    cv2.putText(canvas,f'YOLO-seg tread={tread} riser={riser}',(8,28),0,.65,(0,255,255),2)
    cv2.imwrite(str(out/f'{path.stem}_full.jpg'),canvas); cv2.imwrite(str(out/f'{path.stem}_depth.png'),cv2.applyColorMap(heat,cv2.COLORMAP_TURBO))
    (out/f'{path.stem}.json').write_text(json.dumps({'tread_instances':tread,'riser_instances':riser,'pose_lines':len(lines),'depth_kind':'encoded_depth_0_1_not_meters','lines':lines},indent=2))
    print(json.dumps({'tread':tread,'riser':riser,'pose_lines':len(lines),'output':str(out)}))
if __name__=='__main__': main()

import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor
from depth_model import EncodedDepthModel
from train_segmentation import SmallSegNet
from structure_postprocess import clean

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--yolo',required=True)
    p.add_argument('--seg',required=True)
    p.add_argument('--depth',required=True)
    p.add_argument('--depth-model',default='models/depth-anything-v2-small')
    p.add_argument('--output',default='runs/improved')
    a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    seg=SmallSegNet().to(dev); seg.load_state_dict(torch.load(a.seg,map_location='cpu',weights_only=False)['model']); seg.eval()
    depth=EncodedDepthModel(a.depth_model).to(dev); depth.load_state_dict(torch.load(a.depth,map_location='cpu',weights_only=False)['model']); depth.eval()
    proc=AutoImageProcessor.from_pretrained(a.depth_model,local_files_only=True)
    yolo=YOLO(a.yolo)
    files=[Path(a.source)] if Path(a.source).is_file() else sorted(Path(a.source).glob('*'))
    for path in files:
        if path.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
        image=cv2.imread(str(path)); h,w=image.shape[:2]
        pred=yolo.predict(source=str(path),conf=.25,device=dev,verbose=False)[0]
        lines=[]
        if pred.keypoints is not None:
            for cls,conf,pts in zip(pred.boxes.cls.tolist(),pred.boxes.conf.tolist(),pred.keypoints.data.tolist()):
                lines.append({'class':int(cls),'confidence':float(conf),'endpoints':pts})
        structured=clean({'lines':lines},.25)
        rgb=Image.open(path).convert('RGB')
        pix=proc(images=rgb,return_tensors='pt')['pixel_values'].to(dev)
        with torch.inference_mode():
            d=depth(pix)
            d=torch.nn.functional.interpolate(d[:,None],(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        small=cv2.resize(image,(256,256),interpolation=cv2.INTER_LINEAR)
        x=torch.from_numpy(small[:,:,::-1].copy()).permute(2,0,1).float().div(255).unsqueeze(0).to(dev)
        with torch.inference_mode(): lab=seg(x).argmax(1)[0].cpu().numpy()
        lab=cv2.resize(lab.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)
        overlay=image.copy()
        overlay[lab==1]=(overlay[lab==1]*.45+np.array([255,80,0])*.55).astype('uint8')
        overlay[lab==2]=(overlay[lab==2]*.45+np.array([0,0,255])*.55).astype('uint8')
        result=cv2.addWeighted(image,.45,overlay,.55,0)
        for line in structured['lines']:
            p1=tuple(map(int,line['endpoints'][0][:2])); p2=tuple(map(int,line['endpoints'][1][:2]))
            cv2.line(result,p1,p2,(255,0,0),2)
        cv2.putText(result,f"structured convex: {structured['structured_convex_count']}",(10,30),0,.8,(0,255,255),2)
        cv2.imwrite(str(out/f'{path.stem}_improved.jpg'),result)
        np.save(out/f'{path.stem}_depth.npy',d)
        (out/f'{path.stem}.json').write_text(json.dumps({'image':str(path),'structure':structured,'depth_kind':'encoded_depth_0_1_not_meters','segmentation_classes':['background','tread','riser']},indent=2))
        print(path.name,structured['structured_convex_count'],'convex')
if __name__=='__main__': main()

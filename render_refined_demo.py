import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
from transformers import AutoImageProcessor
from depth_model import EncodedDepthModel

def main():
 p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--refined',required=True); p.add_argument('--seg',required=True); p.add_argument('--depth',required=True); p.add_argument('--output',required=True); p.add_argument('--depth-model',default='models/depth-anything-v2-small'); a=p.parse_args()
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True); dev='cuda' if torch.cuda.is_available() else 'cpu'; im=cv2.imread(a.image); h,w=im.shape[:2]; seg=YOLO(a.seg); sr=seg.predict(a.image,conf=.25,device=dev,verbose=False)[0]; canvas=im.copy(); tread=riser=0
 if sr.masks is not None:
  for mask,cls in zip(sr.masks.data.cpu().numpy(),sr.boxes.cls.cpu().numpy().astype(int)):
   m=cv2.resize((mask>.5).astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool); color=np.array([255,170,0],np.uint8) if cls==0 else np.array([0,0,255],np.uint8); canvas[m]=(canvas[m]*.35+color*.65).astype(np.uint8); tread+=cls==0; riser+=cls==1
 rows=json.loads(Path(a.refined).read_text()); row=next(x for x in rows if Path(x['image']).name==Path(a.image).name)
 for line in row['structured']['lines']:
  p1=tuple(map(int,line['endpoints'][0][:2])); p2=tuple(map(int,line['endpoints'][1][:2])); cv2.line(canvas,p1,p2,(255,0,0),2)
  cv2.circle(canvas,p1,3,(0,255,255),-1); cv2.circle(canvas,p2,3,(0,255,255),-1)
 cv2.putText(canvas,f'refined convex={row["structured"]["structured_convex_count"]} tread={tread} riser={riser}',(8,28),0,.58,(0,255,255),2); cv2.imwrite(str(out/'final_demo.jpg'),canvas)
 dm=EncodedDepthModel(a.depth_model).to(dev); dm.load_state_dict(torch.load(a.depth,map_location='cpu',weights_only=False)['model']); dm.eval(); proc=AutoImageProcessor.from_pretrained(a.depth_model,local_files_only=True); pix=proc(images=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),return_tensors='pt')['pixel_values'].to(dev)
 with torch.inference_mode(): d=dm(pix); d=torch.nn.functional.interpolate(d[:,None],(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
 lo,hi=np.percentile(d,[2,98]); heat=np.uint8(np.clip((d-lo)/max(hi-lo,1e-6),0,1)*255); cv2.imwrite(str(out/'final_depth.png'),cv2.applyColorMap(heat,cv2.COLORMAP_TURBO)); print(json.dumps({'output':str(out),'convex':row['structured']['structured_convex_count'],'tread':tread,'riser':riser}))
if __name__=='__main__': main()

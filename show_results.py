from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch,cv2
from ultralytics import YOLO
from transformers import AutoImageProcessor
from depth_model import EncodedDepthModel
out=Path('runs/show_results');out.mkdir(exist_ok=True)
files=sorted(Path('data/prepared/images/val').glob('*'))
model=YOLO('runs/pose/earlystop_200/weights/best.pt')
processor=AutoImageProcessor.from_pretrained('models/depth-anything-v2-small',local_files_only=True)
depth=EncodedDepthModel('models/depth-anything-v2-small').cuda().eval()
ck=torch.load('runs/depth/earlystop_200/best.pt',map_location='cpu',weights_only=False)
depth.load_state_dict(ck['model'])
font=ImageFont.load_default(size=16)
small=ImageFont.load_default(size=12)
colors=[(20,240,100),(255,155,20)]
canvas=Image.new('RGB',(1536,4*438),(20,25,35));records=[]
for row,idx in enumerate([70,210,350,490]):
 path=files[idx];rgb=Image.open(path).convert('RGB');w,h=rgb.size
 gt=rgb.copy();gd=ImageDraw.Draw(gt)
 labels=Path('data/prepared/labels/val')/(path.stem+'.txt')
 truth=[list(map(float,x.split())) for x in labels.read_text().splitlines() if x.strip()]
 for r in truth:
  pts=[(r[5]*w,r[6]*h),(r[8]*w,r[9]*h)];gd.line(pts,fill=colors[int(r[0])],width=3)
 pred=model.predict(str(path),conf=.25,verbose=False,device=0)[0]
 overlay=rgb.copy();draw=ImageDraw.Draw(overlay)
 for cls,conf,pts in zip(pred.boxes.cls.tolist(),pred.boxes.conf.tolist(),pred.keypoints.xy.cpu().numpy()):
  xy=[tuple(map(float,p)) for p in pts];color=colors[int(cls)];draw.line(xy,fill=color,width=3)
  for x,y in xy: draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
  x,y=xy[0];draw.text((max(0,x),max(0,y-14)),f'{conf:.2f}',font=small,fill=color,stroke_width=1,stroke_fill=(0,0,0))
 pixels=processor(images=rgb,return_tensors='pt',size={'height':294,'width':294},keep_aspect_ratio=False)['pixel_values'].cuda()
 with torch.inference_mode():
  a=depth(pixels);a=torch.nn.functional.interpolate(a[:,None],(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
 np.save(out/(path.stem+'_depth.npy'),a)
 heat=cv2.applyColorMap(np.uint8(np.clip(a,0,1)*255),cv2.COLORMAP_TURBO)
 heat=Image.fromarray(cv2.cvtColor(heat,cv2.COLOR_BGR2RGB))
 strip=Image.new('RGB',(1536,438),(20,25,35));sd=ImageDraw.Draw(strip)
 titles=['RGB input','Ground-truth lines','YOLO prediction (conf >= 0.25)','Predicted encoded depth [0,1]']
 for col,(im,title) in enumerate(zip([rgb,gt,overlay,heat],titles)):
  strip.paste(im.resize((384,384)),(col*384,27));sd.text((col*384+8,5),title,font=font,fill='white')
 actual=sum(int(r[0])==0 for r in truth);detected=int((pred.boxes.cls==0).sum())
 sd.text((8,416),f'{path.name} | visible convex edges: GT {actual}, predicted {detected} | green: convex; orange: concave | depth is not meters',font=small,fill='white')
 strip.save(out/(path.stem+'_comparison.jpg'),quality=94);canvas.paste(strip,(0,row*438))
 records.append({'file':path.name,'validation_index':idx,'gt_convex':actual,'pred_convex':detected})
canvas.save(out/'comparison_grid.jpg',quality=95)
(out/'samples.json').write_text(json.dumps(records,indent=2));print(json.dumps(records),flush=True)

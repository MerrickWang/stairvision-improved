from pathlib import Path
import cv2, shutil
root=Path('data/raw/RGB-D stair dataset/RGB-D stair dataset')
out=Path('data/yolo_seg')
for split in ('train','val'):
    (out/'images'/split).mkdir(parents=True,exist_ok=True)
    (out/'labels'/split).mkdir(parents=True,exist_ok=True)
    for image in sorted((root/split/'images').glob('*.jpg')):
        target=out/'images'/split/image.name
        if not target.exists(): target.symlink_to(image.resolve())
        mask=cv2.imread(str(root/split/'segmentations'/(image.stem+'.png')),cv2.IMREAD_GRAYSCALE)
        h,w=mask.shape
        lines=[]
        for cls,value in ((0,128),(1,255)):
            binary=(mask==value).astype('uint8')
            contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area=cv2.contourArea(contour)
                if area < 40: continue
                eps=0.002*cv2.arcLength(contour,True)
                poly=cv2.approxPolyDP(contour,eps,True).reshape(-1,2)
                if len(poly)<3: continue
                coords=[]
                for x,y in poly: coords.extend((float(x)/w,float(y)/h))
                lines.append(str(cls)+' '+' '.join(f'{v:.6f}' for v in coords))
        (out/'labels'/split/(image.stem+'.txt')).write_text('\n'.join(lines)+'\n')
(out/'data.yaml').write_text('path: '+str(out.resolve())+'\ntrain: images/train\nval: images/val\nnames:\n  0: tread\n  1: riser\n')
print('converted',out)

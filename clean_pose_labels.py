from pathlib import Path
src=Path('data/prepared'); dst=Path('data/prepared_clean')
for split in ('train','val'):
    (dst/'images'/split).mkdir(parents=True,exist_ok=True); (dst/'labels'/split).mkdir(parents=True,exist_ok=True)
    for image in (src/'images'/split).iterdir():
        target=dst/'images'/split/image.name
        if not target.exists(): target.symlink_to(image.resolve())
    for label in (src/'labels'/split).glob('*.txt'):
        rows=[]
        for line in label.read_text().splitlines():
            p=line.split()
            if len(p)!=11: continue
            cls=int(p[0]); v=list(map(float,p[1:]))
            x1,y1,x2,y2=v[4],v[5],v[7],v[8]
            if ((x2-x1)**2+(y2-y1)**2)**0.5 < .02: continue
            if x1>x2: x1,y1,x2,y2=x2,y2,x1,y1
            v[4],v[5],v[7],v[8]=x1,y1,x2,y2
            rows.append((cls,(y1+y2)/2,v))
        rows.sort(key=lambda x:(x[0],x[1])); kept=[]
        for cls,y,v in rows:
            if kept and kept[-1][0]==cls and abs(y-kept[-1][1])<.008: continue
            kept.append((cls,y,v))
        (dst/'labels'/split/label.name).write_text('\n'.join(str(c)+' '+' '.join(f'{x:.8f}' for x in v) for c,_,v in kept)+'\n')
(dst/'stairs_pose.yaml').write_text(f'path: {dst.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: convex\n  1: concave\nkpt_shape:\n- 2\n- 3\nflip_idx:\n- 1\n- 0\n')
print('cleaned',dst)

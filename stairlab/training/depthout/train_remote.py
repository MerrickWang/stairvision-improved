"""Modern training adapter for upstream StairNet-DepthOut, commit 29457b1.

Keeps upstream architecture and dynamic multitask loss. Evaluates all tasks
every epoch, restores model.train(), and provides early stopping and resume.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from nets.NetV3 import StairNet_DepthOut
from loss import Stair_loss_Dynamic
from stairlab.training.depthout.depthout_data import StairPairs


def move(batch, device):
    if isinstance(batch,(list,tuple)):
        return tuple(move(x,device) for x in batch)
    return batch.to(device,non_blocking=True)


def proximity(pred, target):
    p = pred.reshape(pred.shape[0],2,2,*pred.shape[-2:])
    g = target.reshape(target.shape[0],2,2,*target.shape[-2:])
    dist = ((p[:,:,None]-g[:,None,:])**2).sum(3).sqrt().amin(2)
    return ((torch.exp(2*(1-dist).clamp_min(0))-1)/(math.exp(2)-1)).mean(1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval(); confusion=torch.zeros(3,3,device=device,dtype=torch.int64)
    tp=fp=fn=0; abserr=0.; depthn=0; losses=0.; n=0; xerror=yerror=0.
    for batch in loader:
        rgb,depth,blue,red,mask=move(batch,device)
        out=model(rgb);parts=criterion(out,blue,red,mask,depth)
        losses+=sum(x.item() for x in parts)*len(rgb);n+=len(rgb)
        for pc,pl,truth in ((out[0],out[1],blue),(out[2],out[3],red)):
            detected=pc[:,0]>=.5;positive=truth[0][:,0]>=.5
            match=proximity(pl,truth[1])>=.5
            hits=detected & positive & match
            tp+=int(hits.sum());fp+=int((detected & ~hits).sum());fn+=int((positive & ~hits).sum())
            err=(pl-truth[1]).abs()
            xerror+=float((err[:,0::2]*detected[:,None]).sum())
            yerror+=float((err[:,1::2]*detected[:,None]).sum())
        gt=mask.argmax(1);pred=out[4].argmax(1)
        confusion+=torch.bincount((gt*3+pred).flatten(),minlength=9).reshape(3,3)
        valid=(depth>0)&((rgb+1).abs().sum(1,keepdim=True)>1e-5)
        abserr+=float((out[5]-depth).abs()[valid].sum());depthn+=int(valid.sum())
    c=confusion.double();diag=c.diag();support=c.sum(1);union=c.sum(1)+c.sum(0)-diag
    mpa=float((diag/support.clamp_min(1)).mean());miou=float((diag/union.clamp_min(1)).mean())
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);f1=2*tp/max(2*tp+fp+fn,1)
    return {'val_loss':losses/n,'cell_precision':precision,'cell_recall':recall,'cell_f1':f1,
        'seg_pa':float(diag.sum()/c.sum()),'seg_mpa':mpa,'seg_miou':miou,
        'val_encoded_depth_mae':abserr/max(depthn,1),'score':f1+mpa,'x_error':xerror,'y_error':yerror}


def early_update(score,best,bad,delta):
    return (score,0) if score>best+delta else (best,bad+1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--width',type=float,default=.75);p.add_argument('--lr',type=float,default=.00025)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--patience',type=int,default=20)
    p.add_argument('--min-delta',type=float,default=.0001);p.add_argument('--cache',default='data/prepared_depthout')
    p.add_argument('--output',default='runs/depthout_200');p.add_argument('--resume')
    p.add_argument('--limit-train',type=int,default=0);p.add_argument('--limit-val',type=int,default=0)
    a=p.parse_args()
    if a.patience<1 or a.min_delta<0 or a.epochs<1 or a.batch<1:
        raise ValueError('Invalid training or early-stop configuration')
    root=Path(a.output)
    if (root/'last.pt').exists() and not a.resume:
        raise FileExistsError('Choose a new output directory or --resume')
    root.mkdir(parents=True,exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=DataLoader(StairPairs(Path(a.cache)/'train.jsonl',True,a.limit_train),batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=device.type=='cuda')
    val=DataLoader(StairPairs(Path(a.cache)/'val.jsonl',False,a.limit_val),batch_size=a.batch,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda')
    model=StairNet_DepthOut(width=a.width).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=.000001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=.5)
    start=0;best=-float('inf');stopbest=-float('inf');bad=0;lambdas=[8.,12.]
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False)
        if ck['args']['width']!=a.width:
            raise ValueError('Resume width mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        start=ck['epoch'];best=ck['best'];stopbest=ck['stop_best'];bad=ck['bad_epochs'];lambdas=ck['lambdas']
    if start>=a.epochs:
        raise ValueError('--epochs must exceed the checkpoint epoch')
    (root/'config.json').write_text(json.dumps(vars(a),indent=2))
    print(json.dumps({'device':str(device),'train':len(train.dataset),'val':len(val.dataset),'parameters':sum(p.numel() for p in model.parameters()),'grid':[64,32],'depth_unit':'encoded_0_1_not_meters','initialization':'resume' if a.resume else 'random'}),flush=True)
    (root/'status.json').write_text(json.dumps({'status':'running','start_epoch':start}))
    stopped=False
    for epoch in range(start,a.epochs):
        criterion=Stair_loss_Dynamic(*lambdas);model.train();summed=0.;n=0;t0=time.monotonic()
        for step,batch in enumerate(train,1):
            rgb,depth,blue,red,mask=move(batch,device)
            optimizer.zero_grad(set_to_none=True)
            out=model(rgb);parts=criterion(out,blue,red,mask,depth);loss=sum(parts)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite multitask loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            optimizer.step();summed+=loss.item()*len(rgb);n+=len(rgb)
            if step==1 or step%50==0:
                print(json.dumps({'epoch':epoch+1,'step':step,'steps':len(train),'loss':loss.item(),'gradient_norm':float(norm),'elapsed_s':round(time.monotonic()-t0,1)}),flush=True)
        metrics=validate(model,val,criterion,device)
        if not all(np.isfinite(x) for x in metrics.values()):
            raise FloatingPointError('Non-finite validation metric')
        improved=metrics['score']>best;best=max(best,metrics['score'])
        stopbest,bad=early_update(metrics['score'],stopbest,bad,a.min_delta)
        x,y=metrics.pop('x_error'),metrics.pop('y_error')
        if x>0 and y>0:
            adjustment=(x-y)/max(x,y)
            lambdas=[float(np.clip(lambdas[0]+adjustment,.5,19.5)),float(np.clip(lambdas[1]-adjustment,.5,19.5))]
        scheduler.step()
        row={'epoch':epoch+1,'train_loss':summed/n,**metrics,'lr':optimizer.param_groups[0]['lr'],'bad_epochs':bad,'epoch_s':round(time.monotonic()-t0,1)}
        print(json.dumps(row),flush=True)
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        ck={'epoch':epoch+1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'args':vars(a),'best':best,'stop_best':stopbest,'bad_epochs':bad,'lambdas':lambdas}
        torch.save(ck,root/'last.pt')
        if improved:
            torch.save(ck,root/'best.pt');torch.save(model.state_dict(),root/'best.pth')
        (root/'status.json').write_text(json.dumps({'status':'running','epoch':epoch+1,'best_score':best,'bad_epochs':bad}))
        if bad>=a.patience:
            stopped=True;print(json.dumps({'early_stopped':True,'epoch':epoch+1,'best_score':best}),flush=True);break
    (root/'status.json').write_text(json.dumps({'status':'early_stopped' if stopped else 'completed','epoch':epoch+1,'best_score':best}))


if __name__=='__main__':
    main()

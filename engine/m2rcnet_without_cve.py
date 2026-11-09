import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from datasets import build_datasets, make_loaders
from models.m2rcnet_without_cve import build_m2rcnet_without_cve, EXPECTED_PARAMETERS
from models.mbrc import FULL_MBRC
from losses.segmentation import DiceLoss
from losses.mbrc import compute_parts, binary_from_probability
from utils.checkpoint import save_json, save_weights, load_checkpoint
from utils.deterministic import seed_all
from utils.logger import append_history
from utils.metrics import calc_result
from .runner import config_and_flags
from .trainer import train_mbrc_epoch
from .validator import is_new_best, should_stop


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def status(run, phase, **extra):
    save_json(Path(run)/'status.json',{'phase':phase,'pid':os.getpid(),
        'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra})


@torch.no_grad()
def evaluate(model,loader,dice,device,label,complete=False,prediction_dir=None):
    model.eval();rows=[];losses=[]
    for names,image,gt in tqdm(loader,desc=label,total=len(loader),file=sys.stdout,ncols=110,mininterval=1.0):
        image=image.to(device,dtype=torch.float32);gt=gt.to(device,dtype=torch.float32)
        output=model(image)
        assert torch.equal(output['base_prob'],torch.sigmoid(output['z_base']))
        assert torch.equal(output['final_prob'],torch.sigmoid(output['z_final']))
        parts,base,final=compute_parts(output,gt,dice,FULL_MBRC)
        assert torch.isfinite(parts['total'])
        losses.extend([float(parts['total'])]*len(names))
        for j,name in enumerate(names):
            y=gt[j,0].cpu().numpy();p=final[j,0].cpu().numpy();b=base[j,0].cpu().numpy()
            if not np.isfinite(p).all() or not np.isfinite(b).all():raise RuntimeError('Nonfinite validation probability')
            bm=calc_result(b,y,thresh_value=None);fm=calc_result(p,y,thresh_value=None)
            row={'image_id':name,'base_dice':float(bm['dice']),'final_dice':float(fm['dice']),
                 'delta_final_minus_base':float(fm['dice']-bm['dice'])}
            if complete:
                row.update({'iou':float(fm['iou']),'se':float(fm['sen']),'sp':float(fm['spe']),
                            'auc':float(roc_auc_score(y.reshape(-1),p.reshape(-1))),'acc':float(fm['acc'])})
                binary=binary_from_probability(p);truth=y>0.5
                row.update({'TP':int((binary&truth).sum()),'FP':int((binary&~truth).sum()),
                            'FN':int((~binary&truth).sum()),'TN':int((~binary&~truth).sum())})
                if prediction_dir is not None:
                    np.savez_compressed(Path(prediction_dir)/(Path(name).stem+'.npz'),
                        base_prob=b,final_prob=p,base_mask=binary_from_probability(b).astype(np.uint8),
                        final_mask=binary.astype(np.uint8),gt=y)
            rows.append(row)
    assert len(rows)==len(loader.dataset) and len({r['image_id'] for r in rows})==len(rows)
    result={'loss':float(np.mean(losses)),'base_dice':float(np.mean([r['base_dice'] for r in rows])),
            'final_dice':float(np.mean([r['final_dice'] for r in rows])),'images':len(rows)}
    result['delta_final_minus_base']=result['final_dice']-result['base_dice']
    if complete:
        result.update({key:float(np.mean([r[key] for r in rows])) for key in ['iou','se','sp','auc','acc']})
        result['pixel_totals']={k:sum(r[k] for r in rows) for k in ['TP','FP','FN','TN']}
    return result,rows


def train(config_path,run,*,config=None):
    run=Path(run).resolve()
    config,_=config_and_flags(config if config is not None else config_path,'m2rcnet_wo_cve')
    seed=int(config['seed'])
    run.mkdir(parents=True,exist_ok=True)
    for name in ['metadata.json','history.jsonl','model_best.pth','finished.flag','test_result.json']:
        assert not (run/name).exists(),'Refuse existing training output: '+name
    seed_all(seed);device=torch.device('cuda:0')
    sets=build_datasets(config['dataset'],config['data_root'],use_cve=False,seed=seed)
    model=build_m2rcnet_without_cve(seed=seed).to(device)
    assert sum(p.numel() for p in model.parameters())==EXPECTED_PARAMETERS
    assert all(not any(t in k for t in ['caliber_head','vesselness_head']) and k.rsplit('.',1)[-1] not in ['a3','a5','a7'] for k in model.state_dict())
    dice=DiceLoss();optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    loaders=make_loaders(sets,batch_size=config['batch_size'],seed=seed)
    splits={s:d.names for s,d in sets.items()}
    metadata={'model':'M2RC-Net_wo_CVE','model_class':type(model).__module__+'.'+type(model).__name__,
        'dataset':config['dataset'],'data_root':str(Path(config['data_root']).resolve()),'map_root':None,
        'split':{s:len(d) for s,d in sets.items()},'seed':seed,'resolution':[224,224],'batch_size':2,
        'optimizer':'Adam','learning_rate':1e-4,'weight_decay':1e-3,'max_epochs':400,'patience':40,
        'parameters':EXPECTED_PARAMETERS,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'initialization':'random scratch; no existing segmentation checkpoint loaded',
        'initial_model_hash':state_hash(model),'initialization_comparison':{'performed_this_run':False,'result':None},
        'loss':'Base Dice + Final Dice + 0.1*balanced_action_loss + 0.1*boundary_BCE',
        'learned_loss_weights':False,'caliber_dataset':False,'CVE_heads':False,'CVE_alpha':False,
        'MBRC':'Multistage Bidirectional Residual Correction',
        'residual_identity':'z_base.detach() + delta_z','correction_feature_gradients':'connected',
        'selection':'strictly greater Val Final image-mean Otsu Dice','test_policy':'once after training, reload Val-selected best',
        'augmentation':False,'lr_scheduler':None,'AMP':False,'gradient_clipping':False,'gradient_accumulation':1,
        'strict_determinism':torch.are_deterministic_algorithms_enabled(),'config_file':str(Path(config_path).resolve()),
        'torch':torch.__version__,'torch_cuda':torch.version.cuda,'python':sys.version,'gpu':torch.cuda.get_device_name(0)}
    save_json(run/'metadata.json',metadata);save_json(run/'config.json',config)
    save_json(run/'split_manifest.json',splits)
    print('TRAINING_METADATA',json.dumps(metadata,ensure_ascii=False),flush=True)
    checkpoint=run/'model_best.pth';best_val=-1.0;best_epoch=0;started=time.time();last_epoch=0
    for epoch in range(1,config['max_epoch']+1):
        tick=time.time();status(run,'training',epoch=epoch,best_epoch=best_epoch,best_val_dice=best_val)
        train_loss=train_mbrc_epoch(model,optimizer,loaders['train'],dice,FULL_MBRC,epoch)
        if not np.isfinite(train_loss):raise RuntimeError('Nonfinite training loss')
        val,_=evaluate(model,loaders['val'],dice,device,f'VAL {epoch}/400')
        improved=is_new_best(val['final_dice'],best_val)
        if improved:
            best_val=val['final_dice'];best_epoch=epoch
            save_weights(checkpoint,{'model_state_dict':model.state_dict(),'best_epoch':epoch,
                'best_val_dice':best_val,'seed':seed,'variant':'m2rcnet_wo_cve'})
            save_json(run/'best_result.json',{'epoch':epoch,'val':val,'checkpoint':str(checkpoint)})
        record={'epoch':epoch,'train_loss':train_loss,'val_loss':val['loss'],
            'current_val_dice':val['final_dice'],'current_val_base_dice':val['base_dice'],'val':val,
            'saved_best_epoch':best_epoch,'saved_best_val_dice':best_val,'saved_best_test_dice':None,
            'saved_this_epoch':improved,'epoch_seconds':time.time()-tick,'lr':optimizer.param_groups[0]['lr']}
        append_history(run,record)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val['loss']:.6f} "
              f"current_val_dice={val['final_dice']:.9f} saved_best_epoch={best_epoch} "
              f"saved_best_val_dice={best_val:.9f} saved_best_test_dice=NA "
              f"saved_this_epoch={improved} epoch_seconds={record['epoch_seconds']:.1f}",flush=True)
        last_epoch=epoch
        if should_stop(epoch,best_epoch,config['patience']):break
    status(run,'reload_best',best_epoch=best_epoch,best_val_dice=best_val)
    loaded=load_checkpoint(checkpoint,model,map_location='cpu')
    assert loaded['variant']=='m2rcnet_wo_cve' and loaded['best_epoch']==best_epoch and loaded['seed']==seed
    del loaded
    reloaded,_=evaluate(model,loaders['val'],dice,device,'RELOAD VAL')
    assert reloaded['final_dice']==best_val,'Best Val changed after checkpoint reload'
    status(run,'testing',best_epoch=best_epoch)
    predictions=run/'predictions';predictions.mkdir()
    test,rows=evaluate(model,loaders['test'],dice,device,'TEST BEST',complete=True,prediction_dir=predictions)
    assert len(rows)==len(sets['test']) and {r['image_id'] for r in rows}==set(splits['test'])
    with (run/'test_per_image.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(run/'test_result.json',{'checkpoint':str(checkpoint),'best_epoch':best_epoch,**test,
        'binary_protocol':'per-image minmax -> uint8 -> Otsu -> calc_result',
        'auc_protocol':'per-image raw continuous final probability ROC-AUC, mean over test images',
        'aggregation':'image-wise arithmetic mean; pixel totals are additional counts'})
    summary={'model':'M2RC-Net_wo_CVE','seed':seed,'trained_epochs':last_epoch,
        'best_epoch':best_epoch,'best_val_dice':best_val,'best_val_base_dice':reloaded['base_dice'],
        'test':test,'final_test_dice':test['final_dice'],'checkpoint':str(checkpoint),
        'stop_reason':'patience40' if should_stop(last_epoch,best_epoch,40) else 'max400',
        'elapsed_seconds':time.time()-started,'predictions':str(predictions),'success':True}
    save_json(run/'summary.json',summary)
    for name in ['model_best.pth','metadata.json','history.jsonl','summary.json','test_result.json',
                 'test_per_image.csv','console.log']:
        assert (run/name).is_file(), 'Missing archive artifact: '+name
    status(run,'completed',exit_code=0,best_epoch=best_epoch,test_final_dice=test['final_dice'])
    (run/'finished.flag').write_text('training, best reload, test and result archive all succeeded\n',encoding='utf-8')
    print('FINAL_SUCCESS',json.dumps(summary,ensure_ascii=False),flush=True)
    return summary
